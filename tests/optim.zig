const std = @import("std");
const lib = @import("grad");
const kernels = lib.kernels;
const cosineLr = lib.optim.cosineLr;
const Adam = lib.optim.Adam;
const Ema = lib.optim.Ema;
const zeroGrad = lib.optim.zeroGrad;
const sgd = lib.optim.sgd;

const testing = std.testing;
const Tensor = lib.tensor.Tensor;
const Gpu = lib.gpu.Gpu;
const Var = lib.autograd.Var;

test "adam: first step moves by lr, converges on a quadratic" {
    const gpa = testing.allocator;
    const One = Tensor(f32, &.{1});
    const p: Var(One) = .{ .value = try One.fromSlice(gpa, &.{0}), .grad = try One.fromSlice(gpa, &.{1}), .needs_grad = true };
    defer p.value.deinit(gpa);
    defer p.grad.deinit(gpa);

    var first = try Adam.init(gpa, .{}, 0.1, .{p});
    defer first.deinit();
    try first.step(.{}, .{p});
    try testing.expectApproxEqAbs(@as(f32, -0.1), p.value.data[0], 1e-6);

    p.value.data[0] = 0;
    var adam = try Adam.init(gpa, .{}, 0.05, .{p});
    defer adam.deinit();
    for (0..600) |_| {
        p.grad.data[0] = 2 * (p.value.data[0] - 3);
        try adam.step(.{}, .{p});
    }
    try testing.expectApproxEqAbs(@as(f32, 3), p.value.data[0], 0.05);
}

test "adam gives identical weights with one thread and with worker threads" {
    const gpa = testing.allocator;
    const Big = Tensor(f32, &.{100003});
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();

    var single: Var(Big) = .{ .value = try Big.empty(gpa), .grad = try Big.empty(gpa), .needs_grad = true };
    defer single.value.deinit(gpa);
    defer single.grad.deinit(gpa);
    var multi: Var(Big) = .{ .value = try Big.empty(gpa), .grad = try Big.empty(gpa), .needs_grad = true };
    defer multi.value.deinit(gpa);
    defer multi.grad.deinit(gpa);
    for (single.value.data, multi.value.data) |*a, *b| {
        a.* = rng.float(f32);
        b.* = a.*;
    }

    var adam_single = try Adam.init(gpa, .{}, 0.01, .{single});
    defer adam_single.deinit();
    var adam_multi = try Adam.init(gpa, .{}, 0.01, .{multi});
    defer adam_multi.deinit();
    const par: kernels.Parallel = .{ .io = testing.io, .threads = 4 };

    for (0..5) |_| {
        for (single.grad.data, multi.grad.data) |*a, *b| {
            a.* = rng.float(f32) - 0.5;
            b.* = a.*;
        }
        try adam_single.step(.{}, .{single});
        try adam_multi.step(par, .{multi});
    }
    try testing.expectEqualSlices(f32, single.value.data, multi.value.data);
}

test "gpu sgd, adam and zeroGrad match the cpu versions" {
    const gpu = try Gpu.forTest(1 << 20);
    defer gpu.deinit();
    const gpa = testing.allocator;
    const Big = Tensor(f32, &.{50021});
    var prng = std.Random.DefaultPrng.init(4);
    const rng = prng.random();

    var host: [2]Var(Big) = undefined;
    var device: [2]Var(Big) = undefined;
    for (&host, &device) |*h, *d| {
        h.* = .{ .value = try Big.empty(gpa), .grad = try Big.empty(gpa), .needs_grad = true };
        for (h.value.data) |*v| v.* = rng.float(f32);
        d.* = .{ .value = .{ .data = try gpu.alloc(f32, Big.len) }, .grad = .{ .data = try gpu.alloc(f32, Big.len) }, .needs_grad = true };
        try gpu.upload(d.value.data, h.value.data);
    }
    defer for (host) |h| {
        h.value.deinit(gpa);
        h.grad.deinit(gpa);
    };

    const gpu_par: kernels.Parallel = .{ .gpu = gpu };
    var cpu_adam = try Adam.init(gpa, .{}, 0.01, .{host[1]});
    defer cpu_adam.deinit();
    var gpu_adam = try Adam.init(gpa, gpu_par, 0.01, .{device[1]});
    defer gpu_adam.deinit();
    cpu_adam.weight_decay = 0.1;
    gpu_adam.weight_decay = 0.1;

    for (0..5) |_| {
        for (host, device) |h, d| {
            for (h.grad.data) |*g| g.* = rng.float(f32) - 0.5;
            try gpu.upload(d.grad.data, h.grad.data);
        }
        try sgd(.{}, 0.1, .{host[0]});
        try sgd(gpu_par, 0.1, .{device[0]});
        try cpu_adam.step(.{}, .{host[1]});
        try gpu_adam.step(gpu_par, .{device[1]});
    }

    const back = try gpa.alloc(f32, Big.len);
    defer gpa.free(back);
    for (host, device) |h, d| {
        try gpu.download(back, d.value.data);
        for (h.value.data, back) |want, got| try testing.expectApproxEqAbs(want, got, 2e-5);
    }

    try zeroGrad(gpu_par, .{device[0]});
    try gpu.download(back, device[0].grad.data);
    for (back) |g| try testing.expectEqual(@as(f32, 0), g);
}

test "sgd moves weights against the gradient, also across parallel chunks, and zeroGrad clears" {
    const gpa = testing.allocator;
    const Small = Tensor(f32, &.{3});
    const Big = Tensor(f32, &.{100003});

    var small: Var(Small) = .{ .value = try Small.fromSlice(gpa, &.{ 1, 2, 3 }), .grad = try Small.fromSlice(gpa, &.{ 10, 20, 30 }), .needs_grad = true };
    defer small.value.deinit(gpa);
    defer small.grad.deinit(gpa);
    var big: Var(Big) = .{ .value = try Big.full(gpa, 1), .grad = try Big.empty(gpa), .needs_grad = true };
    defer big.value.deinit(gpa);
    defer big.grad.deinit(gpa);
    for (big.grad.data, 0..) |*g, i| g.* = @floatFromInt(i % 11);

    try sgd(.{ .io = testing.io, .threads = 4 }, 0.1, .{ small, big });

    try testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, small.value.data);
    for (big.value.data, 0..) |w, i| {
        try testing.expectApproxEqAbs(1.0 - 0.1 * @as(f32, @floatFromInt(i % 11)), w, 1e-6);
    }

    try zeroGrad(.{ .io = testing.io, .threads = 4 }, .{ small, big });
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, small.grad.data);
    for (big.grad.data) |g| try testing.expectEqual(@as(f32, 0), g);
}

test "adam weight decay shrinks weights even with zero gradient, and cosineLr decays to zero" {
    const gpa = testing.allocator;
    const One = Tensor(f32, &.{1});
    const p: Var(One) = .{ .value = try One.fromSlice(gpa, &.{1}), .grad = try One.fromSlice(gpa, &.{0}), .needs_grad = true };
    defer p.value.deinit(gpa);
    defer p.grad.deinit(gpa);
    var adam = try Adam.init(gpa, .{}, 0.1, .{p});
    defer adam.deinit();
    adam.weight_decay = 0.5;
    try adam.step(.{}, .{p});
    try testing.expectApproxEqAbs(@as(f32, 0.95), p.value.data[0], 1e-6);

    try testing.expectEqual(@as(f32, 0.01), cosineLr(0.01, 0, 100, 0));
    try testing.expectApproxEqAbs(@as(f32, 0.005), cosineLr(0.01, 50, 100, 0), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0), cosineLr(0.01, 100, 100, 0), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.0025), cosineLr(0.01, 0, 100, 4), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.01), cosineLr(0.01, 4, 100, 4), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.005), cosineLr(0.01, 54, 104, 4), 1e-7);
}

test "ema averages with a warm-up decay and swap exchanges live and shadow values, on the cpu and the gpu" {
    const gpa = testing.allocator;
    const gpu = Gpu.init(gpa, .{ .step_bytes = 1 << 20 }) catch |err| switch (err) {
        error.DriverNotFound => null,
        else => return err,
    };
    defer if (gpu) |dev| dev.deinit();
    const count = 70;
    var host: [count][5]f32 = undefined;
    var slices: [count][]f32 = undefined;
    var device: [count][]f32 = undefined;
    for (&host, &slices, 0..) |*h, *s, i| {
        for (h, 0..) |*v, j| v.* = @floatFromInt(i * 10 + j);
        s.* = h;
    }
    const modes = [_]?*Gpu{ null, gpu };
    for (modes) |maybe| {
        const par: kernels.Parallel = .{ .gpu = maybe };
        var live = slices;
        if (maybe) |dev| {
            for (&device, slices) |*d, s| {
                d.* = try dev.alloc(f32, s.len);
                try dev.upload(d.*, s);
            }
            live = device;
        } else {
            for (&live, 0..) |*s, i| s.* = try gpa.dupe(f32, slices[i]);
        }
        defer if (maybe == null) for (live) |s| gpa.free(s);

        var average = try Ema.init(gpa, par, live);
        defer average.deinit();
        average.decay = 0.5;
        for (live) |s| {
            if (maybe) |dev| try dev.fill(s, 8) else @memset(s, 8);
        }
        try average.update(par, 0);
        try average.swap(par);
        var seen: [5]f32 = undefined;
        for ([_]usize{ 0, 69 }) |i| {
            if (maybe) |dev| try dev.download(&seen, live[i]) else seen = live[i][0..5].*;
            for (seen, 0..) |got, j| {
                const original: f32 = @floatFromInt(i * 10 + j);
                const rate = 1.0 / 10.0;
                try testing.expectApproxEqAbs(rate * original + (1 - rate) * 8, got, 1e-5);
            }
        }
        try average.swap(par);
        for ([_]usize{ 0, 69 }) |i| {
            if (maybe) |dev| try dev.download(&seen, live[i]) else seen = live[i][0..5].*;
            for (seen) |got| try testing.expectEqual(@as(f32, 8), got);
        }
    }
}

test "gpu optimizers update more tensors than fit in one device table" {
    const gpu = try Gpu.forTest(1 << 20);
    defer gpu.deinit();
    const gpa = testing.allocator;
    const Small = Tensor(f32, &.{37});
    const count = 70;
    var prng = std.Random.DefaultPrng.init(6);
    const rng = prng.random();

    var host: [count]Var(Small) = undefined;
    var device: [count]Var(Small) = undefined;
    for (&host, &device) |*h, *d| {
        h.* = .{ .value = try Small.empty(gpa), .grad = try Small.empty(gpa), .needs_grad = true };
        for (h.value.data) |*v| v.* = rng.float(f32);
        d.* = .{ .value = .{ .data = try gpu.alloc(f32, Small.len) }, .grad = .{ .data = try gpu.alloc(f32, Small.len) }, .needs_grad = true };
        try gpu.upload(d.value.data, h.value.data);
    }
    defer for (host) |h| {
        h.value.deinit(gpa);
        h.grad.deinit(gpa);
    };

    const gpu_par: kernels.Parallel = .{ .gpu = gpu };
    var cpu_adam = try Adam.init(gpa, .{}, 0.01, host);
    defer cpu_adam.deinit();
    var gpu_adam = try Adam.init(gpa, gpu_par, 0.01, device);
    defer gpu_adam.deinit();
    for (0..3) |_| {
        for (host, device) |h, d| {
            for (h.grad.data) |*g| g.* = rng.float(f32) - 0.5;
            try gpu.upload(d.grad.data, h.grad.data);
        }
        try cpu_adam.step(.{}, host);
        try gpu_adam.step(gpu_par, device);
        try sgd(.{}, 0.1, host);
        try sgd(gpu_par, 0.1, device);
    }

    var back: [Small.len]f32 = undefined;
    for (host, device) |h, d| {
        try gpu.download(&back, d.value.data);
        for (h.value.data, back) |want, got| try testing.expectApproxEqAbs(want, got, 2e-5);
    }

    try zeroGrad(gpu_par, device);
    for (device) |d| {
        try gpu.download(&back, d.grad.data);
        for (back) |g| try testing.expectEqual(@as(f32, 0), g);
    }
}

test "an ema with decay zero copies the live weights into the shadow, which a checkpoint relies on" {
    const gpa = testing.allocator;
    var live = [_]f32{ 1, 2, 3 };
    var average = try Ema.init(gpa, .{}, .{@as([]f32, &live)});
    defer average.deinit();
    live = .{ 7, 8, 9 };
    try average.update(.{}, 100);
    try testing.expectEqualSlices(f32, &.{ 7, 8, 9 }, average.values);
}
