const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const autograd = @import("autograd.zig");
const optim = @import("optim.zig");
const Gpu = @import("gpu/gpu.zig").Gpu;
const Var = autograd.Var;
const Tape = autograd.Tape;

pub fn Linear(comptime n_in: usize, comptime n_out: usize) type {
    const W = Tensor(f32, &.{ n_in, n_out });
    const B = Tensor(f32, &.{n_out});

    return struct {
        const Self = @This();

        w: Var(W),
        b: Var(B),

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            const w = try W.empty(gpa);
            const std_dev = @sqrt(2.0 / @as(f32, n_in));
            for (w.data) |*v| v.* = rng.floatNorm(f32) * std_dev;
            return .{
                .w = .{ .value = w, .grad = try W.zeros(gpa), .needs_grad = true },
                .b = .{ .value = try B.zeros(gpa), .grad = try B.zeros(gpa), .needs_grad = true },
            };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.w.value.deinit(gpa);
            self.w.grad.deinit(gpa);
            self.b.value.deinit(gpa);
            self.b.grad.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            const w = try gpu.alloc(f32, W.len);
            const grad_w = try gpu.alloc(f32, W.len);
            const b = try gpu.alloc(f32, B.len);
            const grad_b = try gpu.alloc(f32, B.len);
            try gpu.upload(w, self.w.value.data);
            try gpu.upload(b, self.b.value.data);
            try gpu.fill(grad_w, 0);
            try gpu.fill(grad_b, 0);
            return .{
                .w = .{ .value = .{ .data = w }, .grad = .{ .data = grad_w }, .needs_grad = true },
                .b = .{ .value = .{ .data = b }, .grad = .{ .data = grad_b }, .needs_grad = true },
            };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(Tensor(f32, &.{ @TypeOf(x.value).shape[0], n_out })) {
            return tape.add(try tape.matmul(x, self.w), self.b);
        }

        pub fn params(self: Self) struct { Var(W), Var(B) } {
            return .{ self.w, self.b };
        }
    };
}

pub fn BatchNorm(comptime c: usize) type {
    const V = Tensor(f32, &.{c});

    return struct {
        const Self = @This();

        gamma: Var(V),
        beta: Var(V),
        mean: V,
        variance: V,

        pub fn init(gpa: Allocator) Allocator.Error!Self {
            const gamma = try V.full(gpa, 1);
            errdefer gamma.deinit(gpa);
            const grad_gamma = try V.zeros(gpa);
            errdefer grad_gamma.deinit(gpa);
            const beta = try V.zeros(gpa);
            errdefer beta.deinit(gpa);
            const grad_beta = try V.zeros(gpa);
            errdefer grad_beta.deinit(gpa);
            const mean = try V.zeros(gpa);
            errdefer mean.deinit(gpa);
            return .{
                .gamma = .{ .value = gamma, .grad = grad_gamma, .needs_grad = true },
                .beta = .{ .value = beta, .grad = grad_beta, .needs_grad = true },
                .mean = mean,
                .variance = try V.full(gpa, 1),
            };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.gamma.value.deinit(gpa);
            self.gamma.grad.deinit(gpa);
            self.beta.value.deinit(gpa);
            self.beta.grad.deinit(gpa);
            self.mean.deinit(gpa);
            self.variance.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            var out: Self = undefined;
            inline for (.{ "gamma", "beta" }) |name| {
                const value = try gpu.alloc(f32, c);
                const grad = try gpu.alloc(f32, c);
                try gpu.upload(value, @field(self, name).value.data);
                try gpu.fill(grad, 0);
                @field(out, name) = .{ .value = .{ .data = value }, .grad = .{ .data = grad }, .needs_grad = true };
            }
            inline for (.{ "mean", "variance" }) |name| {
                const statistic = try gpu.alloc(f32, c);
                try gpu.upload(statistic, @field(self, name).data);
                @field(out, name) = .{ .data = statistic };
            }
            return out;
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            return tape.batchNorm(x, self.gamma, self.beta, .{ .mean = self.mean, .variance = self.variance });
        }

        pub fn params(self: Self) struct { Var(V), Var(V) } {
            return .{ self.gamma, self.beta };
        }
    };
}

pub fn Conv2d(comptime c_in: usize, comptime c_out: usize, comptime k: usize, comptime opts: autograd.ConvOpts) type {
    const Inner = Linear(k * k * c_in, c_out);

    return struct {
        const Self = @This();

        inner: Inner,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            return .{ .inner = try Inner.init(gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.inner.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return .{ .inner = try self.inner.toGpu(gpu) };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(Tensor(f32, &.{ @TypeOf(x.value).shape[0], (@TypeOf(x.value).shape[1] + 2 * opts.pad - k) / opts.stride + 1, (@TypeOf(x.value).shape[2] + 2 * opts.pad - k) / opts.stride + 1, c_out })) {
            return tape.add(try tape.conv2d(x, self.inner.w, opts), self.inner.b);
        }

        pub fn params(self: Self) @TypeOf(self.inner.params()) {
            return self.inner.params();
        }
    };
}

const testing = std.testing;

test "forward is x*w + b and params exposes both parameters" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);
    const layer = try Linear(2, 3).init(gpa, prng.random());
    defer layer.deinit(gpa);
    @memcpy(layer.w.value.data, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    @memcpy(layer.b.value.data, &[_]f32{ 10, 20, 30 });

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var tape: Tape = .{ .gpa = arena.allocator() };
    const x = try tape.constant(try Tensor(f32, &.{ 2, 2 }).fromSlice(arena.allocator(), &.{ 1, 1, 2, 0 }));
    const y = try layer.forward(&tape, x);

    try testing.expectEqual(@as(usize, 3), @TypeOf(y.value).shape[1]);
    try testing.expectEqualSlices(f32, &.{ 15, 27, 39, 12, 24, 36 }, y.value.data);
    try testing.expectEqual(@as(usize, 2), layer.params().len);
}

test "initial weights are zero-mean with variance 2/n_in" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(2);
    const layer = try Linear(400, 100).init(gpa, prng.random());
    defer layer.deinit(gpa);

    var sum: f64 = 0;
    var sq: f64 = 0;
    for (layer.w.value.data) |v| {
        sum += v;
        sq += @as(f64, v) * v;
    }
    const n: f64 = @floatFromInt(layer.w.value.data.len);
    try testing.expectApproxEqAbs(@as(f64, 0), sum / n, 0.01);
    try testing.expectApproxEqRel(@as(f64, 2.0 / 400.0), sq / n, 0.05);
    for (layer.b.value.data) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "adam also trains the linear layer" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const layer = try Linear(2, 2).init(gpa, prng.random());
    defer layer.deinit(gpa);
    var adam = try optim.Adam.init(gpa, .{}, 0.1, layer.params());
    defer adam.deinit();

    const x = try Tensor(f32, &.{ 8, 2 }).fromSlice(gpa, &.{ 1, 0.5, 2, -1, 0.7, 1, 1.5, -0.3, -1, 0.5, -2, -1, -0.7, 1, -1.5, 0.3 });
    defer x.deinit(gpa);
    const labels = [_]u8{ 1, 1, 1, 1, 0, 0, 0, 0 };

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..60) |step| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var tape: Tape = .{ .gpa = arena.allocator() };

        try optim.zeroGrad(.{}, layer.params());
        const loss = try tape.crossEntropy(try layer.forward(&tape, try tape.constant(x)), &labels);
        try tape.backward(loss);
        try adam.step(.{}, layer.params());

        if (step == 0) first = loss.value.data[0];
        last = loss.value.data[0];
    }
    try testing.expect(last < first / 4);
}

test "a linear layer learns a separable problem" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const layer = try Linear(2, 2).init(gpa, prng.random());
    defer layer.deinit(gpa);

    const x = try Tensor(f32, &.{ 8, 2 }).fromSlice(gpa, &.{ 1, 0.5, 2, -1, 0.7, 1, 1.5, -0.3, -1, 0.5, -2, -1, -0.7, 1, -1.5, 0.3 });
    defer x.deinit(gpa);
    const labels = [_]u8{ 1, 1, 1, 1, 0, 0, 0, 0 };

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..200) |step| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var tape: Tape = .{ .gpa = arena.allocator() };

        try optim.zeroGrad(.{}, layer.params());
        const logits = try layer.forward(&tape, try tape.constant(x));
        const loss = try tape.crossEntropy(logits, &labels);
        try tape.backward(loss);
        try optim.sgd(.{}, 0.5, layer.params());

        if (step == 0) first = loss.value.data[0];
        last = loss.value.data[0];
    }
    try testing.expect(last < first / 4);
}
