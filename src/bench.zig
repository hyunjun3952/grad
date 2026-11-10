const std = @import("std");
const Io = std.Io;

const grad = @import("grad");
const kernels = grad.kernels;
const Tensor = grad.Tensor;

const T = f32;
const Kernel = *const fn (usize, usize, usize, []const T, []const T, []T, []T) void;

fn wrap(comptime f: anytype) Kernel {
    return struct {
        fn call(m: usize, k: usize, n: usize, a: []const T, b: []const T, out: []T, work: []T) void {
            _ = work;
            f(T, m, k, n, a, b, out);
        }
    }.call;
}

fn tiled(m: usize, k: usize, n: usize, a: []const T, b: []const T, out: []T, work: []T) void {
    kernels.matmulTiled(T, .{}, .{}, m, k, n, a, b, out, work);
}

fn tiledWork(n: usize) usize {
    return kernels.matmulTiledWorkLen(T, .{}, .{}, n, n, n);
}

fn noWork(n: usize) usize {
    _ = n;
    return 0;
}

const Entry = struct { name: []const u8, run: Kernel, work_len: *const fn (usize) usize = noWork };
const entries = [_]Entry{
    .{ .name = "naive (ijk)", .run = wrap(kernels.matmulNaive) },
    .{ .name = "@Vector", .run = wrap(kernels.matmulVector) },
    .{ .name = "tiled + packed 6x2", .run = tiled, .work_len = tiledWork },
};

fn kernelTable(io: Io, gpa: std.mem.Allocator, out: *Io.Writer, rng: std.Random) !void {
    try out.print("f32 matmul, vector lanes = {d}\n\n", .{std.simd.suggestVectorLength(T) orelse 1});
    try out.print("{s:>5}  {s:<20} {s:>10} {s:>9} {s:>8}\n", .{ "size", "kernel", "best ms", "GFLOPS", "speedup" });

    for ([_]usize{ 64, 256, 512, 1024 }) |n| {
        const a = try gpa.alloc(T, n * n);
        defer gpa.free(a);
        const b = try gpa.alloc(T, n * n);
        defer gpa.free(b);
        const ref = try gpa.alloc(T, n * n);
        defer gpa.free(ref);
        const res = try gpa.alloc(T, n * n);
        defer gpa.free(res);
        for (a) |*x| x.* = rng.float(T) - 0.5;
        for (b) |*x| x.* = rng.float(T) - 0.5;

        kernels.matmulNaive(T, n, n, n, a, b, ref);

        const flops = 2.0 * std.math.pow(f64, @floatFromInt(n), 3);
        const reps: usize = @max(3, @as(usize, @intFromFloat(2e9 / flops)));

        var base_ms: f64 = 0;
        for (entries, 0..) |e, idx| {
            const work = try gpa.alignedAlloc(T, .@"64", e.work_len(n));
            defer gpa.free(work);

            e.run(n, n, n, a, b, res, work);

            var best_ns: i96 = std.math.maxInt(i96);
            for (0..reps) |_| {
                const t0 = Io.Clock.awake.now(io);
                e.run(n, n, n, a, b, res, work);
                const t1 = Io.Clock.awake.now(io);
                std.mem.doNotOptimizeAway(res.ptr);
                best_ns = @min(best_ns, t1.nanoseconds - t0.nanoseconds);
            }

            var max_err: T = 0;
            for (ref, res) |r, g| max_err = @max(max_err, @abs(r - g));
            if (max_err > 1e-3) {
                try out.print("MISMATCH in {s} at n={d}: max abs err {d}\n", .{ e.name, n, max_err });
                try out.flush();
                return error.KernelMismatch;
            }

            const ms = @as(f64, @floatFromInt(best_ns)) / 1e6;
            if (idx == 0) base_ms = ms;
            try out.print("{d:>5}  {s:<20} {d:>10.3} {d:>9.2} {d:>7.1}x\n", .{ n, e.name, ms, flops / (ms * 1e6), base_ms / ms });
        }
        try out.writeAll("\n");
        try out.flush();
    }
}

fn matmulMs(io: Io, alloc: std.mem.Allocator, arena: ?*std.heap.ArenaAllocator, ta: anytype, tb: anytype) !f64 {
    var best: i96 = std.math.maxInt(i96);
    for (0..200) |_| {
        if (arena) |ar| _ = ar.reset(.retain_capacity);
        const t0 = Io.Clock.awake.now(io);
        const c = try grad.matmul(alloc, .{}, ta, tb, .{});
        const t1 = Io.Clock.awake.now(io);
        std.mem.doNotOptimizeAway(c.data.ptr);
        if (arena == null) c.deinit(alloc);
        best = @min(best, t1.nanoseconds - t0.nanoseconds);
    }
    return @as(f64, @floatFromInt(best)) / 1e6;
}

fn matmulOverhead(io: Io, gpa: std.mem.Allocator, out: *Io.Writer, rng: std.Random) !void {
    const Mat = Tensor(T, &.{ 256, 256 });
    const ta = try Mat.empty(gpa);
    defer ta.deinit(gpa);
    const tb = try Mat.empty(gpa);
    defer tb.deinit(gpa);
    for (ta.data) |*x| x.* = rng.float(T) - 0.5;
    for (tb.data) |*x| x.* = rng.float(T) - 0.5;

    const flops_256 = 2.0 * 256 * 256 * 256;
    const gpa_ms = try matmulMs(io, gpa, null, ta, tb);
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const arena_ms = try matmulMs(io, arena.allocator(), &arena, ta, tb);
    try out.print("Tensor.matmul 256x256, gpa:          {d:.3} ms, {d:.2} GFLOPS\n", .{ gpa_ms, flops_256 / (gpa_ms * 1e6) });
    try out.print("Tensor.matmul 256x256, reused arena: {d:.3} ms, {d:.2} GFLOPS\n\n", .{ arena_ms, flops_256 / (arena_ms * 1e6) });
}

fn scaling(io: Io, gpa: std.mem.Allocator, out: *Io.Writer, rng: std.Random) !void {
    const shapes = [_][3]usize{ .{ 64, 784, 128 }, .{ 784, 64, 128 }, .{ 64, 128, 10 }, .{ 256, 256, 256 } };
    try out.print("thread scaling of tiled matmul (best of 200, us)\n{s:<16}", .{"m x k x n"});
    for ([_]usize{ 1, 2, 3, 4, 6 }) |t| try out.print("{d:>9}", .{t});
    try out.writeAll("   threads\n");
    for (shapes) |s| {
        const a = try gpa.alloc(T, s[0] * s[1]);
        defer gpa.free(a);
        const b = try gpa.alloc(T, s[1] * s[2]);
        defer gpa.free(b);
        const c = try gpa.alloc(T, s[0] * s[2]);
        defer gpa.free(c);
        for (a) |*v| v.* = rng.float(T);
        for (b) |*v| v.* = rng.float(T);
        try out.print("{d:>4}x{d:>4}x{d:>4} ", .{ s[0], s[1], s[2] });
        for ([_]usize{ 1, 2, 3, 4, 6 }) |t| {
            const par: kernels.Parallel = if (t > 1) .{ .io = io, .threads = t } else .{};
            const work = try gpa.alloc(T, kernels.matmulTiledWorkLen(T, .{}, par, s[0], s[1], s[2]));
            defer gpa.free(work);
            var best: i96 = std.math.maxInt(i96);
            for (0..200) |_| {
                const t0 = Io.Clock.awake.now(io);
                kernels.matmulTiled(T, .{}, par, s[0], s[1], s[2], a, b, c, work);
                best = @min(best, Io.Clock.awake.now(io).nanoseconds - t0.nanoseconds);
            }
            try out.print("{d:>9.1}", .{@as(f64, @floatFromInt(best)) / 1000.0});
        }
        try out.writeAll("\n");
    }
    try out.writeAll("\n");
}

const Lap = struct {
    io: Io,
    last: i96,

    fn lap(self: *Lap) i96 {
        const now = Io.Clock.awake.now(self.io).nanoseconds;
        defer self.last = now;
        return now - self.last;
    }
};

const batch = 64;
const n_in = 784;
const n_hidden = 128;
const n_out = 10;

fn stepProfile(io: Io, gpa: std.mem.Allocator, out: *Io.Writer, rng: std.Random, threads: usize) !void {
    const par: kernels.Parallel = if (threads > 1) .{ .io = io, .threads = threads } else .{};
    const Tape = grad.autograd.Tape;
    const names = [_][]const u8{ "linear1 (matmul+bias)", "relu", "linear2 (matmul+bias)", "cross-entropy" };

    const l1 = try grad.nn.Linear(n_in, n_hidden).init(gpa, rng);
    defer l1.deinit(gpa);
    const l2 = try grad.nn.Linear(n_hidden, n_out).init(gpa, rng);
    defer l2.deinit(gpa);
    const params = l1.params() ++ l2.params();
    const x = try Tensor(T, &.{ batch, n_in }).empty(gpa);
    defer x.deinit(gpa);
    for (x.data) |*v| v.* = rng.float(T);
    var labels: [batch]u8 = undefined;
    for (&labels) |*l| l.* = rng.uintLessThan(u8, n_out);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var fwd: [4]i96 = @splat(0);
    var bwd: [6]i96 = @splat(0);
    var zero: i96 = 0;
    var update: i96 = 0;
    var total: i96 = 0;
    var loss_value: T = 0;
    const warmup = 20;
    const reps = 300;

    for (0..warmup + reps) |it| {
        _ = arena.reset(.retain_capacity);
        var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
        var lap: Lap = .{ .io = io, .last = Io.Clock.awake.now(io).nanoseconds };
        const start = lap.last;
        var f: [4]i96 = undefined;
        var b: [6]i96 = undefined;

        try grad.optim.zeroGrad(par, params);
        const zeroed = lap.lap();
        const vx = try tape.constant(x);
        _ = lap.lap();

        const h0 = try l1.forward(&tape, vx);
        f[0] = lap.lap();
        const h1 = try tape.relu(h0);
        f[1] = lap.lap();
        const logits = try l2.forward(&tape, h1);
        f[2] = lap.lap();
        const loss = try tape.crossEntropy(logits, &labels);
        f[3] = lap.lap();

        loss.grad.data[0] = 1;
        var i = tape.nodes.items.len;
        while (i > 0) {
            i -= 1;
            const node = tape.nodes.items[i];
            try node.run(&tape, node.ctx);
            b[i] = lap.lap();
        }

        try grad.optim.sgd(par, 0.01, params);
        const upd = lap.lap();
        const elapsed = lap.last - start;

        if (it >= warmup) {
            for (0..4) |k| fwd[k] += f[k];
            for (0..6) |k| bwd[k] += b[k];
            zero += zeroed;
            update += upd;
            total += elapsed;
        }
        loss_value = loss.value.data[0];
    }

    const to_us = 1.0 / (1000.0 * reps);
    try out.print("training step, MLP {d}-{d}-{d}, batch {d}, {d} thread(s) (mean of {d} steps, last loss {d:.3})\n", .{ n_in, n_hidden, n_out, batch, threads, reps, loss_value });
    try out.print("{s:<22} {s:>10} {s:>10}\n", .{ "op", "fwd us", "bwd us" });
    const bwd_rows = [4]i96{ bwd[0] + bwd[1], bwd[2], bwd[3] + bwd[4], bwd[5] };
    for (names, 0..) |name, k| {
        try out.print("{s:<22} {d:>10.1} {d:>10.1}\n", .{ name, @as(f64, @floatFromInt(fwd[k])) * to_us, @as(f64, @floatFromInt(bwd_rows[k])) * to_us });
    }
    try out.print("{s:<22} {d:>10.1}\n", .{ "zero grad", @as(f64, @floatFromInt(zero)) * to_us });
    try out.print("{s:<22} {d:>10.1}\n", .{ "sgd update", @as(f64, @floatFromInt(update)) * to_us });
    try out.print("{s:<22} {d:>10.1}  ({d:.0} steps/s)\n", .{ "total step", @as(f64, @floatFromInt(total)) * to_us, 1e6 / (@as(f64, @floatFromInt(total)) * to_us) });
    var adam = try grad.optim.Adam.init(gpa, par, 0.001, params);
    defer adam.deinit();
    const adam_start = Io.Clock.awake.now(io).nanoseconds;
    for (0..reps) |_| try adam.step(par, params);
    const adam_ns = Io.Clock.awake.now(io).nanoseconds - adam_start;
    try out.print("{s:<22} {d:>10.1}  (in place of sgd)\n", .{ "adam update", @as(f64, @floatFromInt(adam_ns)) * to_us });
    const matmul_flops = 2.0 * batch * n_hidden * (n_in + n_out) * 3;
    try out.print("matmul work per step: {d:.1} MFLOP = {d:.0} us at 100 GFLOPS\n", .{ matmul_flops / 1e6, matmul_flops / 1e5 });
}

fn gpuMatmul(io: Io, gpa: std.mem.Allocator, out: *Io.Writer) !void {
    const dev = grad.gpu.Gpu.init(gpa, 1 << 20) catch |err| switch (err) {
        error.DriverNotFound => return out.writeAll("no NVIDIA driver\n"),
        else => return err,
    };
    defer dev.deinit();
    const shapes = [_][3]usize{ .{ 512, 512, 512 }, .{ 2048, 2048, 2048 }, .{ 4096, 4096, 4096 }, .{ 512, 4096, 784 }, .{ 784, 4096, 512 }, .{ 512, 784, 4096 } };
    try out.print("gpu sgemm (best of 20)\n{s:<18}{s:>10}{s:>10}{s:>10}   {s}\n", .{ "m x n x k", "nn", "nt", "tn", "TFLOPS" });
    for (shapes) |s| {
        const m, const n, const k = s;
        const a = try dev.alloc(T, @max(m * k, k * n));
        const b = try dev.alloc(T, @max(m * k, k * n));
        const c = try dev.alloc(T, m * n);
        try dev.fill(a, 0.5);
        try dev.fill(b, 0.25);
        try out.print("{d:>5}x{d:>5}x{d:>5}", .{ m, n, k });
        for ([_][2]bool{ .{ false, false }, .{ false, true }, .{ true, false } }) |t| {
            try dev.matmul(a, b, c, .{ .m = m, .n = n, .k = k }, .{ .trans_a = t[0], .trans_b = t[1] });
            try dev.cu.sync();
            const t0 = Io.Clock.awake.now(io);
            for (0..20) |_| try dev.matmul(a, b, c, .{ .m = m, .n = n, .k = k }, .{ .trans_a = t[0], .trans_b = t[1] });
            try dev.cu.sync();
            const ns: f64 = @floatFromInt(Io.Clock.awake.now(io).nanoseconds - t0.nanoseconds);
            try out.print("{d:>10.2}", .{2.0 * @as(f64, @floatFromInt(m * n * k)) * 20 / ns / 1000});
        }
        try out.writeAll("\n");
    }
    try out.writeAll("\n");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len > 1 and std.mem.eql(u8, args[1], "gpu")) {
        try gpuMatmul(io, gpa, out);
        try out.flush();
        return;
    }
    const only_step = args.len > 1 and std.mem.eql(u8, args[1], "step");
    if (!only_step) {
        try kernelTable(io, gpa, out, rng);
        try matmulOverhead(io, gpa, out, rng);
    }
    if (only_step) try scaling(io, gpa, out, rng);
    try stepProfile(io, gpa, out, rng, 1);
    try out.writeAll("\n");
    try stepProfile(io, gpa, out, rng, std.Thread.getCpuCount() catch 1);
    try out.flush();
}
