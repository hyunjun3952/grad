const std = @import("std");
const Io = std.Io;

const grad = @import("grad");
const kernels = grad.kernels;
const Tensor = grad.Tensor;

const T = f32;
const Problem = kernels.Problem(T);
const Kernel = *const fn (Problem, []T) void;

fn wrap(comptime f: anytype) Kernel {
    return struct {
        fn call(p: Problem, _: []T) void {
            f(T, p);
        }
    }.call;
}

fn tiled(p: Problem, work: []T) void {
    kernels.matmulTiled(T, .{}, .{}, p, work);
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

        kernels.matmulNaive(T, .{ .m = n, .k = n, .n = n, .a = a, .b = b, .out = ref });

        const flops = 2.0 * std.math.pow(f64, @floatFromInt(n), 3);
        const reps: usize = @max(3, @as(usize, @intFromFloat(2e9 / flops)));

        var base_ms: f64 = 0;
        for (entries, 0..) |e, idx| {
            const work = try gpa.alignedAlloc(T, .@"64", e.work_len(n));
            defer gpa.free(work);
            const problem: Problem = .{ .m = n, .k = n, .n = n, .a = a, .b = b, .out = res };

            e.run(problem, work);

            var best_ns: i96 = std.math.maxInt(i96);
            for (0..reps) |_| {
                const t0 = Io.Clock.awake.now(io);
                e.run(problem, work);
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
                kernels.matmulTiled(T, .{}, par, .{ .m = s[0], .k = s[1], .n = s[2], .a = a, .b = b, .out = c }, work);
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
    const dev = grad.gpu.Gpu.init(gpa, .{ .step_bytes = 1 << 20 }) catch |err| switch (err) {
        error.DriverNotFound => return out.writeAll("no NVIDIA driver\n"),
        else => return err,
    };
    defer dev.deinit();
    const shapes = [_][3]usize{
        .{ 512, 512, 512 },
        .{ 2048, 2048, 2048 },
        .{ 4096, 4096, 4096 },
        .{ 512, 4096, 784 },
        .{ 784, 4096, 512 },
        .{ 512, 784, 4096 },
        .{ 65536, 64, 576 },
        .{ 65536, 128, 576 },
        .{ 576, 64, 65536 },
    };
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
            const shape: grad.gpu.MatmulShape = .{ .m = m, .n = n, .k = k };
            const opts: grad.gpu.MatmulOpts = .{ .trans_a = t[0], .trans_b = t[1] };
            const ns = try measureGpu(io, dev, grad.gpu.Gpu.matmul, .{ dev, a, b, c, shape, opts });
            try out.print("{d:>10.2}", .{2.0 * @as(f64, @floatFromInt(m * n * k)) * 20 / ns / 1000});
        }
        try out.writeAll("\n");
    }
    try out.writeAll("\n");
}

fn cudnnConvolutions(io: Io, gpa: std.mem.Allocator, out: *Io.Writer, dir: [:0]const u8) !void {
    const dev = grad.gpu.Gpu.init(gpa, .{ .step_bytes = 1 << 20 }) catch |err| switch (err) {
        error.DriverNotFound => return,
        else => return err,
    };
    defer dev.deinit();
    var lib = grad.gpu.cudnn.Cudnn.open(dir) catch |err| {
        try out.print("cuDNN could not be loaded from {s}: {s}\n", .{ dir, @errorName(err) });
        return;
    };
    defer lib.close();
    try out.print("cuDNN {d} convolutions, NHWC, best algorithm per shape (data only, no packing), best of 20\n{s:<30}{s:<10}{s:>10}{s:>10}{s:>10}\n", .{ lib.version(), "case", "type", "fwd", "dgrad", "wgrad" });
    const cases = [_]struct { name: []const u8, g: grad.gpu.cudnn.Geometry }{
        .{ .name = "stage 1: 64x32x32, 64->64", .g = .{ .n = 64, .h = 32, .w = 32, .c = 64, .k = 64, .r = 3, .pad = 1, .stride = 1 } },
        .{ .name = "stage 2: 64x16x16, 128->128", .g = .{ .n = 64, .h = 16, .w = 16, .c = 128, .k = 128, .r = 3, .pad = 1, .stride = 1 } },
        .{ .name = "stage 3: 64x8x8, 256->256", .g = .{ .n = 64, .h = 8, .w = 8, .c = 256, .k = 256, .r = 3, .pad = 1, .stride = 1 } },
    };
    for (cases) |case| {
        for ([_]grad.gpu.cudnn.DataType{ .float, .bfloat16 }) |data| {
            const g = case.g;
            const size: usize = if (data == .float) 4 else 2;
            const plan = try lib.plan(g, data, 1 << 30, false);
            defer lib.destroyPlan(plan);
            const x = try dev.alloc(u8, g.n * g.h * g.w * g.c * size);
            const y = try dev.alloc(u8, g.n * g.h * g.w * g.k * size);
            const w = try dev.alloc(u8, g.k * g.r * g.r * g.c * size);
            const workspace = try dev.alloc(u8, plan.workspaceBytes() + 256);
            const host = try gpa.alloc(u8, @max(x.len, y.len));
            defer gpa.free(host);
            if (data == .float) {
                for (std.mem.bytesAsSlice(f32, host)) |*v| v.* = 0.01;
            } else {
                for (std.mem.bytesAsSlice(u16, host)) |*v| v.* = 0x3C23;
            }
            try dev.upload(x, host[0..x.len]);
            try dev.upload(y, host[0..y.len]);
            try dev.upload(w, host[0..w.len]);

            const flops = 2.0 * @as(f64, @floatFromInt(g.n * g.h * g.w)) * @as(f64, @floatFromInt(g.r * g.r * g.c)) * @as(f64, @floatFromInt(g.k));
            var tflops: [3]f64 = undefined;
            const addr = struct {
                fn of(slice: []u8) u64 {
                    return @intFromPtr(slice.ptr);
                }
            }.of;
            for (0..3) |direction| {
                const ns = switch (direction) {
                    0 => try measureGpu(io, dev, grad.gpu.cudnn.Cudnn.forward, .{ &lib, plan, addr(x), addr(w), addr(y), addr(workspace), false }),
                    1 => try measureGpu(io, dev, grad.gpu.cudnn.Cudnn.backwardData, .{ &lib, plan, addr(w), addr(y), addr(x), addr(workspace), false }),
                    else => try measureGpu(io, dev, grad.gpu.cudnn.Cudnn.backwardFilter, .{ &lib, plan, addr(x), addr(y), addr(w), addr(workspace), false }),
                };
                tflops[direction] = flops * 20 / ns / 1000;
            }
            try out.print("{s:<30}{s:<10}{d:>10.2}{d:>10.2}{d:>10.2}\n", .{ case.name, @tagName(data), tflops[0], tflops[1], tflops[2] });
        }
    }
    try out.writeAll("\n");
}

fn implicitGemm(io: Io, gpa: std.mem.Allocator, out: *Io.Writer) !void {
    const dev = grad.gpu.Gpu.init(gpa, .{ .step_bytes = 2 << 30 }) catch |err| switch (err) {
        error.DriverNotFound => return,
        else => return err,
    };
    defer dev.deinit();
    const cases = [_]struct { name: []const u8, n: u32, side: u32, c_in: u32, c_out: u32 }{
        .{ .name = "stage 1: 64x32x32, 64->64", .n = 64, .side = 32, .c_in = 64, .c_out = 64 },
        .{ .name = "stage 2: 64x16x16, 128->128", .n = 64, .side = 16, .c_in = 128, .c_out = 128 },
        .{ .name = "stage 3: 64x8x8, 256->256", .n = 64, .side = 8, .c_in = 256, .c_out = 256 },
    };
    try out.print(
        "implicit-GEMM 3x3 convolution kernels, kernel only, best of 20 (bf16 in; TF32 tensor cores; strict f32 on the CUDA cores)\n{s:<30}{s:>14}{s:>14}{s:>14}{s:>14}{s:>14}{s:>14}\n",
        .{ "case", "bf16 forward", "bf16 dW", "tf32 forward", "tf32 dW", "f32 forward", "f32 dW" },
    );
    for (cases) |case| {
        const rows: usize = case.n * case.side * case.side;
        const kc: usize = 9 * case.c_in;
        const x = try dev.alloc(T, rows * case.c_in);
        const w = try dev.alloc(T, kc * case.c_out);
        const y = try dev.alloc(T, rows * case.c_out);
        const dw = try dev.alloc(T, kc * case.c_out);
        const host = try gpa.alloc(T, @max(x.len, y.len));
        defer gpa.free(host);
        var prng = std.Random.DefaultPrng.init(case.side);
        for (host) |*v| v.* = prng.random().float(T) - 0.5;
        try dev.upload(x, host[0..x.len]);
        try dev.upload(y, host[0..y.len]);
        try dev.upload(w, host[0..w.len]);

        const geo: grad.gpu.ConvGeometry = .{
            .n = case.n,
            .h = case.side,
            .w = case.side,
            .c = case.c_in,
            .k = 3,
            .stride = 1,
            .pad = 1,
            .out_h = case.side,
            .out_w = case.side,
            .positions = @intCast(rows),
            .c_out = case.c_out,
        };
        const x_half = (try dev.packMatrix(x, rows, case.c_in, .{})).data;
        const w_half = try dev.packMatrix(w, kc, case.c_out, .{});
        const dy_half = (try dev.packMatrix(y, rows, case.c_out, .{})).data;
        const flops = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(kc)) * @as(f64, @floatFromInt(case.c_out));
        var tflops: [2]f64 = undefined;
        for (0..2) |variant| {
            const ns = if (variant == 0)
                try measureGpu(io, dev, grad.gpu.Gpu.convImplicitForward, .{ dev, x_half, w_half, y, geo, false })
            else
                try measureGpu(io, dev, grad.gpu.Gpu.convImplicitWeightGrad, .{ dev, dy_half, x_half, dw, geo });
            tflops[variant] = flops * 20 / ns / 1000;
        }
        const positions_padded = grad.gpu.padTo(rows, .depth);
        const dy_t32 = try dev.transposeF32(y, case.c_out, rows, grad.gpu.padTo(case.c_out, .depth), positions_padded, null);
        // Rows: TF32 on tensor cores, then strict f32; columns: forward, weight gradient.
        var f32_rates: [2][2]f64 = undefined;
        for (0..2) |mode| {
            dev.tf32 = mode == 0;
            for (0..2) |variant| {
                const ns = if (variant == 0)
                    try measureGpu(io, dev, grad.gpu.Gpu.convF32Forward, .{ dev, x, w, y, geo, grad.gpu.Gpu.ConvF32ForwardOpts{} })
                else
                    try measureGpu(io, dev, grad.gpu.Gpu.convF32WeightGrad, .{ dev, dy_t32, x, dw, geo, null });
                f32_rates[mode][variant] = flops * 20 / ns / 1000;
            }
        }
        try out.print("{s:<30}{d:>14.2}{d:>14.2}{d:>14.2}{d:>14.2}{d:>14.2}{d:>14.2}\n", .{ case.name, tflops[0], tflops[1], f32_rates[0][0], f32_rates[0][1], f32_rates[1][0], f32_rates[1][1] });
    }
    try out.writeAll("\n");
}

fn gpuTensorCore(io: Io, gpa: std.mem.Allocator, out: *Io.Writer) !void {
    const dev = grad.gpu.Gpu.init(gpa, .{ .step_bytes = 1 << 30 }) catch |err| switch (err) {
        error.DriverNotFound => return out.writeAll("no NVIDIA driver\n"),
        else => return err,
    };
    defer dev.deinit();
    dev.tensor_core_min_work = 0;
    const cases = [_]struct { name: []const u8, m: usize, n: usize, k: usize, opts: grad.gpu.MatmulOpts }{
        .{ .name = "square 1024", .m = 1024, .n = 1024, .k = 1024, .opts = .{} },
        .{ .name = "square 4096", .m = 4096, .n = 4096, .k = 4096, .opts = .{} },
        .{ .name = "conv fwd 65536x64x576", .m = 65536, .n = 64, .k = 576, .opts = .{} },
        .{ .name = "conv dW  576x64x65536", .m = 576, .n = 64, .k = 65536, .opts = .{ .trans_a = true } },
        .{ .name = "conv dX  65536x576x64", .m = 65536, .n = 576, .k = 64, .opts = .{ .trans_b = true } },
        .{ .name = "mlp fwd  512x4096x784", .m = 512, .n = 4096, .k = 784, .opts = .{} },
    };
    try out.print(
        "matmul including operand packing, f32 vs bf16 tensor cores (best of 20)\n{s:<24}{s:>12}{s:>12}{s:>10}{s:>14}{s:>14}\n",
        .{ "case", "f32 TFLOPS", "bf16 TFLOPS", "speedup", "kernel only", "rms rel err" },
    );
    for (cases) |case| {
        const a = try dev.alloc(T, @max(case.m * case.k, case.k * case.n));
        const b = try dev.alloc(T, @max(case.m * case.k, case.k * case.n));
        const c32 = try dev.alloc(T, case.m * case.n);
        const c16 = try dev.alloc(T, case.m * case.n);
        const host = try gpa.alloc(T, a.len);
        defer gpa.free(host);
        var prng = std.Random.DefaultPrng.init(case.m + case.n);
        for (host) |*x| x.* = prng.random().float(T) * 2 - 1;
        try dev.upload(a, host);
        for (host) |*x| x.* = prng.random().float(T) * 2 - 1;
        try dev.upload(b, host);

        const flops = 2.0 * @as(f64, @floatFromInt(case.m)) * @as(f64, @floatFromInt(case.n)) * @as(f64, @floatFromInt(case.k));
        var tflops: [2]f64 = undefined;
        for (0..2) |variant| {
            dev.tensor_cores = variant == 1;
            const target = if (variant == 0) c32 else c16;
            const ns = try measureGpu(io, dev, grad.gpu.Gpu.matmul, .{ dev, a, b, target, grad.gpu.MatmulShape{ .m = case.m, .n = case.n, .k = case.k }, case.opts });
            tflops[variant] = flops * 20 / ns / 1000;
        }

        const scope = dev.scratchScope();
        const a_packed = try dev.packMatrix(a, case.m, case.k, .{ .rows_padded = grad.gpu.padTo(case.m, .rows), .cols_padded = grad.gpu.padTo(case.k, .depth), .transpose = case.opts.trans_a });
        const b_packed = try dev.packMatrix(b, case.k, case.n, .{ .rows_padded = grad.gpu.padTo(case.k, .depth), .cols_padded = grad.gpu.padTo(case.n, .columns), .transpose = case.opts.trans_b });
        const kernel_ns = try measureGpu(io, dev, grad.gpu.Gpu.matmulPacked, .{ dev, a_packed, b_packed, c16, grad.gpu.MatmulShape{ .m = case.m, .n = case.n, .k = case.k }, false, false });
        const kernel_tflops = flops * 20 / kernel_ns / 1000;
        scope.restore();

        const full = try gpa.alloc(T, case.m * case.n);
        defer gpa.free(full);
        const half = try gpa.alloc(T, case.m * case.n);
        defer gpa.free(half);
        try dev.download(full, c32);
        try dev.download(half, c16);
        var error_sum: f64 = 0;
        var value_sum: f64 = 0;
        for (full, half) |f, h| {
            error_sum += (f - h) * (f - h);
            value_sum += f * f;
        }
        const worst: f64 = @sqrt(error_sum / value_sum);
        try out.print("{s:<24}{d:>12.2}{d:>12.2}{d:>9.1}x{d:>14.2}{e:>14.2}\n", .{ case.name, tflops[0], tflops[1], tflops[1] / tflops[0], kernel_tflops, worst });
    }
    try out.writeAll("\n");
}
// Time since a clock reading, as a float.
fn nanoseconds(io: Io, since: Io.Timestamp) f64 {
    return @floatFromInt(Io.Clock.awake.now(io).nanoseconds - since.nanoseconds);
}

fn milliseconds(io: Io, since: Io.Timestamp) f64 {
    return nanoseconds(io, since) / 1e6;
}

fn measureGpu(io: Io, dev: *grad.gpu.Gpu, comptime operation: anytype, args: anytype) !f64 {
    try @call(.auto, operation, args);
    try dev.cu.sync();
    const started = Io.Clock.awake.now(io);
    for (0..20) |_| try @call(.auto, operation, args);
    try dev.cu.sync();
    return nanoseconds(io, started);
}

fn cpuOps(io: Io, gpa: std.mem.Allocator, out: *Io.Writer) !void {
    const par: kernels.Parallel = .{ .io = io, .threads = std.Thread.getCpuCount() catch 1 };
    const Tape = grad.autograd.Tape;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    try out.print("CPU ops of the MNIST CNN at batch 64, {d} threads (best of 20, ms)\n{s:<18} {s:>9} {s:>9}\n", .{ par.threads, "op", "forward", "backward" });

    const Conv1 = struct { x: Tensor(f32, &.{ 64, 28, 28, 1 }), w: Tensor(f32, &.{ 25, 16 }) };
    const Conv2 = struct { x: Tensor(f32, &.{ 64, 12, 12, 16 }), w: Tensor(f32, &.{ 400, 32 }) };
    const Act = Tensor(f32, &.{ 64, 24, 24, 16 });
    const Channels = Tensor(f32, &.{16});

    inline for (.{ .{ "conv 5x5 1->16 (image input)", Conv1, false }, .{ "conv 5x5 16->32", Conv2, true } }) |case| {
        var best_forward: f64 = 1e9;
        var best_backward: f64 = 1e9;
        for (0..20) |_| {
            _ = arena.reset(.retain_capacity);
            var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
            const input = try @FieldType(case[1], "x").empty(arena.allocator());
            const x = if (case[2]) try tape.leaf(input) else try tape.constant(input);
            const w = try tape.leaf(try @FieldType(case[1], "w").empty(arena.allocator()));
            @memset(x.value.data, 0.5);
            @memset(w.value.data, 0.25);
            const started = Io.Clock.awake.now(io);
            const y = try tape.conv2d(x, w, .{});
            best_forward = @min(best_forward, milliseconds(io, started));
            @memset(y.grad.data, 1);
            const back = Io.Clock.awake.now(io);
            try tape.nodes.items[0].run(&tape, tape.nodes.items[0].ctx);
            best_backward = @min(best_backward, milliseconds(io, back));
        }
        try out.print("{s:<18} {d:>9.3} {d:>9.3}\n", .{ case[0], best_forward, best_backward });
    }

    var best: [3][2]f64 = @splat(@splat(1e9));
    for (0..20) |_| {
        _ = arena.reset(.retain_capacity);
        var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true };
        const x = try tape.leaf(try Act.empty(arena.allocator()));
        @memset(x.value.data, 0.5);
        const gamma = try tape.leaf(try Channels.empty(arena.allocator()));
        const beta = try tape.leaf(try Channels.empty(arena.allocator()));
        @memset(gamma.value.data, 1);
        @memset(beta.value.data, 0);
        const mean = try Channels.empty(arena.allocator());
        const variance = try Channels.empty(arena.allocator());
        @memset(mean.data, 0);
        @memset(variance.data, 1);

        var started = Io.Clock.awake.now(io);
        const pooled = try tape.maxPool(x, .{});
        best[0][0] = @min(best[0][0], milliseconds(io, started));
        @memset(pooled.grad.data, 1);
        started = Io.Clock.awake.now(io);
        try tape.nodes.items[0].run(&tape, tape.nodes.items[0].ctx);
        best[0][1] = @min(best[0][1], milliseconds(io, started));

        started = Io.Clock.awake.now(io);
        const normalized = try tape.batchNormRelu(x, gamma, beta, .{ .mean = mean, .variance = variance });
        best[1][0] = @min(best[1][0], milliseconds(io, started));
        @memset(normalized.grad.data, 1);
        started = Io.Clock.awake.now(io);
        try tape.nodes.items[1].run(&tape, tape.nodes.items[1].ctx);
        best[1][1] = @min(best[1][1], milliseconds(io, started));

        started = Io.Clock.awake.now(io);
        const activated = try tape.relu(x);
        best[2][0] = @min(best[2][0], milliseconds(io, started));
        @memset(activated.grad.data, 1);
        started = Io.Clock.awake.now(io);
        try tape.nodes.items[2].run(&tape, tape.nodes.items[2].ctx);
        best[2][1] = @min(best[2][1], milliseconds(io, started));
    }
    inline for (.{ "maxPool 2x2", "batchNorm + relu", "relu" }, 0..) |name, i| try out.print("{s:<18} {d:>9.3} {d:>9.3}\n", .{ name, best[i][0], best[i][1] });
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
    if (args.len > 1 and std.mem.eql(u8, args[1], "cudnn")) {
        try cudnnConvolutions(io, gpa, out, if (args.len > 2) args[2] else "data/cudnn");
        try out.flush();
        return;
    }
    if (args.len > 1 and std.mem.eql(u8, args[1], "tc")) {
        try gpuTensorCore(io, gpa, out);
        try implicitGemm(io, gpa, out);
        try out.flush();
        return;
    }
    if (args.len > 1 and std.mem.eql(u8, args[1], "cpu")) {
        try cpuOps(io, gpa, out);
        try out.flush();
        return;
    }
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
