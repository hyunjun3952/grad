const std = @import("std");
const Io = std.Io;

const grad = @import("grad");
const kernels = grad.kernels;

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

fn blocked(comptime E: type, m: usize, k: usize, n: usize, a: []const E, b: []const E, out: []E) void {
    kernels.blockedWith(E, 4, 2, m, k, n, a, b, out);
}

fn tiled(m: usize, k: usize, n: usize, a: []const T, b: []const T, out: []T, work: []T) void {
    kernels.tiledWith(T, .{}, m, k, n, a, b, out, work);
}

fn tiledWork(n: usize) usize {
    return kernels.tiledWorkLenFor(T, .{}, n, n, n);
}

fn noWork(n: usize) usize {
    _ = n;
    return 0;
}

const Entry = struct { name: []const u8, run: Kernel, work_len: *const fn (usize) usize = noWork };
const entries = [_]Entry{
    .{ .name = "naive (ijk)", .run = wrap(kernels.naive) },
    .{ .name = "@Vector", .run = wrap(kernels.vec) },
    .{ .name = "register block 4x2", .run = wrap(blocked) },
    .{ .name = "tiled + packed 6x2", .run = tiled, .work_len = tiledWork },
};

fn matmulMs(io: Io, alloc: std.mem.Allocator, arena: ?*std.heap.ArenaAllocator, ta: anytype, tb: anytype) !f64 {
    var best: i96 = std.math.maxInt(i96);
    for (0..200) |_| {
        if (arena) |ar| _ = ar.reset(.retain_capacity);
        const t0 = Io.Clock.awake.now(io);
        const c = try grad.matmul(alloc, ta, tb, .{});
        const t1 = Io.Clock.awake.now(io);
        std.mem.doNotOptimizeAway(c.data.ptr);
        if (arena == null) c.deinit(alloc);
        best = @min(best, t1.nanoseconds - t0.nanoseconds);
    }
    return @as(f64, @floatFromInt(best)) / 1e6;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    try out.print("f32 matmul, vector lanes = {d}\n\n", .{std.simd.suggestVectorLength(T) orelse 1});
    try out.print("{s:>5}  {s:<20} {s:>10} {s:>9} {s:>8}\n", .{ "size", "kernel", "best ms", "GFLOPS", "speedup" });

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

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

        kernels.naive(T, n, n, n, a, b, ref);

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

    const Mat = grad.Tensor(T, &.{ 256, 256 });
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
    try out.print("Tensor.matmul 256x256, reused arena: {d:.3} ms, {d:.2} GFLOPS\n", .{ arena_ms, flops_256 / (arena_ms * 1e6) });
    try out.flush();
}
