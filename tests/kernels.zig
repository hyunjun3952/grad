const std = @import("std");
const lib = @import("grad");
const Problem = lib.kernels.Problem;
const matmulNaive = lib.kernels.matmulNaive;
const matmulTiledWorkLen = lib.kernels.matmulTiledWorkLen;
const matmulTiled = lib.kernels.matmulTiled;
const TileConfig = lib.kernels.TileConfig;
const Parallel = lib.kernels.Parallel;
const splitFor = lib.kernels.splitFor;
const flushDenormals = lib.kernels.flushDenormals;
const matmulVector = lib.kernels.matmulVector;

fn expectTiledCombos(comptime T: type, comptime base: TileConfig, par: Parallel, p: Problem(T), transposed: [2][]const T, want: []const T, work: []T) !void {
    inline for (.{ .{ false, false }, .{ true, false }, .{ false, true }, .{ true, true } }) |flags| {
        const cfg = comptime blk: {
            var c = base;
            c.trans_a = flags[0];
            c.trans_b = flags[1];
            break :blk c;
        };
        @memset(p.out, 99);
        matmulTiled(T, cfg, par, .{ .m = p.m, .k = p.k, .n = p.n, .a = if (flags[0]) transposed[0] else p.a, .b = if (flags[1]) transposed[1] else p.b, .out = p.out }, work);
        try std.testing.expectEqualSlices(T, want, p.out);
    }
}

fn expectSameResult(comptime T: type, par: Parallel, m: usize, k: usize, n: usize) !void {
    const gpa = std.testing.allocator;
    const a = try gpa.alloc(T, m * k);
    defer gpa.free(a);
    const b = try gpa.alloc(T, k * n);
    defer gpa.free(b);
    const want = try gpa.alloc(T, m * n);
    defer gpa.free(want);
    const got = try gpa.alloc(T, m * n);
    defer gpa.free(got);

    for (a, 0..) |*x, idx| x.* = @as(T, @floatFromInt(idx % 7)) - 3;
    for (b, 0..) |*x, idx| x.* = @as(T, @floatFromInt(idx % 5)) - 2;

    const problem: Problem(T) = .{ .m = m, .k = k, .n = n, .a = a, .b = b, .out = got };
    matmulNaive(T, .{ .m = m, .k = k, .n = n, .a = a, .b = b, .out = want });

    @memset(got, 99);
    matmulVector(T, problem);
    try std.testing.expectEqualSlices(T, want, got);

    const at = try gpa.alloc(T, m * k);
    defer gpa.free(at);
    const bt = try gpa.alloc(T, k * n);
    defer gpa.free(bt);
    for (0..m) |i| {
        for (0..k) |p| at[p * m + i] = a[i * k + p];
    }
    for (0..k) |p| {
        for (0..n) |j| bt[j * k + p] = b[p * n + j];
    }

    const small: TileConfig = .{ .mr = 4, .nv = 2, .mc = 8, .kc = 5, .nc = 16 };
    const work_small = try gpa.alloc(T, matmulTiledWorkLen(T, small, par, m, k, n));
    defer gpa.free(work_small);
    try expectTiledCombos(T, small, par, problem, .{ at, bt }, want, work_small);

    const dflt: TileConfig = .{};
    const work_dflt = try gpa.alloc(T, matmulTiledWorkLen(T, dflt, par, m, k, n));
    defer gpa.free(work_dflt);
    try expectTiledCombos(T, dflt, par, problem, .{ at, bt }, want, work_dflt);
}

test "parallel tiled matmul matches in both split directions" {
    const par: Parallel = .{ .io = std.testing.io, .threads = 4 };
    const by_rows = splitFor(f32, .{}, par, 200, 193, 197);
    const by_cols = splitFor(f32, .{}, par, 150, 200, 260);
    try std.testing.expect(by_rows.by_rows and by_rows.parts > 1);
    try std.testing.expect(!by_cols.by_rows and by_cols.parts > 1);

    try expectSameResult(f32, par, 200, 193, 197);
    try expectSameResult(f32, par, 150, 200, 260);
    try expectSameResult(f64, par, 120, 150, 140);
}

test "flushDenormals makes subnormal results zero on the calling thread only" {
    if (@import("builtin").cpu.arch != .x86_64) return error.SkipZigTest;
    const Probe = struct {
        fn product(flush: bool, out: *f32) void {
            if (flush) flushDenormals();
            var a: f32 = 1e-30;
            var b: f32 = 1e-10;
            const pa: *volatile f32 = &a;
            const pb: *volatile f32 = &b;
            out.* = pa.* * pb.*;
        }
    };
    var plain: f32 = undefined;
    var flushed: f32 = undefined;
    const t1 = try std.Thread.spawn(.{}, Probe.product, .{ false, &plain });
    t1.join();
    const t2 = try std.Thread.spawn(.{}, Probe.product, .{ true, &flushed });
    t2.join();
    try std.testing.expect(plain != 0);
    try std.testing.expectEqual(@as(f32, 0), flushed);
}

test "worker pool survives many park and wake cycles" {
    const par: Parallel = .{ .io = std.testing.io, .threads = 6 };
    for (0..60) |i| try expectSameResult(f32, par, 110 + i % 7, 130, 120 + i % 5);
}

test "kernels agree, sizes not a multiple of the vector width" {
    const shapes = [_][3]usize{
        .{ 5, 7, 11 },   .{ 3, 1, 9 },   .{ 1, 4, 1 },  .{ 16, 16, 16 }, .{ 17, 33, 65 }, .{ 6, 5, 16 },
        .{ 12, 12, 32 }, .{ 13, 9, 40 }, .{ 7, 3, 17 }, .{ 24, 10, 48 }, .{ 50, 70, 90 },
    };
    for (shapes) |s| try expectSameResult(f32, .{}, s[0], s[1], s[2]);
    try expectSameResult(f64, .{}, 9, 5, 13);
    try expectSameResult(f64, .{}, 33, 21, 29);
}
