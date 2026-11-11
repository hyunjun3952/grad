const std = @import("std");
const Allocator = std.mem.Allocator;
const kernels = @import("../cpu/kernels.zig");

pub fn Tensor(comptime T: type, comptime dims: []const usize) type {
    comptime {
        for (dims) |d| {
            if (d == 0) @compileError("zero-sized dimensions are not supported");
        }
    }

    return struct {
        const Self = @This();

        pub const Elem = T;
        pub const shape = dims;
        pub const rank = dims.len;
        pub const len = blk: {
            var n: usize = 1;
            for (dims) |d| n *= d;
            break :blk n;
        };

        data: []T,

        pub fn empty(gpa: Allocator) Allocator.Error!Self {
            return .{ .data = try gpa.alloc(T, len) };
        }

        pub fn zeros(gpa: Allocator) Allocator.Error!Self {
            const t = try empty(gpa);
            @memset(t.data, 0);
            return t;
        }

        pub fn full(gpa: Allocator, value: T) Allocator.Error!Self {
            const t = try empty(gpa);
            @memset(t.data, value);
            return t;
        }

        pub fn fromSlice(gpa: Allocator, values: []const T) Allocator.Error!Self {
            std.debug.assert(values.len == len);
            const t = try empty(gpa);
            @memcpy(t.data, values);
            return t;
        }

        pub fn clone(self: Self, gpa: Allocator) Allocator.Error!Self {
            return fromSlice(gpa, self.data);
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            gpa.free(self.data);
        }
    };
}

pub const MatmulOpts = @import("../gpu/gpu.zig").MatmulOpts;

pub fn MatmulResult(comptime A: type, comptime B: type, comptime opts: MatmulOpts) type {
    if (A.rank != 2 or B.rank != 2) {
        @compileError("matmul requires rank-2 tensors");
    }
    if (A.Elem != B.Elem) {
        @compileError("matmul element types differ");
    }
    const m = if (opts.trans_a) A.shape[1] else A.shape[0];
    const ka = if (opts.trans_a) A.shape[0] else A.shape[1];
    const kb = if (opts.trans_b) B.shape[1] else B.shape[0];
    const n = if (opts.trans_b) B.shape[0] else B.shape[1];
    if (ka != kb) {
        @compileError(std.fmt.comptimePrint(
            "matmul shape mismatch: ({d}x{d}) * ({d}x{d})",
            .{ m, ka, kb, n },
        ));
    }
    return Tensor(A.Elem, &.{ m, n });
}

pub fn matmulInto(
    gpa: Allocator,
    par: kernels.Parallel,
    a: anytype,
    b: anytype,
    out: anytype,
    comptime opts: MatmulOpts,
) !void {
    const A = @TypeOf(a);
    comptime {
        if (@TypeOf(out) != MatmulResult(A, @TypeOf(b), opts)) {
            @compileError("matmulInto: out does not have the product's shape");
        }
    }
    const M = @TypeOf(out).shape[0];
    const N = @TypeOf(out).shape[1];
    const K = if (opts.trans_a) A.shape[0] else A.shape[1];

    if (par.gpu) |g| return g.matmul(a.data, b.data, out.data, .{ .m = M, .n = N, .k = K }, opts);

    const cfg: kernels.TileConfig = .{ .trans_a = opts.trans_a, .trans_b = opts.trans_b, .accumulate = opts.accumulate };
    const work = try gpa.alloc(A.Elem, kernels.matmulTiledWorkLen(A.Elem, cfg, par, M, K, N));
    defer gpa.free(work);
    kernels.matmulTiled(A.Elem, cfg, par, .{ .m = M, .k = K, .n = N, .a = a.data, .b = b.data, .out = out.data }, work);
}

pub fn matmul(
    gpa: Allocator,
    par: kernels.Parallel,
    a: anytype,
    b: anytype,
    comptime opts: MatmulOpts,
) !MatmulResult(@TypeOf(a), @TypeOf(b), opts) {
    comptime {
        if (opts.accumulate) @compileError("matmul returns a fresh tensor; use matmulInto to accumulate");
    }
    std.debug.assert(par.gpu == null);
    const out = try MatmulResult(@TypeOf(a), @TypeOf(b), opts).empty(gpa);
    errdefer out.deinit(gpa);
    try matmulInto(gpa, par, a, b, out, opts);
    return out;
}

const testing = std.testing;

test "zeros, full, fromSlice, clone" {
    const gpa = testing.allocator;
    const T23 = Tensor(f32, &.{ 2, 3 });

    const z = try T23.zeros(gpa);
    defer z.deinit(gpa);
    try testing.expectEqual(@as(usize, 6), T23.len);
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 0 }, z.data);

    const f = try T23.full(gpa, 2.5);
    defer f.deinit(gpa);
    try testing.expectEqualSlices(f32, &.{ 2.5, 2.5, 2.5, 2.5, 2.5, 2.5 }, f.data);

    const s = try T23.fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });
    defer s.deinit(gpa);
    const c = try s.clone(gpa);
    defer c.deinit(gpa);
    s.data[0] = 99;
    try testing.expectEqual(@as(f32, 1), c.data[0]);
}

test "matmul small shape uses the untiled kernel" {
    const gpa = testing.allocator;
    const a = try Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit(gpa);
    const b = try Tensor(f32, &.{ 3, 2 }).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit(gpa);
    const c = try matmul(gpa, .{}, a, b, .{});
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), @TypeOf(c).shape[0]);
    try testing.expectEqual(@as(usize, 2), @TypeOf(c).shape[1]);
    try testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, c.data);
}

fn expectMatmulMatchesNaive(
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    comptime opts: MatmulOpts,
) !void {
    const gpa = testing.allocator;
    const a_dims = comptime if (opts.trans_a) [_]usize{ K, M } else [_]usize{ M, K };
    const b_dims = comptime if (opts.trans_b) [_]usize{ N, K } else [_]usize{ K, N };
    const a = try Tensor(f32, &a_dims).empty(gpa);
    defer a.deinit(gpa);
    const b = try Tensor(f32, &b_dims).empty(gpa);
    defer b.deinit(gpa);
    for (a.data, 0..) |*x, i| x.* = @floatFromInt(i % 7);
    for (b.data, 0..) |*x, i| x.* = @floatFromInt(i % 5);

    const la = try gpa.alloc(f32, M * K);
    defer gpa.free(la);
    const lb = try gpa.alloc(f32, K * N);
    defer gpa.free(lb);
    for (0..M) |i| {
        for (0..K) |p| la[i * K + p] = if (opts.trans_a) a.data[p * M + i] else a.data[i * K + p];
    }
    for (0..K) |p| {
        for (0..N) |j| lb[p * N + j] = if (opts.trans_b) b.data[j * K + p] else b.data[p * N + j];
    }

    const want = try gpa.alloc(f32, M * N);
    defer gpa.free(want);
    kernels.matmulNaive(f32, .{ .m = M, .k = K, .n = N, .a = la, .b = lb, .out = want });

    const got = try matmul(gpa, .{}, a, b, opts);
    defer got.deinit(gpa);
    try testing.expectEqual(M, @TypeOf(got).shape[0]);
    try testing.expectEqual(N, @TypeOf(got).shape[1]);
    try testing.expectEqualSlices(f32, want, got.data);
}

test "matmul matches the naive kernel on both sides of the tiling threshold" {
    try expectMatmulMatchesNaive(7, 5, 3, .{});
    try expectMatmulMatchesNaive(50, 64, 33, .{});
    try expectMatmulMatchesNaive(190, 192, 191, .{});
    try expectMatmulMatchesNaive(200, 193, 197, .{});
    try expectMatmulMatchesNaive(257, 130, 100, .{});
}

test "matmulInto accumulates on top of what is already in out" {
    const gpa = testing.allocator;
    const a = try Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit(gpa);
    const b = try Tensor(f32, &.{ 3, 2 }).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit(gpa);
    const out = try Tensor(f32, &.{ 2, 2 }).fromSlice(gpa, &.{ 1, 1, 1, 1 });
    defer out.deinit(gpa);

    try matmulInto(gpa, .{}, a, b, out, .{ .accumulate = true });
    try testing.expectEqualSlices(f32, &.{ 59, 65, 140, 155 }, out.data);
    try matmulInto(gpa, .{}, a, b, out, .{});
    try testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, out.data);
}

test "matmul with transposed operands reads them in place" {
    inline for (.{ .{ true, false }, .{ false, true }, .{ true, true } }) |flags| {
        const opts: MatmulOpts = .{ .trans_a = flags[0], .trans_b = flags[1] };
        try expectMatmulMatchesNaive(7, 5, 3, opts);
        try expectMatmulMatchesNaive(50, 64, 33, opts);
        try expectMatmulMatchesNaive(200, 193, 197, opts);
        try expectMatmulMatchesNaive(1, 9, 1, opts);
    }
}
