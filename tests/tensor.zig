const std = @import("std");
const lib = @import("grad");
const matmulInto = lib.tensor.matmulInto;
const kernels = lib.kernels;
const MatmulOpts = lib.tensor.MatmulOpts;
const matmul = lib.tensor.matmul;
const Tensor = lib.tensor.Tensor;

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
