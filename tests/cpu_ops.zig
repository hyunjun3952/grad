const std = @import("std");
const lib = @import("grad");
const kernels = lib.kernels;
const col2im = lib.cpu_ops.col2im;
const normGradInput = lib.cpu_ops.normGradInput;
const im2col = lib.cpu_ops.im2col;
const cv = lib.math.conv;
const rowsOf = lib.cpu_ops.rowsOf;
const normApply = lib.cpu_ops.normApply;
const norm = lib.math.norm;

const testing = std.testing;

test "row-wise im2col and col2im equal the per-element reference for strides, padding and channel counts" {
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    const gpa = testing.allocator;
    const par: kernels.Parallel = .{ .io = testing.io, .threads = 3 };
    for ([_]cv.Dims{
        .{ .n = 2, .h = 6, .w = 7, .c = 1, .k = 3, .stride = 1, .pad = 0 },
        .{ .n = 2, .h = 6, .w = 7, .c = 3, .k = 3, .stride = 1, .pad = 1 },
        .{ .n = 1, .h = 9, .w = 8, .c = 4, .k = 5, .stride = 2, .pad = 2 },
        .{ .n = 3, .h = 5, .w = 5, .c = 16, .k = 1, .stride = 1, .pad = 0 },
        .{ .n = 2, .h = 8, .w = 8, .c = 2, .k = 3, .stride = 2, .pad = 1 },
    }) |d| {
        const x = try gpa.alloc(f32, @as(usize, d.n) * d.h * d.w * d.c);
        defer gpa.free(x);
        for (x) |*v| v.* = rng.floatNorm(f32);
        const columns = rowsOf(d) * d.k * d.k * d.c;
        const want = try gpa.alloc(f32, columns);
        defer gpa.free(want);
        const got = try gpa.alloc(f32, columns);
        defer gpa.free(got);

        for (0..columns) |t| cv.im2col(@intCast(t), x.ptr, want.ptr, d);
        @memset(got, 99);
        im2col(par, x, got, d);
        try testing.expectEqualSlices(f32, want, got);

        const dcol = want;
        for (dcol) |*v| v.* = rng.floatNorm(f32);
        const want_x = try gpa.alloc(f32, x.len);
        defer gpa.free(want_x);
        const got_x = try gpa.alloc(f32, x.len);
        defer gpa.free(got_x);
        @memset(want_x, 0);
        @memset(got_x, 0);
        for (0..x.len) |t| cv.col2im(@intCast(t), dcol.ptr, want_x.ptr, d);
        col2im(par, dcol, got_x, d);
        for (want_x, got_x) |a, b| try testing.expectApproxEqAbs(a, b, 1e-6);
    }
}

test "row-wise batch-norm apply and input gradient equal the per-element reference" {
    var prng = std.Random.DefaultPrng.init(8);
    const rng = prng.random();
    const gpa = testing.allocator;
    const par: kernels.Parallel = .{ .io = testing.io, .threads = 3 };
    const c = 5;
    const rows = 301;
    const len = rows * c;
    const x = try gpa.alloc(f32, len);
    defer gpa.free(x);
    const dy = try gpa.alloc(f32, len);
    defer gpa.free(dy);
    const mask = try gpa.alloc(f32, len);
    defer gpa.free(mask);
    for (x, dy, mask) |*a, *b, *m| {
        a.* = rng.floatNorm(f32);
        b.* = rng.floatNorm(f32);
        m.* = rng.floatNorm(f32);
    }
    var channel: [5][c]f32 = undefined;
    for (&channel) |*row| {
        for (row) |*v| v.* = rng.float(f32) + 0.5;
    }
    var sums: [2 * c]f32 = undefined;
    for (&sums) |*v| v.* = rng.floatNorm(f32);
    const p: norm.Affine = .{ .mean = &channel[0], .inv_std = &channel[1], .gamma = &channel[2], .beta = &channel[3] };

    for ([_]u32{ 0, 1 }) |flag| {
        const l: norm.Layout = .{ .len = len, .channels = c, .flag = flag };
        const want = try gpa.alloc(f32, len);
        defer gpa.free(want);
        const got = try gpa.alloc(f32, len);
        defer gpa.free(got);
        for (0..len) |t| norm.apply(f32, t, x.ptr, want.ptr, p, l);
        normApply(par, x, got, p, l);
        try testing.expectEqualSlices(f32, want, got);

        @memset(want, 0.25);
        @memset(got, 0.25);
        for (0..len) |t| norm.gradInput(f32, t, .{ .x = x.ptr, .dy = dy.ptr, .dx = want.ptr, .sums = &sums, .mask = mask.ptr }, p, l);
        normGradInput(par, .{ .x = x.ptr, .dy = dy.ptr, .dx = got.ptr, .sums = &sums, .mask = mask.ptr }, p, l);
        for (want, got) |a, b| try testing.expectApproxEqAbs(a, b, 1e-6);
    }
}
