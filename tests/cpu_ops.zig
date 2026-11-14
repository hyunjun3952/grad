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

test "convForward equals the per-element im2col product on the direct single-channel path and on the GEMM path" {
    var prng = std.Random.DefaultPrng.init(21);
    const rng = prng.random();
    const gpa = testing.allocator;
    // One worker runs every share inline, so the single-threaded split of the direct paths is covered as well.
    for ([_]kernels.Parallel{ .{}, .{ .io = testing.io, .threads = 3 } }) |par| {
        for ([_]cv.Dims{
            // Single channel, no padding: the direct path for c_out = 16.
            .{ .n = 3, .h = 12, .w = 11, .c = 1, .k = 5, .c_out = 16 },
            .{ .n = 2, .h = 13, .w = 12, .c = 1, .k = 3, .stride = 2, .c_out = 32 },
            .{ .n = 1, .h = 6, .w = 6, .c = 1, .k = 1, .c_out = 16 },
            // Padding, several channels and a c_out that is no multiple of any vector width stay on the GEMM path.
            .{ .n = 2, .h = 8, .w = 8, .c = 1, .k = 3, .pad = 1, .c_out = 16 },
            .{ .n = 2, .h = 8, .w = 8, .c = 3, .k = 3, .c_out = 16 },
            .{ .n = 2, .h = 9, .w = 9, .c = 1, .k = 3, .c_out = 6 },
        }) |d| {
            const kc = @as(usize, d.k) * d.k * d.c;
            const rows = rowsOf(d);
            const x = try gpa.alloc(f32, @as(usize, d.n) * d.h * d.w * d.c);
            defer gpa.free(x);
            const w = try gpa.alloc(f32, kc * d.c_out);
            defer gpa.free(w);
            const columns = try gpa.alloc(f32, rows * kc);
            defer gpa.free(columns);
            const want = try gpa.alloc(f32, rows * d.c_out);
            defer gpa.free(want);
            const got = try gpa.alloc(f32, rows * d.c_out);
            defer gpa.free(got);
            for (x) |*v| v.* = rng.floatNorm(f32);
            for (w) |*v| v.* = rng.floatNorm(f32);

            for (0..columns.len) |t| cv.im2col(@intCast(t), x.ptr, columns.ptr, d);
            for (0..rows) |r| {
                for (0..d.c_out) |o| {
                    var sum: f64 = 0;
                    for (0..kc) |q| sum += @as(f64, columns[r * kc + q]) * w[q * d.c_out + o];
                    want[r * d.c_out + o] = @floatCast(sum);
                }
            }
            @memset(got, 99);
            try lib.cpu_ops.convForward(gpa, par, x, w, got, d);
            for (want, got) |a, b| try testing.expectApproxEqAbs(a, b, 1e-4);

            // The weight gradient adds im2col(x)^T * grad to what dw already holds.
            const upstream = try gpa.alloc(f32, rows * d.c_out);
            defer gpa.free(upstream);
            for (upstream) |*v| v.* = rng.floatNorm(f32);
            const want_dw = try gpa.alloc(f32, w.len);
            defer gpa.free(want_dw);
            const got_dw = try gpa.alloc(f32, w.len);
            defer gpa.free(got_dw);
            for (want_dw, got_dw, 0..) |*a, *b, q| {
                const start = rng.floatNorm(f32);
                a.* = start;
                b.* = start;
                var sum: f64 = start;
                for (0..rows) |r| sum += @as(f64, columns[r * kc + q / d.c_out]) * upstream[r * d.c_out + q % d.c_out];
                a.* = @floatCast(sum);
            }
            try lib.cpu_ops.convWeightGrad(gpa, par, x, upstream, got_dw, d);
            for (want_dw, got_dw) |a, b| try testing.expectApproxEqAbs(a, b, 1e-3);
        }
    }
}

test "row-wise max pooling gives the same values and winner indices as the per-element reference" {
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();
    const gpa = testing.allocator;
    const par: kernels.Parallel = .{ .io = testing.io, .threads = 3 };
    for ([_]cv.Dims{
        .{ .n = 2, .h = 8, .w = 8, .c = 5, .k = 2, .stride = 2 },
        .{ .n = 3, .h = 7, .w = 9, .c = 16, .k = 3, .stride = 2, .pad = 1 },
        .{ .n = 1, .h = 5, .w = 6, .c = 3, .k = 3, .stride = 1, .pad = 1 },
    }) |d| {
        const x = try gpa.alloc(f32, @as(usize, d.n) * d.h * d.w * d.c);
        defer gpa.free(x);
        // NOTE: values from a small set, so that windows hold ties and the first-maximum rule is exercised.
        for (x) |*v| v.* = @floatFromInt(rng.uintLessThan(u32, 6));
        const len = cv.workItems(.maxPool, d);
        const buffers = try gpa.alloc(f32, 2 * len);
        defer gpa.free(buffers);
        const indices = try gpa.alloc(u32, 2 * len);
        defer gpa.free(indices);
        for (0..len) |t| cv.maxPool(@intCast(t), x.ptr, buffers.ptr, indices.ptr, d);
        lib.cpu_ops.maxPool(par, x, buffers[len..], indices[len..], d);
        try testing.expectEqualSlices(f32, buffers[0..len], buffers[len..]);
        try testing.expectEqualSlices(u32, indices[0..len], indices[len..]);
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

    for ([_]u32{ 0, 1, 3 }) |flag| {
        const l: norm.Layout = .{ .len = len, .channels = c, .flag = flag };
        const want = try gpa.alloc(f32, len);
        defer gpa.free(want);
        const got = try gpa.alloc(f32, len);
        defer gpa.free(got);
        for (0..len) |t| norm.apply(f32, t, x.ptr, mask.ptr, want.ptr, p, l);
        normApply(par, x, mask, got, p, l);
        try testing.expectEqualSlices(f32, want, got);

        const want_dr = try gpa.alloc(f32, len);
        defer gpa.free(want_dr);
        const got_dr = try gpa.alloc(f32, len);
        defer gpa.free(got_dr);
        @memset(want, 0.25);
        @memset(got, 0.25);
        @memset(want_dr, -0.5);
        @memset(got_dr, -0.5);
        for (0..len) |t| norm.gradInput(f32, t, .{ .x = x.ptr, .dy = dy.ptr, .dx = want.ptr, .sums = &sums, .mask = mask.ptr, .dr = want_dr.ptr }, p, l);
        normGradInput(par, .{ .x = x.ptr, .dy = dy.ptr, .dx = got.ptr, .sums = &sums, .mask = mask.ptr, .dr = got_dr.ptr }, p, l);
        for (want, got) |a, b| try testing.expectApproxEqAbs(a, b, 1e-6);
        try testing.expectEqualSlices(f32, want_dr, got_dr);
    }
}

test "batch norm statistics on the CPU cover 384 channels, the widest layer of the wide CIFAR network" {
    const gpa = testing.allocator;
    const c = 384;
    const rows = 10;
    var prng = std.Random.DefaultPrng.init(8);
    const rng = prng.random();
    const x = try gpa.alloc(f32, rows * c);
    defer gpa.free(x);
    for (x) |*v| v.* = rng.floatNorm(f32) * 2 + 1;
    var mean: [c]f32 = undefined;
    var inv_std: [c]f32 = undefined;
    var running_mean: [c]f32 = @splat(0);
    var running_var: [c]f32 = @splat(1);
    const st: norm.Stats = .{ .mean = &mean, .inv_std = &inv_std, .running_mean = &running_mean, .running_var = &running_var };
    for ([_]kernels.Parallel{ .{}, .{ .io = testing.io, .threads = 3 } }) |par| {
        lib.cpu_ops.normStats(par, x, st, c, true);
        for (0..c) |ch| {
            var sum: f64 = 0;
            for (0..rows) |r| sum += x[r * c + ch];
            const mu = sum / rows;
            var square: f64 = 0;
            for (0..rows) |r| square += (x[r * c + ch] - mu) * (x[r * c + ch] - mu);
            const want_inv: f64 = 1 / @sqrt(square / rows + norm.eps);
            try testing.expectApproxEqAbs(@as(f32, @floatCast(mu)), mean[ch], 1e-5);
            try testing.expectApproxEqRel(@as(f32, @floatCast(want_inv)), inv_std[ch], 1e-4);
        }
    }
}
