const std = @import("std");
const lib = @import("grad");
const testing = std.testing;

const gpu_mod = lib.gpu;
const Gpu = gpu_mod.Gpu;
const kernels = lib.kernels;
const bf16 = lib.math.bf16;

fn expectClose(want: []const f32, got: []const f32, tolerance: f32) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| {
        if (@abs(w - g) > tolerance * @max(1, @abs(w))) {
            std.debug.print("mismatch at {d}: want {d}, got {d}\n", .{ i, w, g });
            return error.TestExpectedApproxEqAbs;
        }
    }
}

test "gpu matmul, f32 and bf16 tensor cores, matches the cpu reference for every transpose, shape and accumulate flag" {
    const gpu = try Gpu.forTest(64 << 20);
    defer gpu.deinit();

    const shapes = [_][3]usize{
        .{ 1, 1, 1 },      .{ 5, 7, 3 },       .{ 64, 64, 16 },   .{ 65, 70, 17 },
        .{ 130, 33, 200 }, .{ 784, 128, 64 },  .{ 64, 10, 128 },  .{ 200, 300, 784 },
        .{ 25, 16, 4000 }, .{ 400, 32, 4099 }, .{ 512, 64, 576 }, .{ 300, 130, 96 },
        // Few rows or few columns: these go to the smaller tiles chosen by block count (a GRU step has 32 rows).
        .{ 32, 768, 128 }, .{ 32, 512, 256 },  .{ 32, 256, 768 }, .{ 256, 768, 32 },
        .{ 100, 600, 40 },
    };
    for (shapes) |s| {
        for ([_][2]bool{ .{ false, false }, .{ true, false }, .{ false, true } }) |t| {
            for ([_]bool{ false, true }) |acc| {
                try checkMatmul(gpu, 1, s[0], s[1], s[2], t[0], t[1], acc, false);
                try checkMatmul(gpu, 1, s[0], s[1], s[2], t[0], t[1], acc, true);
            }
        }
    }
}

fn checkMatmul(gpu: *Gpu, batch: usize, m: usize, n: usize, k: usize, ta: bool, tb: bool, accumulate: bool, tensor: bool) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(batch * 7919 + m * 1000003 + n * 1009 + k);
    const rng = prng.random();

    const a = try gpa.alloc(f32, batch * m * k);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, batch * k * n);
    defer gpa.free(b);
    const want = try gpa.alloc(f32, batch * m * n);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, batch * m * n);
    defer gpa.free(got);
    const la = try gpa.alloc(f32, m * k);
    defer gpa.free(la);
    const lb = try gpa.alloc(f32, k * n);
    defer gpa.free(lb);
    for (a) |*x| x.* = rng.float(f32) - 0.5;
    for (b) |*x| x.* = rng.float(f32) - 0.5;
    gpu.tensor_cores = tensor;
    gpu.tensor_core_min_work = 0;
    if (tensor) {
        for (a) |*x| x.* = bf16.decode(bf16.encode(x.*));
        for (b) |*x| x.* = bf16.decode(bf16.encode(x.*));
    }
    // The reference multiplies each matrix of the batch on its own; the result of matrix i must not depend on its neighbours.
    for (0..batch) |s| {
        const sa = a[s * m * k ..][0 .. m * k];
        const sb = b[s * k * n ..][0 .. k * n];
        for (0..m) |i| {
            for (0..k) |p| la[i * k + p] = if (ta) sa[p * m + i] else sa[i * k + p];
        }
        for (0..k) |p| {
            for (0..n) |j| lb[p * n + j] = if (tb) sb[j * k + p] else sb[p * n + j];
        }
        kernels.matmulNaive(f32, .{ .m = m, .k = k, .n = n, .a = la, .b = lb, .out = want[s * m * n ..][0 .. m * n] });
    }
    @memset(got, 1);
    if (accumulate) {
        for (want) |*w| w.* += 1;
    }

    try gpu.resetStep();
    const da = try gpu.scratch(f32, a.len);
    const db = try gpu.scratch(f32, b.len);
    const dc = try gpu.scratch(f32, got.len);
    try gpu.upload(da, a);
    try gpu.upload(db, b);
    try gpu.upload(dc, got);
    try gpu.matmul(da, db, dc, .{ .m = m, .n = n, .k = k, .batch = batch }, .{ .trans_a = ta, .trans_b = tb, .accumulate = accumulate });
    try gpu.download(got, dc);
    try expectClose(want, got, 2e-4);
}

test "gpu batched matmul matches one product per matrix for every transpose, tile shape and accumulate flag" {
    const gpu = try Gpu.forTest(64 << 20);
    defer gpu.deinit();

    // Tile kernels are chosen by n and m: 128x128, 128x32 and 32x32; k above 256 would split a single product.
    const shapes = [_][4]usize{
        .{ 3, 5, 7, 3 },    .{ 8, 64, 64, 16 },  .{ 5, 65, 70, 17 },   .{ 4, 16, 16, 64 },
        .{ 6, 40, 20, 33 }, .{ 2, 20, 12, 700 }, .{ 12, 33, 129, 40 }, .{ 1, 7, 9, 11 },
    };
    for (shapes) |s| {
        for ([_][2]bool{ .{ false, false }, .{ true, false }, .{ false, true } }) |t| {
            for ([_]bool{ false, true }) |acc| {
                try checkMatmul(gpu, s[0], s[1], s[2], s[3], t[0], t[1], acc, false);
                try checkMatmul(gpu, s[0], s[1], s[2], s[3], t[0], t[1], acc, true);
            }
        }
    }
}
