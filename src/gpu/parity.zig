const std = @import("std");
const testing = std.testing;

const gpu_mod = @import("gpu.zig");
const Gpu = gpu_mod.Gpu;
const kernels = @import("../cpu/kernels.zig");
const bf16 = @import("math/bf16.zig");

fn initForTest() !*Gpu {
    return Gpu.init(testing.allocator, 64 << 20) catch |err| switch (err) {
        error.DriverNotFound => error.SkipZigTest,
        else => err,
    };
}

fn expectClose(want: []const f32, got: []const f32, tolerance: f32) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| {
        if (@abs(w - g) > tolerance * @max(1, @abs(w))) {
            std.debug.print("mismatch at {d}: want {d}, got {d}\n", .{ i, w, g });
            return error.TestExpectedApproxEqAbs;
        }
    }
}

fn checkMatmul(gpu: *Gpu, m: usize, n: usize, k: usize, ta: bool, tb: bool, accumulate: bool, tensor: bool) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(m * 1000003 + n * 1009 + k);
    const rng = prng.random();

    const a = try gpa.alloc(f32, m * k);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, k * n);
    defer gpa.free(b);
    const want = try gpa.alloc(f32, m * n);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, m * n);
    defer gpa.free(got);
    for (a) |*x| x.* = rng.float(f32) - 0.5;
    for (b) |*x| x.* = rng.float(f32) - 0.5;
    gpu.tensor_cores = tensor;
    gpu.tensor_core_min_work = 0;
    if (tensor) {
        for (a) |*x| x.* = bf16.decode(bf16.encode(x.*));
        for (b) |*x| x.* = bf16.decode(bf16.encode(x.*));
    }

    const la = try gpa.alloc(f32, m * k);
    defer gpa.free(la);
    const lb = try gpa.alloc(f32, k * n);
    defer gpa.free(lb);
    for (0..m) |i| {
        for (0..k) |p| la[i * k + p] = if (ta) a[p * m + i] else a[i * k + p];
    }
    for (0..k) |p| {
        for (0..n) |j| lb[p * n + j] = if (tb) b[j * k + p] else b[p * n + j];
    }
    kernels.matmulNaive(f32, m, k, n, la, lb, want);
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
    try gpu.matmul(da, db, dc, .{ .m = m, .n = n, .k = k }, .{ .trans_a = ta, .trans_b = tb, .accumulate = accumulate });
    try gpu.download(got, dc);
    try expectClose(want, got, 2e-4);
}

test "gpu matmul, f32 and bf16 tensor cores, matches the cpu reference for every transpose, shape and accumulate flag" {
    const gpu = try initForTest();
    defer gpu.deinit();

    const shapes = [_][3]usize{
        .{ 1, 1, 1 },      .{ 5, 7, 3 },       .{ 64, 64, 16 },   .{ 65, 70, 17 },
        .{ 130, 33, 200 }, .{ 784, 128, 64 },  .{ 64, 10, 128 },  .{ 200, 300, 784 },
        .{ 25, 16, 4000 }, .{ 400, 32, 4099 }, .{ 512, 64, 576 }, .{ 300, 130, 96 },
    };
    for (shapes) |s| {
        for ([_][2]bool{ .{ false, false }, .{ true, false }, .{ false, true } }) |t| {
            for ([_]bool{ false, true }) |acc| {
                try checkMatmul(gpu, s[0], s[1], s[2], t[0], t[1], acc, false);
                try checkMatmul(gpu, s[0], s[1], s[2], t[0], t[1], acc, true);
            }
        }
    }
}
