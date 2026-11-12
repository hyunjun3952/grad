const std = @import("std");
const lib = @import("grad");
const Gpu = lib.gpu.Gpu;
const norm = lib.math.norm;

const testing = std.testing;

test "device memory round trip, fill, and the step arena" {
    const gpu = try Gpu.forTest(16 << 20);
    defer gpu.deinit();

    const a = try gpu.alloc(f32, 1000);
    var host: [1000]f32 = undefined;
    for (&host, 0..) |*x, i| x.* = @floatFromInt(i);
    try gpu.upload(a, &host);
    var back: [1000]f32 = undefined;
    try gpu.download(&back, a);
    try testing.expectEqualSlices(f32, &host, &back);

    try gpu.fill(a, 2.5);
    try gpu.download(&back, a);
    for (back) |x| try testing.expectEqual(@as(f32, 2.5), x);
    try testing.expectEqual(@as(f32, 2.5), try gpu.scalar(a));

    const s1 = try gpu.scratch(f32, 10);
    const s2 = try gpu.scratch(f32, 10);
    try testing.expect(@intFromPtr(s2.ptr) >= @intFromPtr(s1.ptr) + 40);
    try testing.expectEqual(@as(usize, 0), @intFromPtr(s2.ptr) % 256);
    try gpu.resetStep();
    const s3 = try gpu.scratch(f32, 10);
    try testing.expectEqual(@intFromPtr(s1.ptr), @intFromPtr(s3.ptr));
    try testing.expectError(error.OutOfMemory, gpu.scratch(u8, 1 << 30));
}

test "deterministic mode gives bit-identical results on every run, for split-K products and batch-norm statistics" {
    const gpu = try Gpu.forTest(16 << 20);
    defer gpu.deinit();
    gpu.tensor_core_min_work = 0;
    gpu.deterministic = true;
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(17);
    const rng = prng.random();

    const m = 64;
    const n = 64;
    const k = 16384;
    const a = try gpa.alloc(f32, m * k);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, k * n);
    defer gpa.free(b);
    for (a) |*v| v.* = rng.floatNorm(f32);
    for (b) |*v| v.* = rng.floatNorm(f32);
    const da = try gpu.alloc(f32, a.len);
    const db = try gpu.alloc(f32, b.len);
    const dc = try gpu.alloc(f32, m * n);
    try gpu.upload(da, a);
    try gpu.upload(db, b);

    const rows = 20000;
    const channels = 48;
    const x = try gpa.alloc(f32, rows * channels);
    defer gpa.free(x);
    for (x) |*v| v.* = rng.floatNorm(f32) * 3 + 1;
    const dx = try gpu.alloc(f32, x.len);
    try gpu.upload(dx, x);
    const mean = try gpu.alloc(f32, channels);
    const inv_std = try gpu.alloc(f32, channels);
    const running_mean = try gpu.alloc(f32, channels);
    const running_var = try gpu.alloc(f32, channels);
    try gpu.fill(running_mean, 0);
    try gpu.fill(running_var, 1);

    var first_product: [m * n]f32 = undefined;
    var first_stats: [2 * channels]f32 = undefined;
    for (0..2) |tensor_cores| {
        gpu.tensor_cores = tensor_cores == 1;
        for (0..6) |run| {
            try gpu.matmul(da, db, dc, .{ .m = m, .n = n, .k = k }, .{});
            try gpu.batchNormStats(dx, .{ .mean = mean.ptr, .inv_std = inv_std.ptr, .running_mean = running_mean.ptr, .running_var = running_var.ptr }, channels, true);
            var product: [m * n]f32 = undefined;
            var stats: [2 * channels]f32 = undefined;
            try gpu.download(&product, dc);
            try gpu.download(stats[0..channels], mean);
            try gpu.download(stats[channels..], inv_std);
            if (run == 0) {
                first_product = product;
                first_stats = stats;
            } else {
                try testing.expectEqualSlices(f32, &first_product, &product);
                try testing.expectEqualSlices(f32, &first_stats, &stats);
            }
        }
        const tolerance: f32 = if (tensor_cores == 1) 2.0 else 0.05;
        for (0..4) |i| {
            for (0..n) |j| {
                var want: f64 = 0;
                for (0..k) |p| want += @as(f64, a[i * k + p]) * b[p * n + j];
                try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), first_product[i * n + j], tolerance);
            }
        }
    }
    for (0..channels) |ch| {
        var sum: f64 = 0;
        var square: f64 = 0;
        for (0..rows) |r| sum += x[r * channels + ch];
        const mu = sum / rows;
        for (0..rows) |r| square += (x[r * channels + ch] - mu) * (x[r * channels + ch] - mu);
        try testing.expectApproxEqAbs(@as(f32, @floatCast(mu)), first_stats[ch], 1e-4);
        try testing.expectApproxEqAbs(@as(f32, @floatCast(1 / @sqrt(square / rows + norm.eps))), first_stats[channels + ch], 1e-4);
    }
}

test "a recorded graph runs only when replayed, zeroes its gradient arena each time, and repeats fills and launches" {
    const gpu = try Gpu.forTest(16 << 20);
    defer gpu.deinit();
    const x = try gpu.alloc(f32, 300);
    const y = try gpu.alloc(f32, 300);
    try gpu.fill(y, -1);
    var host: [300]f32 = undefined;
    for (&host, 0..) |*v, i| v.* = @floatFromInt(i);
    try gpu.upload(x, &host);

    try gpu.resetStep();
    const acc = try gpu.scratchZeroed(f32, 300);
    gpu.beginCapture() catch |err| switch (err) {
        error.NoGraphSupport => return error.SkipZigTest,
        else => return err,
    };
    try gpu.addInPlace(acc, x);
    try gpu.fill(y, 3);
    const graph = try gpu.endCapture();

    // Recording must not execute anything.
    var back: [300]f32 = undefined;
    try gpu.download(&back, y);
    for (back) |v| try testing.expectEqual(@as(f32, -1), v);

    for (0..3) |round| {
        for (&host, 0..) |*v, i| v.* = @floatFromInt(i * (round + 1));
        try gpu.upload(x, &host);
        try gpu.fill(y, 0);
        try gpu.replay(graph);
        // The accumulator holds this round's input alone, so it was cleared before the replay.
        try gpu.download(&back, acc);
        try testing.expectEqualSlices(f32, &host, &back);
        try gpu.download(&back, y);
        for (back) |v| try testing.expectEqual(@as(f32, 3), v);
    }

    gpu.startProfile(testing.io);
    try testing.expectError(error.ProfilingWhileCapturing, gpu.beginCapture());
}

test "a recording is not invalidated by default-stream work of another Gpu on the same device" {
    // NOTE: two Gpu values on one device share the primary context and its legacy default stream, as two training replicas do.
    // A blocking capture stream is implicitly ordered against that stream, so the memset below used to invalidate the recording.
    const gpu = try Gpu.forTest(16 << 20);
    defer gpu.deinit();
    const other = try Gpu.forTest(1 << 20);
    defer other.deinit();
    const y = try gpu.alloc(f32, 64);
    const z = try other.alloc(f32, 64);
    try gpu.fill(y, -1);

    gpu.beginCapture() catch |err| switch (err) {
        error.NoGraphSupport => return error.SkipZigTest,
        else => return err,
    };
    try gpu.fill(y, 3);
    try other.fill(z, 1);
    const graph = try gpu.endCapture();

    try gpu.replay(graph);
    var back: [64]f32 = undefined;
    try gpu.download(&back, y);
    for (back) |v| try testing.expectEqual(@as(f32, 3), v);
}

test "a cuDNN convolution can be recorded once its plan exists, and a new shape cannot be planned during a recording" {
    const gpu = try Gpu.forTest(16 << 20);
    defer gpu.deinit();
    gpu.enableCudnn("data/cudnn") catch |err| switch (err) {
        error.CudnnNotFound => return error.SkipZigTest,
        else => return err,
    };
    const geometry: lib.gpu.cudnn.Geometry = .{ .n = 2, .h = 8, .w = 8, .c = 16, .k = 16, .r = 3, .pad = 1, .stride = 1 };
    const x = try gpu.alloc(f32, 2 * 8 * 8 * 16);
    const w = try gpu.alloc(f32, 16 * 3 * 3 * 16);
    const y = try gpu.alloc(f32, 2 * 8 * 8 * 16);
    var prng = std.Random.DefaultPrng.init(3);
    const host = try testing.allocator.alloc(f32, @max(x.len, w.len));
    defer testing.allocator.free(host);
    for (host[0..x.len]) |*v| v.* = prng.random().floatNorm(f32);
    try gpu.upload(x, host[0..x.len]);
    for (host[0..w.len]) |*v| v.* = prng.random().floatNorm(f32) * 0.1;
    try gpu.upload(w, host[0..w.len]);

    // The first call plans the shape (it times real convolutions), so it has to run normally.
    try gpu.cudnnForward(geometry, x, w, y);
    const live = try testing.allocator.alloc(f32, y.len);
    defer testing.allocator.free(live);
    try gpu.download(live, y);

    try gpu.fill(y, 0);
    gpu.beginCapture() catch |err| switch (err) {
        error.NoGraphSupport => return error.SkipZigTest,
        else => return err,
    };
    try gpu.cudnnForward(geometry, x, w, y);
    var unplanned = geometry;
    unplanned.n = 3;
    try testing.expectError(error.CudnnPlanWhileCapturing, gpu.cudnnForward(unplanned, x, w, y));
    const graph = try gpu.endCapture();

    try gpu.replay(graph);
    const replayed = try testing.allocator.alloc(f32, y.len);
    defer testing.allocator.free(replayed);
    try gpu.download(replayed, y);
    try testing.expectEqualSlices(f32, live, replayed);
}
