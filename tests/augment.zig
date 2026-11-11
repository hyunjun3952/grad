const std = @import("std");
const lib = @import("grad");
const Config = lib.math.augment.Config;
const Transform = lib.math.augment.Transform;
const transformOf = lib.math.augment.transformOf;
const pixel = lib.math.augment.pixel;
const identity = lib.math.augment.identity;

const Gpu = lib.gpu.Gpu;

fn expectedPixel(image: []const f32, p: usize, width: usize, tr: Transform) f32 {
    const middle = (@as(f64, @floatFromInt(width)) - 1) / 2;
    const ox = @as(f64, @floatFromInt(p % width)) - middle;
    const oy = @as(f64, @floatFromInt(p / width)) - middle;
    const sx = tr.a * ox + tr.b * oy + middle - tr.dx;
    const sy = tr.c * ox + tr.d * oy + middle - tr.dy;
    var total: f64 = 0;
    for (0..width) |y| {
        for (0..width) |x| {
            const wx = @max(0, 1 - @abs(sx - @as(f64, @floatFromInt(x))));
            const wy = @max(0, 1 - @abs(sy - @as(f64, @floatFromInt(y))));
            total += wx * wy * image[y * width + x];
        }
    }
    return @floatCast(total);
}

test "the identity transform copies the image and a shift moves it with zero padding" {
    const image = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var out: [9]f32 = undefined;
    for (&out, 0..) |*v, p| v.* = pixel(&image, p, 3, 1, identity);
    try std.testing.expectEqualSlices(f32, &image, &out);
    var right = identity;
    right.dx = 1;
    for (&out, 0..) |*v, p| v.* = pixel(&image, p, 3, 1, right);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 0, 4, 5, 0, 7, 8 }, &out);
    var up = identity;
    up.dy = -1;
    for (&out, 0..) |*v, p| v.* = pixel(&image, p, 3, 1, up);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6, 7, 8, 9, 0, 0, 0 }, &out);
}

test "bilinear sampling averages neighbours and rotation keeps the centre fixed" {
    const image = [_]f32{ 0, 2, 4, 0, 2, 4, 0, 2, 4 };
    var half = identity;
    half.dx = -0.5;
    try std.testing.expectApproxEqAbs(@as(f32, 1), pixel(&image, 3, 3, 1, half), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), pixel(&image, 4, 3, 1, half), 1e-6);

    const rotated = transformOf(0, .{ .max_angle = 40, .seed = 3 });
    try std.testing.expect(rotated.b != 0);
    const grid = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    try std.testing.expectApproxEqAbs(@as(f32, 5), pixel(&grid, 4, 3, 1, rotated), 1e-5);
}

test "pixel matches an independent bilinear implementation for random transforms" {
    var prng = std.Random.DefaultPrng.init(11);
    var image: [28 * 28]f32 = undefined;
    for (&image) |*v| v.* = prng.random().float(f32);
    for (0..20) |sample| {
        const tr = transformOf(sample, .{ .max_shift = 2, .max_angle = 30, .max_scale = 0.2, .seed = 4 });
        for (0..28 * 28) |p| try std.testing.expectApproxEqAbs(expectedPixel(&image, p, 28, tr), pixel(&image, p, 28, 1, tr), 1e-4);
    }
}

test "transforms stay inside the configured ranges and reach both ends" {
    var low = false;
    var high = false;
    for (0..4000) |sample| {
        const config: Config = .{ .max_shift = 2, .max_angle = 30, .max_scale = 0.2, .seed = 9 };
        const tr = transformOf(sample, config);
        try std.testing.expect(tr.dx >= -2 and tr.dx <= 2 and tr.dy >= -2 and tr.dy <= 2);
        try std.testing.expectEqual(-tr.b, tr.c);
        try std.testing.expectEqual(tr.a, tr.d);
        const angle = std.math.atan2(tr.b, tr.a) * 180.0 / std.math.pi;
        try std.testing.expect(@abs(angle) <= 30.01);
        const zoom = 1 / @sqrt(tr.a * tr.a + tr.b * tr.b);
        try std.testing.expect(zoom >= 0.799 and zoom <= 1.201);
        low = low or angle < -29;
        high = high or angle > 29;
    }
    try std.testing.expect(low and high);
    const none = transformOf(5, .{ .seed = 1 });
    var expected = identity;
    expected.cut_x = none.cut_x;
    expected.cut_y = none.cut_y;
    try std.testing.expectEqual(expected, none);
}

test "gpu gather applies the same augmentation as the cpu" {
    const gpu = try Gpu.forTest(1 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(3);
    var images: [4 * 784]f32 = undefined;
    for (&images) |*v| v.* = prng.random().float(f32);
    const labels = [_]u8{ 4, 5, 6, 7 };
    const order = [_]u32{ 2, 0, 1, 3 };
    const start = 2;
    const config: Config = .{ .max_shift = 2, .max_angle = 25, .max_scale = 0.15, .mirror = true, .box = .{ .x0 = 3, .y0 = 5, .x1 = 20, .y1 = 12 }, .seed = 5 };

    const d_images = try gpu.alloc(f32, images.len);
    const d_labels = try gpu.alloc(u8, labels.len);
    const d_order = try gpu.alloc(u32, order.len);
    const d_batch = try gpu.alloc(f32, 2 * 784);
    const d_out_labels = try gpu.alloc(u8, 2);
    const d_other_labels = try gpu.alloc(u8, 2);
    try gpu.upload(d_images, &images);
    try gpu.upload(d_labels, &labels);
    try gpu.upload(d_order, &order);
    try gpu.gather(.{ .images = d_images, .labels = d_labels, .order = d_order }, .{ .batch = d_batch, .labels = d_out_labels, .other_labels = d_other_labels }, start, config);

    var got: [2 * 784]f32 = undefined;
    var got_labels: [2]u8 = undefined;
    var other_labels: [2]u8 = undefined;
    try gpu.download(&got, d_batch);
    try gpu.download(&got_labels, d_out_labels);
    try gpu.download(&other_labels, d_other_labels);
    try std.testing.expectEqualSlices(u8, &.{ 5, 7 }, &got_labels);
    try std.testing.expectEqualSlices(u8, &.{ 7, 5 }, &other_labels);
    var pasted: usize = 0;
    for (0..2) |r| {
        for (0..784) |p| {
            const from = if (config.box.contains(p, 28, 1)) (r + 1) % 2 else r;
            pasted += @intFromBool(from != r);
            const tr = transformOf(start + from, config);
            try std.testing.expectApproxEqAbs(pixel(images[order[start + from] * 784 ..].ptr, p, 28, 1, tr), got[r * 784 + p], 1e-5);
        }
    }
    try std.testing.expectEqual(@as(usize, 2 * 17 * 7), pasted);
}
test "flipping mirrors the image, and colour channels are sampled independently" {
    var mirrored: ?Transform = null;
    for (0..64) |sample| {
        const tr = transformOf(sample, .{ .flip = true, .seed = 2 });
        if (tr.a < 0) mirrored = tr;
    }
    const tr = mirrored.?;
    const image = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var out: [9]f32 = undefined;
    for (&out, 0..) |*v, p| v.* = pixel(&image, p, 3, 1, tr);
    try std.testing.expectEqualSlices(f32, &.{ 3, 2, 1, 6, 5, 4, 9, 8, 7 }, &out);

    const colour = [_]f32{ 1, 10, 2, 20, 3, 30, 4, 40 };
    var shifted = identity;
    shifted.dx = 1;
    var got: [8]f32 = undefined;
    for (&got, 0..) |*v, i| v.* = pixel(&colour, i, 2, 2, shifted);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 10, 0, 0, 3, 30 }, &got);
    try std.testing.expect(!(Config{}).active());
    try std.testing.expect((Config{ .flip = true }).active());
}

test "cutout blanks a square around a random centre and leaves the rest unchanged" {
    const config: Config = .{ .cutout = 2, .seed = 6 };
    const image: [16]f32 = @splat(1);
    var found = false;
    var low = false;
    var high = false;
    for (0..400) |sample| {
        const tr = transformOf(sample, config);
        try std.testing.expect(tr.cut_x >= 0 and tr.cut_x < 1 and tr.cut_y >= 0 and tr.cut_y < 1);
        low = low or tr.cut_x < 0.2;
        high = high or tr.cut_x > 0.8;
        if (tr.cut_x < 0.45 or tr.cut_x > 0.55 or tr.cut_y < 0.45 or tr.cut_y > 0.55) continue;
        found = true;
        var blank: usize = 0;
        for (0..16) |p| {
            const v = pixel(&image, p, 4, 1, tr);
            try std.testing.expect(v == 0 or v == 1);
            if (v == 0) {
                blank += 1;
                try std.testing.expect(p % 4 == 1 or p % 4 == 2);
                try std.testing.expect(p / 4 == 1 or p / 4 == 2);
            }
        }
        try std.testing.expectEqual(@as(usize, 4), blank);
    }
    try std.testing.expect(found and low and high);
    try std.testing.expect(config.active());
}

test "mirror flips every sample, flip only some" {
    const image = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var always: usize = 0;
    var sometimes: usize = 0;
    for (0..64) |sample| {
        const forced = transformOf(sample, .{ .mirror = true, .seed = 8 });
        always += @intFromBool(forced.a < 0);
        sometimes += @intFromBool(transformOf(sample, .{ .flip = true, .seed = 8 }).a < 0);
        var out: [9]f32 = undefined;
        for (&out, 0..) |*v, p| v.* = pixel(&image, p, 3, 1, forced);
        try std.testing.expectEqualSlices(f32, &.{ 3, 2, 1, 6, 5, 4, 9, 8, 7 }, &out);
    }
    try std.testing.expectEqual(@as(usize, 64), always);
    try std.testing.expect(sometimes > 10 and sometimes < 54);
    try std.testing.expect((Config{ .mirror = true }).active());
}
