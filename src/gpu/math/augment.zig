const std = @import("std");
const drop = @import("dropout.zig");

pub const Box = struct {
    x0: u32 = 0,
    y0: u32 = 0,
    x1: u32 = 0,
    y1: u32 = 0,

    pub inline fn contains(self: Box, index: usize, width: usize, channels: usize) bool {
        const x = (index / channels) % width;
        const y = (index / channels) / width;
        return x >= self.x0 and x < self.x1 and y >= self.y0 and y < self.y1;
    }
};

pub const Config = struct {
    max_shift: u32 = 0,
    max_angle: f32 = 0,
    max_scale: f32 = 0,
    flip: bool = false,
    mirror: bool = false,
    cutout: f32 = 0,
    box: Box = .{},
    channels: u32 = 1,
    seed: u64 = 0,

    pub inline fn active(self: Config) bool {
        return self.max_shift != 0 or self.max_angle != 0 or self.max_scale != 0 or self.flip or self.mirror or self.cutout != 0;
    }
};

pub const Transform = struct { a: f32, b: f32, c: f32, d: f32, dx: f32, dy: f32, cut: f32 = 0, cut_x: f32 = 0, cut_y: f32 = 0 };

pub const identity: Transform = .{ .a = 1, .b = 0, .c = 0, .d = 1, .dx = 0, .dy = 0 };

inline fn unit(bits: u64) f32 {
    return @as(f32, @floatFromInt(bits >> 40)) * (2.0 / 16777216.0) - 1;
}

inline fn sine(x: f32) f32 {
    const x2 = x * x;
    return x * (1 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0)))));
}

inline fn cosine(x: f32) f32 {
    const x2 = x * x;
    return 1 + x2 * (-1.0 / 2.0 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0))));
}

pub inline fn transformOf(sample: usize, config: Config) Transform {
    const span: u64 = 2 * @as(u64, config.max_shift) + 1;
    const shift_bits = drop.hash(config.seed, 4 * sample);
    const dx = @as(i32, @intCast(shift_bits % span)) - @as(i32, @intCast(config.max_shift));
    const dy = @as(i32, @intCast((shift_bits >> 32) % span)) - @as(i32, @intCast(config.max_shift));
    // Degrees to radians; the polynomials above stay accurate up to about 45 degrees.
    const angle = unit(drop.hash(config.seed, 4 * sample + 1)) * config.max_angle * (3.14159265 / 180.0);
    const zoom = 1 + unit(drop.hash(config.seed, 4 * sample + 2)) * config.max_scale;
    const mirror: f32 = if (config.mirror or (config.flip and drop.hash(config.seed, 4 * sample + 3) & 1 == 1)) -1 else 1;
    const cut_bits = drop.hash(config.seed, 4 * sample + 3) >> 8;
    return .{
        .a = mirror * cosine(angle) / zoom,
        .b = sine(angle) / zoom,
        .c = -mirror * sine(angle) / zoom,
        .d = cosine(angle) / zoom,
        .dx = @floatFromInt(dx),
        .dy = @floatFromInt(dy),
        .cut = config.cutout,
        .cut_x = @as(f32, @floatFromInt(cut_bits & 0xFFFFFF)) / 16777216.0,
        .cut_y = @as(f32, @floatFromInt((cut_bits >> 24) & 0xFFFFFF)) / 16777216.0,
    };
}

inline fn at(image: [*]const f32, x: i32, y: i32, width: i32, channels: usize, channel: usize) f32 {
    return if (x < 0 or y < 0 or x >= width or y >= width) 0 else image[(@as(usize, @intCast(y)) * @as(usize, @intCast(width)) + @as(usize, @intCast(x))) * channels + channel];
}

pub inline fn pixel(image: [*]const f32, index: usize, width: usize, channels: usize, tr: Transform) f32 {
    const p = index / channels;
    const channel = index % channels;
    const w: i32 = @intCast(width);
    const wf: f32 = @floatFromInt(width);
    if (tr.cut > 0 and @abs(@as(f32, @floatFromInt(p % width)) + 0.5 - tr.cut_x * wf) < tr.cut / 2 and @abs(@as(f32, @floatFromInt(p / width)) + 0.5 - tr.cut_y * wf) < tr.cut / 2) return 0;
    const middle = (@as(f32, @floatFromInt(width)) - 1) / 2;
    const ox = @as(f32, @floatFromInt(p % width)) - middle;
    const oy = @as(f32, @floatFromInt(p / width)) - middle;
    const sx = tr.a * ox + tr.b * oy + middle - tr.dx;
    const sy = tr.c * ox + tr.d * oy + middle - tr.dy;
    const x0: i32 = @intFromFloat(@floor(sx));
    const y0: i32 = @intFromFloat(@floor(sy));
    const fx = sx - @as(f32, @floatFromInt(x0));
    const fy = sy - @as(f32, @floatFromInt(y0));
    const top = at(image, x0, y0, w, channels, channel) * (1 - fx) + at(image, x0 + 1, y0, w, channels, channel) * fx;
    const bottom = at(image, x0, y0 + 1, w, channels, channel) * (1 - fx) + at(image, x0 + 1, y0 + 1, w, channels, channel) * fx;
    return top * (1 - fy) + bottom * fy;
}

pub inline fn side(floats: usize, channels: usize) usize {
    return @intFromFloat(@sqrt(@as(f32, @floatFromInt(floats / channels))));
}

const Gpu = @import("../gpu.zig").Gpu;

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
    const gpu = Gpu.init(std.testing.allocator, 1 << 20) catch |err| switch (err) {
        error.DriverNotFound => return error.SkipZigTest,
        else => return err,
    };
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
    try gpu.gather(d_images, d_labels, d_order, d_batch, d_out_labels, d_other_labels, start, config);

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
