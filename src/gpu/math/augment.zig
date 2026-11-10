const drop = @import("dropout.zig");

pub const Shift = struct { dx: i32, dy: i32 };

pub inline fn shiftOf(seed: u64, sample: usize, max_shift: u32) Shift {
    const h = drop.hash(seed, sample);
    const span: u64 = 2 * @as(u64, max_shift) + 1;
    return .{
        .dx = @as(i32, @intCast(h % span)) - @as(i32, @intCast(max_shift)),
        .dy = @as(i32, @intCast((h >> 32) % span)) - @as(i32, @intCast(max_shift)),
    };
}

pub inline fn pixel(image: [*]const f32, p: usize, width: usize, shift: Shift) f32 {
    const y = @as(i32, @intCast(p / width)) - shift.dy;
    const x = @as(i32, @intCast(p % width)) - shift.dx;
    const n: i32 = @intCast(width);
    return if (y < 0 or y >= n or x < 0 or x >= n) 0 else image[@as(usize, @intCast(y)) * width + @as(usize, @intCast(x))];
}

pub fn side(pixels: usize) usize {
    var s: usize = 1;
    while (s * s < pixels) s += 1;
    return s;
}

const std = @import("std");
const Gpu = @import("../gpu.zig").Gpu;

test "pixel shifts the image and pads with zeros" {
    const image = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var out: [9]f32 = undefined;
    for (&out, 0..) |*v, p| v.* = pixel(&image, p, 3, .{ .dx = 1, .dy = 0 });
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 0, 4, 5, 0, 7, 8 }, &out);
    for (&out, 0..) |*v, p| v.* = pixel(&image, p, 3, .{ .dx = 0, .dy = -1 });
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6, 7, 8, 9, 0, 0, 0 }, &out);
}

test "shifts stay inside the range and reach both ends" {
    var low = false;
    var high = false;
    for (0..2000) |sample| {
        const s = shiftOf(9, sample, 2);
        try std.testing.expect(s.dx >= -2 and s.dx <= 2 and s.dy >= -2 and s.dy <= 2);
        low = low or s.dx == -2;
        high = high or s.dx == 2;
    }
    try std.testing.expect(low and high);
    try std.testing.expectEqual(Shift{ .dx = 0, .dy = 0 }, shiftOf(1, 5, 0));
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

    const d_images = try gpu.alloc(f32, images.len);
    const d_labels = try gpu.alloc(u8, labels.len);
    const d_order = try gpu.alloc(u32, order.len);
    const d_batch = try gpu.alloc(f32, 2 * 784);
    const d_out_labels = try gpu.alloc(u8, 2);
    try gpu.upload(d_images, &images);
    try gpu.upload(d_labels, &labels);
    try gpu.upload(d_order, &order);
    try gpu.gather(d_images, d_labels, d_order, d_batch, d_out_labels, start, .{ .max_shift = 2, .seed = 5 });

    var got: [2 * 784]f32 = undefined;
    var got_labels: [2]u8 = undefined;
    try gpu.download(&got, d_batch);
    try gpu.download(&got_labels, d_out_labels);
    try std.testing.expectEqualSlices(u8, &.{ 5, 7 }, &got_labels);
    for (0..2) |r| {
        const shift = shiftOf(5, start + r, 2);
        for (0..784) |p| {
            try std.testing.expectEqual(pixel(images[order[start + r] * 784 ..].ptr, p, 28, shift), got[r * 784 + p]);
        }
    }
}
