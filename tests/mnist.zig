const std = @import("std");
const lib = @import("grad");
const load = lib.mnist.load;
const parse = lib.mnist.parse;

const testing = std.testing;

fn idxHeader(comptime magic: u32, comptime fields: []const u32) [4 + 4 * fields.len]u8 {
    var out: [4 + 4 * fields.len]u8 = undefined;
    std.mem.writeInt(u32, out[0..4], magic, .big);
    inline for (fields, 0..) |f, i| std.mem.writeInt(u32, out[4 + 4 * i ..][0..4], f, .big);
    return out;
}

test "parse normalizes pixels to 0..1 and keeps labels" {
    const images = idxHeader(2051, &.{ 2, 1, 3 }) ++ [_]u8{ 0, 255, 51, 255, 0, 102 };
    const labels = idxHeader(2049, &.{2}) ++ [_]u8{ 7, 3 };
    const data = try parse(testing.allocator, &images, &labels);
    defer data.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), data.count());
    try testing.expectEqual(@as(usize, 3), data.pixels);
    try testing.expectEqualSlices(f32, &.{ 0, 1, 0.2, 1, 0, 0.4 }, data.images);
    try testing.expectEqualSlices(u8, &.{ 7, 3 }, data.labels);
}

test "parse rejects wrong magic numbers and inconsistent sizes" {
    const images = idxHeader(2051, &.{ 2, 1, 3 }) ++ [_]u8{ 0, 255, 51, 255, 0, 102 };
    const labels = idxHeader(2049, &.{2}) ++ [_]u8{ 7, 3 };
    const short_labels = idxHeader(2049, &.{3}) ++ [_]u8{ 7, 3, 1 };
    const swapped = idxHeader(2049, &.{ 2, 1, 3 }) ++ [_]u8{ 0, 255, 51, 255, 0, 102 };

    try testing.expectError(error.BadImageFile, parse(testing.allocator, &swapped, &labels));
    try testing.expectError(error.BadLabelFile, parse(testing.allocator, &images, &images));
    try testing.expectError(error.SizeMismatch, parse(testing.allocator, &images, &short_labels));
    try testing.expectError(error.SizeMismatch, parse(testing.allocator, images[0 .. images.len - 1], &labels));
}

test "the real MNIST files, when present, have the documented shape" {
    const train = load(testing.allocator, testing.io, "data/mnist", "train") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer train.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 60000), train.count());
    try testing.expectEqual(@as(usize, 784), train.pixels);
    for (train.labels) |l| try testing.expect(l < 10);
    for (train.images) |v| try testing.expect(v >= 0 and v <= 1);
}

test "parse rejects a header whose sizes overflow, and labels can be checked against the class count" {
    const images = idxHeader(2051, &.{ 0xFFFFFFFF, 0xFFFFFFFF, 2 });
    const labels = idxHeader(2049, &.{0xFFFFFFFF});
    try testing.expectError(error.SizeMismatch, parse(testing.allocator, &images, &labels));

    const ok_images = idxHeader(2051, &.{ 2, 1, 3 }) ++ [_]u8{ 0, 255, 51, 255, 0, 102 };
    const ok_labels = idxHeader(2049, &.{2}) ++ [_]u8{ 7, 3 };
    const data = try parse(testing.allocator, &ok_images, &ok_labels);
    defer data.deinit(testing.allocator);
    try data.checkLabels(10);
    try testing.expectError(error.LabelOutOfRange, data.checkLabels(5));
}
