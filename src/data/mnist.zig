const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Dataset = @import("dataset.zig").Dataset;

fn bigEndian(bytes: []const u8, at: usize) usize {
    return std.mem.readInt(u32, bytes[at..][0..4], .big);
}

pub fn parse(gpa: Allocator, image_bytes: []const u8, label_bytes: []const u8) !Dataset {
    if (image_bytes.len < 16 or bigEndian(image_bytes, 0) != 2051) return error.BadImageFile;
    if (label_bytes.len < 8 or bigEndian(label_bytes, 0) != 2049) return error.BadLabelFile;

    const n = bigEndian(image_bytes, 4);
    const pixels = bigEndian(image_bytes, 8) * bigEndian(image_bytes, 12);
    if (bigEndian(label_bytes, 4) != n or image_bytes.len != 16 + n * pixels or label_bytes.len != 8 + n) {
        return error.SizeMismatch;
    }

    const images = try gpa.alloc(f32, n * pixels);
    errdefer gpa.free(images);
    for (images, image_bytes[16..]) |*x, b| x.* = @as(f32, @floatFromInt(b)) / 255.0;
    const labels = try gpa.dupe(u8, label_bytes[8..]);
    return .{ .pixels = pixels, .images = images, .labels = labels };
}

pub fn load(gpa: Allocator, io: Io, dir: []const u8, split: []const u8) !Dataset {
    var path_buf: [256]u8 = undefined;
    const limit: Io.Limit = .limited(1 << 28);

    const images_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}-images-idx3-ubyte", .{ dir, split });
    const image_bytes = try Io.Dir.cwd().readFileAlloc(io, images_path, gpa, limit);
    defer gpa.free(image_bytes);

    const labels_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}-labels-idx1-ubyte", .{ dir, split });
    const label_bytes = try Io.Dir.cwd().readFileAlloc(io, labels_path, gpa, limit);
    defer gpa.free(label_bytes);

    return parse(gpa, image_bytes, label_bytes);
}

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
