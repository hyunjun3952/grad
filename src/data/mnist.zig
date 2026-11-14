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
    // NOTE: the header fields are 32 bits each, so their products can overflow usize in a corrupt file.
    const pixels = std.math.mul(usize, bigEndian(image_bytes, 8), bigEndian(image_bytes, 12)) catch return error.SizeMismatch;
    const body = std.math.mul(usize, n, pixels) catch return error.SizeMismatch;
    if (bigEndian(label_bytes, 4) != n or image_bytes.len - 16 != body or label_bytes.len != 8 + n) {
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
