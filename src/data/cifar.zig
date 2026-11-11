const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Dataset = @import("dataset.zig").Dataset;

const side = 32;
const channels = 3;
pub const pixels = side * side * channels;
pub const record = 1 + pixels;
pub const mean = [channels]f32{ 0.4914, 0.4822, 0.4465 };
pub const deviation = [channels]f32{ 0.2470, 0.2435, 0.2616 };

pub const Format = struct { label_bytes: usize, classes: u8 };
pub const ten: Format = .{ .label_bytes = 1, .classes = 10 };
pub const hundred: Format = .{ .label_bytes = 2, .classes = 100 };

pub fn parse(gpa: Allocator, files: []const []const u8) !Dataset {
    return parseAs(gpa, files, ten);
}

pub fn parseAs(gpa: Allocator, files: []const []const u8, format: Format) !Dataset {
    const size = format.label_bytes + pixels;
    var total: usize = 0;
    for (files) |bytes| {
        if (bytes.len % size != 0) return error.BadCifarFile;
        total += bytes.len / size;
    }
    const images = try gpa.alloc(f32, total * pixels);
    errdefer gpa.free(images);
    const labels = try gpa.alloc(u8, total);
    errdefer gpa.free(labels);

    var at: usize = 0;
    for (files) |bytes| {
        for (0..bytes.len / size) |i| {
            const entry = bytes[i * size ..][0..size];
            const label = entry[format.label_bytes - 1];
            if (label >= format.classes) return error.BadCifarFile;
            labels[at] = label;
            for (0..channels) |c| {
                for (0..side * side) |p| {
                    const value = @as(f32, @floatFromInt(entry[format.label_bytes + c * side * side + p])) / 255.0;
                    images[at * pixels + p * channels + c] = (value - mean[c]) / deviation[c];
                }
            }
            at += 1;
        }
    }
    return .{ .pixels = pixels, .images = images, .labels = labels };
}

pub fn load(gpa: Allocator, io: Io, dir: []const u8, split: []const u8) !Dataset {
    const train_files = [_][]const u8{ "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" };
    const test_files = [_][]const u8{"test_batch.bin"};
    const names: []const []const u8 = if (std.mem.eql(u8, split, "train")) &train_files else &test_files;

    var loaded: [train_files.len][]u8 = undefined;
    var count: usize = 0;
    defer for (loaded[0..count]) |bytes| gpa.free(bytes);
    var path_buf: [256]u8 = undefined;
    for (names) |name| {
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, name });
        loaded[count] = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
        count += 1;
    }
    var views: [train_files.len][]const u8 = undefined;
    for (loaded[0..count], 0..) |bytes, i| views[i] = bytes;
    return parse(gpa, views[0..count]);
}
