const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const cifar = @import("cifar.zig");
const Dataset = @import("dataset.zig").Dataset;

pub fn load(gpa: Allocator, io: Io, dir: []const u8, split: []const u8) !Dataset {
    var path_buf: [256]u8 = undefined;
    const name = if (std.mem.eql(u8, split, "train")) "train.bin" else "test.bin";
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, name });
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(bytes);
    return cifar.parseAs(gpa, &.{bytes}, cifar.hundred);
}
