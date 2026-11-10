const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Gpu = @import("gpu/gpu.zig").Gpu;

const magic: u32 = 0x47524144;
const header_bytes = 8;

pub fn save(gpa: Allocator, io: Io, path: []const u8, gpu: ?*Gpu, state: anytype) !void {
    var total: usize = header_bytes;
    inline for (state) |s| total += 4 + 4 * s.len;
    const bytes = try gpa.alignedAlloc(u8, .@"4", total);
    defer gpa.free(bytes);

    std.mem.writeInt(u32, bytes[0..4], magic, .little);
    std.mem.writeInt(u32, bytes[4..8], state.len, .little);
    var at: usize = header_bytes;
    inline for (state) |s| {
        std.mem.writeInt(u32, bytes[at..][0..4], @intCast(s.len), .little);
        const values: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes[at + 4 ..][0 .. 4 * s.len]));
        if (gpu) |dev| try dev.download(values, s) else @memcpy(values, s);
        at += 4 + 4 * s.len;
    }
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

pub fn load(gpa: Allocator, io: Io, path: []const u8, gpu: ?*Gpu, state: anytype) !void {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31));
    defer gpa.free(bytes);
    if (bytes.len < header_bytes or std.mem.readInt(u32, bytes[0..4], .little) != magic) return error.BadCheckpoint;
    if (std.mem.readInt(u32, bytes[4..8], .little) != state.len) return error.CheckpointMismatch;

    var at: usize = header_bytes;
    inline for (state) |s| {
        if (at + 4 > bytes.len or std.mem.readInt(u32, bytes[at..][0..4], .little) != s.len) return error.CheckpointMismatch;
        if (at + 4 + 4 * s.len > bytes.len) return error.BadCheckpoint;
        const values = try gpa.alloc(f32, s.len);
        defer gpa.free(values);
        @memcpy(std.mem.sliceAsBytes(values), bytes[at + 4 ..][0 .. 4 * s.len]);
        if (gpu) |dev| try dev.upload(s, values) else @memcpy(s, values);
        at += 4 + 4 * s.len;
    }
}

const testing = std.testing;

test "a checkpoint restores every tensor and rejects a different layout" {
    const gpa = testing.allocator;
    const path = ".zig-cache/checkpoint-test.bin";
    defer Io.Dir.cwd().deleteFile(testing.io, path) catch {};

    var a = [_]f32{ 1, 2, 3 };
    var b = [_]f32{ -0.5, 7 };
    try save(gpa, testing.io, path, null, .{ @as([]f32, &a), @as([]f32, &b) });

    var a2 = [_]f32{ 0, 0, 0 };
    var b2 = [_]f32{ 0, 0 };
    try load(gpa, testing.io, path, null, .{ @as([]f32, &a2), @as([]f32, &b2) });
    try testing.expectEqualSlices(f32, &a, &a2);
    try testing.expectEqualSlices(f32, &b, &b2);

    var wrong = [_]f32{ 0, 0, 0, 0 };
    try testing.expectError(error.CheckpointMismatch, load(gpa, testing.io, path, null, .{ @as([]f32, &wrong), @as([]f32, &b2) }));
    try testing.expectError(error.CheckpointMismatch, load(gpa, testing.io, path, null, .{@as([]f32, &a2)}));
}

test "a gpu checkpoint round trips through device memory" {
    const gpu = Gpu.init(testing.allocator, 1 << 20) catch |err| switch (err) {
        error.DriverNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer gpu.deinit();
    const path = ".zig-cache/checkpoint-gpu-test.bin";
    defer Io.Dir.cwd().deleteFile(testing.io, path) catch {};

    const device = try gpu.alloc(f32, 4);
    try gpu.upload(device, &[_]f32{ 4, 3, 2, 1 });
    try save(testing.allocator, testing.io, path, gpu, .{device});
    try gpu.fill(device, 0);
    try load(testing.allocator, testing.io, path, gpu, .{device});

    var back: [4]f32 = undefined;
    try gpu.download(&back, device);
    try testing.expectEqualSlices(f32, &.{ 4, 3, 2, 1 }, &back);
}
