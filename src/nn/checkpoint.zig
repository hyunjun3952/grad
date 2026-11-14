const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Gpu = @import("../gpu/gpu.zig").Gpu;

const magic: u32 = 0x32445247;

fn headerBytes(counters: usize) usize {
    return 12 + 4 * counters;
}

pub fn save(gpa: Allocator, io: Io, path: []const u8, gpu: ?*Gpu, state: anytype, counters: []const u32) !void {
    var total: usize = headerBytes(counters.len);
    inline for (state) |s| total += 4 + 4 * s.len;
    const bytes = try gpa.alignedAlloc(u8, .@"4", total);
    defer gpa.free(bytes);

    std.mem.writeInt(u32, bytes[0..4], magic, .little);
    std.mem.writeInt(u32, bytes[4..8], state.len, .little);
    std.mem.writeInt(u32, bytes[8..12], @intCast(counters.len), .little);
    for (counters, 0..) |c, i| std.mem.writeInt(u32, bytes[12 + 4 * i ..][0..4], c, .little);
    var at: usize = headerBytes(counters.len);
    inline for (state) |s| {
        std.mem.writeInt(u32, bytes[at..][0..4], @intCast(s.len), .little);
        const values: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes[at + 4 ..][0 .. 4 * s.len]));
        if (gpu) |dev| try dev.download(values, s) else @memcpy(values, s);
        at += 4 + 4 * s.len;
    }
    // NOTE: written next to the target and renamed over it, so a crash during the write keeps the previous checkpoint.
    var temp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const temp = try std.fmt.bufPrint(&temp_buf, "{s}.tmp", .{path});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = temp, .data = bytes });
    try Io.Dir.rename(Io.Dir.cwd(), temp, Io.Dir.cwd(), path, io);
}

pub fn load(gpa: Allocator, io: Io, path: []const u8, gpu: ?*Gpu, state: anytype, counters: []u32) !void {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31));
    defer gpa.free(bytes);
    if (bytes.len < 12 or std.mem.readInt(u32, bytes[0..4], .little) != magic) return error.BadCheckpoint;
    if (std.mem.readInt(u32, bytes[4..8], .little) != state.len) return error.CheckpointMismatch;
    if (std.mem.readInt(u32, bytes[8..12], .little) != counters.len) return error.CheckpointMismatch;
    if (bytes.len < headerBytes(counters.len)) return error.BadCheckpoint;

    // NOTE: every size is checked before anything is written, so a rejected file leaves the model and counters as they were.
    var check: usize = headerBytes(counters.len);
    inline for (state) |s| {
        if (check + 4 > bytes.len or std.mem.readInt(u32, bytes[check..][0..4], .little) != s.len) return error.CheckpointMismatch;
        if (check + 4 + 4 * s.len > bytes.len) return error.BadCheckpoint;
        check += 4 + 4 * s.len;
    }

    for (counters, 0..) |*c, i| c.* = std.mem.readInt(u32, bytes[12 + 4 * i ..][0..4], .little);
    var at: usize = headerBytes(counters.len);
    inline for (state) |s| {
        const values = try gpa.alloc(f32, s.len);
        defer gpa.free(values);
        @memcpy(std.mem.sliceAsBytes(values), bytes[at + 4 ..][0 .. 4 * s.len]);
        if (gpu) |dev| try dev.upload(s, values) else @memcpy(s, values);
        at += 4 + 4 * s.len;
    }
}
