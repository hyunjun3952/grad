const std = @import("std");
const lib = @import("grad");
const maxPoolBackward = lib.math.conv.maxPoolBackward;
const maxPool = lib.math.conv.maxPool;
const Dims = lib.math.conv.Dims;

test "max maxPool picks the first maximum, so a tied window routes its gradient to one element" {
    const d: Dims = .{ .n = 1, .h = 2, .w = 2, .c = 1, .k = 2 };
    const gout = [_]f32{5};
    var out: [1]f32 = undefined;
    var index: [1]u32 = undefined;
    const tied = [_]f32{ 1, 1, 1, 1 };
    var gin = [_]f32{ 0, 0, 0, 0 };
    maxPool(0, &tied, &out, &index, d);
    for (0..4) |i| maxPoolBackward(@intCast(i), &index, &gout, &gin, d);
    try std.testing.expectEqualSlices(f32, &.{ 5, 0, 0, 0 }, &gin);

    const y = [_]f32{ 1, 3, 3, 2 };
    maxPool(0, &y, &out, &index, d);
    try std.testing.expectEqual(@as(f32, 3), out[0]);
    gin = .{ 0, 0, 0, 0 };
    for (0..4) |i| maxPoolBackward(@intCast(i), &index, &gout, &gin, d);
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 0, 0 }, &gin);
}
