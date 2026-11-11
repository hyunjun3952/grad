const std = @import("std");
const lib = @import("grad");
const limitOf = lib.math.softmax.limitOf;
const forwardRow = lib.math.softmax.forwardRow;

test "causal softmax rows sum to one over the allowed columns and zero elsewhere" {
    const x = [_]f32{ 1, 2, 3, 4 };
    var y: [4]f32 = undefined;
    forwardRow(f32, &x, &y, 4, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1), y[0] + y[1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), y[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.26894143), y[0], 1e-5);
    try std.testing.expectEqual(@as(usize, 2), limitOf(5, 6, 4));
    try std.testing.expectEqual(@as(usize, 6), limitOf(5, 6, 0));
}
