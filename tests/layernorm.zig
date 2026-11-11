const std = @import("std");
const lib = @import("grad");
const forwardRow = lib.math.layernorm.forwardRow;

test "a normalized row has zero mean and unit variance before the affine part" {
    const x = [_]f32{ 1, 2, 3, 4 };
    const gamma = [_]f32{ 1, 1, 1, 1 };
    const beta = [_]f32{ 0, 0, 0, 0 };
    var y: [4]f32 = undefined;
    const row = forwardRow(f32, &x, &y, &gamma, &beta, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), row.mean, 1e-6);
    var mean: f32 = 0;
    var square: f32 = 0;
    for (y) |v| {
        mean += v / 4;
        square += v * v / 4;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0), mean, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), square, 1e-3);
}
