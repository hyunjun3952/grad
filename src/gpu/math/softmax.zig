const activation = @import("activation.zig");

pub inline fn limitOf(row: usize, cols: usize, period: usize) usize {
    return if (period == 0) cols else row % period + 1;
}

pub inline fn forwardRow(x: [*]const f32, y: [*]f32, cols: usize, limit: usize) void {
    var largest = x[0];
    for (1..limit) |j| largest = @max(largest, x[j]);
    var total: f32 = 0;
    for (0..limit) |j| {
        const e = activation.exponential(x[j] - largest);
        y[j] = e;
        total += e;
    }
    const inverse = 1 / total;
    for (0..limit) |j| y[j] *= inverse;
    for (limit..cols) |j| y[j] = 0;
}

pub inline fn backwardRow(y: [*]const f32, dy: [*]const f32, dx: [*]f32, cols: usize) void {
    var dot: f32 = 0;
    for (0..cols) |j| dot += y[j] * dy[j];
    for (0..cols) |j| dx[j] += y[j] * (dy[j] - dot);
}

const std = @import("std");

test "causal softmax rows sum to one over the allowed columns and zero elsewhere" {
    const x = [_]f32{ 1, 2, 3, 4 };
    var y: [4]f32 = undefined;
    forwardRow(&x, &y, 4, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1), y[0] + y[1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), y[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.26894143), y[0], 1e-5);
    try std.testing.expectEqual(@as(usize, 2), limitOf(5, 6, 4));
    try std.testing.expectEqual(@as(usize, 6), limitOf(5, 6, 0));
}
