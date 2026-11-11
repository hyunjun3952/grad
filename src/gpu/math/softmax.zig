const std = @import("std");
const activation = @import("activation.zig");
const bf16 = @import("bf16.zig");

pub inline fn limitOf(row: usize, cols: usize, period: usize) usize {
    return if (period == 0) cols else row % period + 1;
}

// Rows of x, y, dy and dx may be f32 or bf16 (E); the sums run in f32.
// NOTE: a bf16 row recomputes the exponentials in the last pass instead of storing them, because rounding them before the division would add error.
pub inline fn forwardRow(comptime E: type, x: [*]const E, y: [*]E, cols: usize, limit: usize) void {
    var largest = bf16.load(E, x, 0);
    for (1..limit) |j| largest = @max(largest, bf16.load(E, x, j));
    var total: f32 = 0;
    for (0..limit) |j| {
        const e = activation.exponential(bf16.load(E, x, j) - largest);
        if (E == f32) y[j] = e;
        total += e;
    }
    const inverse = 1 / total;
    for (0..limit) |j| {
        if (E == f32) {
            y[j] *= inverse;
        } else {
            bf16.store(E, y, j, activation.exponential(bf16.load(E, x, j) - largest) * inverse);
        }
    }
    for (limit..cols) |j| bf16.store(E, y, j, 0);
}

pub inline fn backwardRow(comptime E: type, y: [*]const E, dy: [*]const E, dx: [*]E, cols: usize) void {
    var dot: f32 = 0;
    for (0..cols) |j| dot += bf16.load(E, y, j) * bf16.load(E, dy, j);
    for (0..cols) |j| bf16.store(E, dx, j, bf16.load(E, dx, j) + bf16.load(E, y, j) * (bf16.load(E, dy, j) - dot));
}

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
