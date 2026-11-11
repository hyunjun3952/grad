const std = @import("std");
const bf16 = @import("bf16.zig");

pub const eps: f32 = 1e-5;

pub const Row = struct { mean: f32, inv_std: f32 };

// Activations x, y, dy and dx may be f32 or bf16 (E); the scale, shift and statistics are always f32 and the sums run in f32.
pub inline fn forwardRow(comptime E: type, x: [*]const E, y: [*]E, gamma: [*]const f32, beta: [*]const f32, cols: usize) Row {
    var sum: f32 = 0;
    for (0..cols) |j| sum += bf16.load(E, x, j);
    const mean = sum / @as(f32, @floatFromInt(cols));
    var square: f32 = 0;
    for (0..cols) |j| square += (bf16.load(E, x, j) - mean) * (bf16.load(E, x, j) - mean);
    const inv_std = 1 / @sqrt(square / @as(f32, @floatFromInt(cols)) + eps);
    for (0..cols) |j| bf16.store(E, y, j, gamma[j] * (bf16.load(E, x, j) - mean) * inv_std + beta[j]);
    return .{ .mean = mean, .inv_std = inv_std };
}

pub inline fn backwardRow(comptime E: type, x: [*]const E, dy: [*]const E, dx: [*]E, gamma: [*]const f32, row: Row, cols: usize) void {
    var sum_g: f32 = 0;
    var sum_gx: f32 = 0;
    for (0..cols) |j| {
        const g = bf16.load(E, dy, j) * gamma[j];
        sum_g += g;
        sum_gx += g * (bf16.load(E, x, j) - row.mean) * row.inv_std;
    }
    const n: f32 = @floatFromInt(cols);
    for (0..cols) |j| {
        const g = bf16.load(E, dy, j) * gamma[j];
        bf16.store(E, dx, j, bf16.load(E, dx, j) + row.inv_std * (g - sum_g / n - (bf16.load(E, x, j) - row.mean) * row.inv_std * sum_gx / n));
    }
}

pub inline fn parameterGradient(comptime E: type, x: [*]const E, dy: [*]const E, mean: [*]const f32, inv_std: [*]const f32, dgamma: [*]f32, dbeta: [*]f32, rows: usize, cols: usize, j: usize) void {
    var g: f32 = 0;
    var b: f32 = 0;
    for (0..rows) |r| {
        g += bf16.load(E, dy, r * cols + j) * (bf16.load(E, x, r * cols + j) - mean[r]) * inv_std[r];
        b += bf16.load(E, dy, r * cols + j);
    }
    dgamma[j] += g;
    dbeta[j] += b;
}

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
