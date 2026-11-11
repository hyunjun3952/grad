const std = @import("std");

pub const eps: f32 = 1e-5;

pub const Row = struct { mean: f32, inv_std: f32 };

pub inline fn forwardRow(x: [*]const f32, y: [*]f32, gamma: [*]const f32, beta: [*]const f32, cols: usize) Row {
    var sum: f32 = 0;
    for (0..cols) |j| sum += x[j];
    const mean = sum / @as(f32, @floatFromInt(cols));
    var square: f32 = 0;
    for (0..cols) |j| square += (x[j] - mean) * (x[j] - mean);
    const inv_std = 1 / @sqrt(square / @as(f32, @floatFromInt(cols)) + eps);
    for (0..cols) |j| y[j] = gamma[j] * (x[j] - mean) * inv_std + beta[j];
    return .{ .mean = mean, .inv_std = inv_std };
}

pub inline fn backwardRow(x: [*]const f32, dy: [*]const f32, dx: [*]f32, gamma: [*]const f32, row: Row, cols: usize) void {
    var sum_g: f32 = 0;
    var sum_gx: f32 = 0;
    for (0..cols) |j| {
        const g = dy[j] * gamma[j];
        sum_g += g;
        sum_gx += g * (x[j] - row.mean) * row.inv_std;
    }
    const n: f32 = @floatFromInt(cols);
    for (0..cols) |j| {
        const g = dy[j] * gamma[j];
        dx[j] += row.inv_std * (g - sum_g / n - (x[j] - row.mean) * row.inv_std * sum_gx / n);
    }
}

pub inline fn parameterGradient(x: [*]const f32, dy: [*]const f32, mean: [*]const f32, inv_std: [*]const f32, dgamma: [*]f32, dbeta: [*]f32, rows: usize, cols: usize, j: usize) void {
    var g: f32 = 0;
    var b: f32 = 0;
    for (0..rows) |r| {
        g += dy[r * cols + j] * (x[r * cols + j] - mean[r]) * inv_std[r];
        b += dy[r * cols + j];
    }
    dgamma[j] += g;
    dbeta[j] += b;
}

test "a normalized row has zero mean and unit variance before the affine part" {
    const x = [_]f32{ 1, 2, 3, 4 };
    const gamma = [_]f32{ 1, 1, 1, 1 };
    const beta = [_]f32{ 0, 0, 0, 0 };
    var y: [4]f32 = undefined;
    const row = forwardRow(&x, &y, &gamma, &beta, 4);
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
