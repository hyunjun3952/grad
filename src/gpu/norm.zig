const std = @import("std");

pub const eps: f32 = 1e-5;
pub const momentum: f32 = 0.1;
pub const max_channels = 256;

pub inline fn apply(t: usize, x: [*]const f32, y: [*]f32, c: usize, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32) void {
    const ch = t % c;
    y[t] = gamma[ch] * (x[t] - mean[ch]) * inv_std[ch] + beta[ch];
}

pub inline fn gradInput(t: usize, x: [*]const f32, dy: [*]const f32, dx: [*]f32, c: usize, rows: usize, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, sums: [*]const f32) void {
    const ch = t % c;
    const n: f32 = @floatFromInt(rows);
    const x_hat = (x[t] - mean[ch]) * inv_std[ch];
    dx[t] += gamma[ch] * inv_std[ch] / n * (n * dy[t] - sums[ch] - x_hat * sums[c + ch]);
}

pub fn stats(x: []const f32, mean: []f32, inv_std: []f32, running_mean: []f32, running_var: []f32, training: bool) void {
    const c = mean.len;
    std.debug.assert(c <= max_channels);
    if (!training) {
        for (0..c) |ch| {
            mean[ch] = running_mean[ch];
            inv_std[ch] = 1 / @sqrt(running_var[ch] + eps);
        }
        return;
    }
    const rows = x.len / c;
    var sum: [max_channels]f64 = @splat(0);
    for (0..rows) |r| {
        for (0..c) |ch| sum[ch] += x[r * c + ch];
    }
    var square: [max_channels]f64 = @splat(0);
    for (0..rows) |r| {
        for (0..c) |ch| {
            const d = x[r * c + ch] - sum[ch] / @as(f64, @floatFromInt(rows));
            square[ch] += d * d;
        }
    }
    for (0..c) |ch| {
        const mu: f32 = @floatCast(sum[ch] / @as(f64, @floatFromInt(rows)));
        const variance: f32 = @floatCast(square[ch] / @as(f64, @floatFromInt(rows)));
        mean[ch] = mu;
        inv_std[ch] = 1 / @sqrt(variance + eps);
        running_mean[ch] += momentum * (mu - running_mean[ch]);
        running_var[ch] += momentum * (variance - running_var[ch]);
    }
}

pub fn gradSums(x: []const f32, dy: []const f32, mean: []const f32, inv_std: []const f32, sums: []f32, gamma_grad: []f32, beta_grad: []f32) void {
    const c = mean.len;
    std.debug.assert(c <= max_channels);
    var dbeta: [max_channels]f64 = @splat(0);
    var dgamma: [max_channels]f64 = @splat(0);
    for (0..x.len / c) |r| {
        for (0..c) |ch| {
            dbeta[ch] += dy[r * c + ch];
            dgamma[ch] += dy[r * c + ch] * ((x[r * c + ch] - mean[ch]) * inv_std[ch]);
        }
    }
    for (0..c) |ch| {
        sums[ch] = @floatCast(dbeta[ch]);
        sums[c + ch] = @floatCast(dgamma[ch]);
        beta_grad[ch] += sums[ch];
        gamma_grad[ch] += sums[c + ch];
    }
}
