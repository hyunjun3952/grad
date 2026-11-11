const std = @import("std");
const bf16 = @import("bf16.zig");

pub const eps: f32 = 1e-5;
pub const momentum: f32 = 0.1;
pub const max_channels = 256;

pub const Affine = extern struct { mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32 };
pub const Stats = extern struct { mean: [*]f32, inv_std: [*]f32, running_mean: [*]f32, running_var: [*]f32 };
pub const Layout = extern struct { len: u32, channels: u32, flag: u32 };
pub fn GradIo(comptime E: type) type {
    return extern struct { x: [*]const E, dy: [*]const E, dx: [*]E, sums: [*]const f32, mask: [*]const E };
}
pub const GradOut = struct { sums: []f32, gamma: []f32, beta: []f32 };

pub inline fn apply(comptime E: type, t: usize, x: [*]const E, y: [*]E, p: Affine, l: Layout) void {
    const ch = t % l.channels;
    const v = p.gamma[ch] * (bf16.load(E, x, t) - p.mean[ch]) * p.inv_std[ch] + p.beta[ch];
    bf16.store(E, y, t, if (l.flag != 0) @max(v, 0) else v);
}

pub inline fn gradInput(comptime E: type, t: usize, io: GradIo(E), p: Affine, l: Layout) void {
    const ch = t % l.channels;
    const n: f32 = @floatFromInt(l.len / l.channels);
    const x_hat = (bf16.load(E, io.x, t) - p.mean[ch]) * p.inv_std[ch];
    const g = if (l.flag != 0 and bf16.load(E, io.mask, t) <= 0) 0 else bf16.load(E, io.dy, t);
    bf16.store(E, io.dx, t, bf16.load(E, io.dx, t) + p.gamma[ch] * p.inv_std[ch] / n * (n * g - io.sums[ch] - x_hat * io.sums[l.channels + ch]));
}

pub fn stats(x: []const f32, st: Stats, c: usize, training: bool) void {
    std.debug.assert(c <= max_channels);
    if (!training) {
        for (0..c) |ch| {
            st.mean[ch] = st.running_mean[ch];
            st.inv_std[ch] = 1 / @sqrt(st.running_var[ch] + eps);
        }
        return;
    }
    const rows = x.len / c;
    const count: f64 = @floatFromInt(rows);
    var sum: [max_channels]f64 = @splat(0);
    for (0..rows) |r| {
        for (0..c) |ch| sum[ch] += x[r * c + ch];
    }
    var square: [max_channels]f64 = @splat(0);
    for (0..rows) |r| {
        for (0..c) |ch| {
            const d = x[r * c + ch] - sum[ch] / count;
            square[ch] += d * d;
        }
    }
    for (0..c) |ch| {
        const mu: f32 = @floatCast(sum[ch] / count);
        const variance: f32 = @floatCast(square[ch] / count);
        st.mean[ch] = mu;
        st.inv_std[ch] = 1 / @sqrt(variance + eps);
        st.running_mean[ch] += momentum * (mu - st.running_mean[ch]);
        st.running_var[ch] += momentum * (variance - st.running_var[ch]);
    }
}

pub fn gradSums(x: []const f32, dy: []const f32, p: Affine, mask: []const f32, out: GradOut) void {
    const c = out.beta.len;
    std.debug.assert(c <= max_channels);
    var dbeta: [max_channels]f64 = @splat(0);
    var dgamma: [max_channels]f64 = @splat(0);
    for (0..x.len / c) |r| {
        for (0..c) |ch| {
            const g = if (mask.len != 0 and mask[r * c + ch] <= 0) 0 else dy[r * c + ch];
            dbeta[ch] += g;
            dgamma[ch] += g * ((x[r * c + ch] - p.mean[ch]) * p.inv_std[ch]);
        }
    }
    for (0..c) |ch| {
        out.sums[ch] = @floatCast(dbeta[ch]);
        out.sums[c + ch] = @floatCast(dgamma[ch]);
        out.beta[ch] += out.sums[ch];
        out.gamma[ch] += out.sums[c + ch];
    }
}
