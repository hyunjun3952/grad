const std = @import("std");
const bf16 = @import("bf16.zig");

pub const eps: f32 = 1e-5;
pub const momentum: f32 = 0.1;
/// Most channels the CPU batch-norm kernels hold in stack buffers; the wide CIFAR network needs 384.
pub const max_channels = 512;

pub const Affine = extern struct { mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32 };
pub const Stats = extern struct { mean: [*]f32, inv_std: [*]f32, running_mean: [*]f32, running_var: [*]f32 };
/// Layout.flag of apply and gradInput: the bits below. batchNormFinish uses the same field as its training switch.
pub const Layout = extern struct { len: u32, channels: u32, flag: u32 };
/// Bit of Layout.flag: ReLU after the affine map (backward: the output is the mask).
pub const relu_flag: u32 = 1;
/// Bit of Layout.flag: a residual tensor is added before the ReLU (backward: its gradient is accumulated into dr).
pub const residual_flag: u32 = 2;
/// Bit of Layout.flag: the input gradient is stored, not accumulated (the target is a fresh buffer, not a zeroed one).
pub const store_flag: u32 = 4;
/// Bit of Layout.flag: the ReLU mask is recomputed from the input and the affine map instead of read from mask
/// (for a virtual normalized map that was never written); only meaningful with relu_flag.
pub const mask_x_flag: u32 = 8;
pub fn GradIo(comptime E: type) type {
    return extern struct { x: [*]const E, dy: [*]const E, dx: [*]E, sums: [*]const f32, mask: [*]const E, dr: [*]E };
}
pub const GradOut = struct { sums: []f32, gamma: []f32, beta: []f32 };

pub inline fn apply(comptime E: type, t: usize, x: [*]const E, residual: [*]const E, y: [*]E, p: Affine, l: Layout) void {
    const ch = t % l.channels;
    var v = p.gamma[ch] * (bf16.load(E, x, t) - p.mean[ch]) * p.inv_std[ch] + p.beta[ch];
    if (l.flag & residual_flag != 0) v += bf16.load(E, residual, t);
    bf16.store(E, y, t, if (l.flag & relu_flag != 0) @max(v, 0) else v);
}

pub inline fn gradInput(comptime E: type, t: usize, io: GradIo(E), p: Affine, l: Layout) void {
    const ch = t % l.channels;
    const n: f32 = @floatFromInt(l.len / l.channels);
    const x_hat = (bf16.load(E, io.x, t) - p.mean[ch]) * p.inv_std[ch];
    const live = if (l.flag & mask_x_flag != 0) p.gamma[ch] * x_hat + p.beta[ch] > 0 else bf16.load(E, io.mask, t) > 0;
    const g = if (l.flag & relu_flag != 0 and !live) 0 else bf16.load(E, io.dy, t);
    if (l.flag & residual_flag != 0) bf16.store(E, io.dr, t, bf16.load(E, io.dr, t) + g);
    const dx = p.gamma[ch] * p.inv_std[ch] / n * (n * g - io.sums[ch] - x_hat * io.sums[l.channels + ch]);
    bf16.store(E, io.dx, t, if (l.flag & store_flag != 0) dx else bf16.load(E, io.dx, t) + dx);
}

pub inline fn applyQuad(comptime E: type, t: usize, x: [*]const E, residual: [*]const E, y: [*]E, p: Affine, l: Layout) void {
    const ch = t % l.channels;
    const gamma: bf16.Quad = p.gamma[ch..][0..4].*;
    const mean: bf16.Quad = p.mean[ch..][0..4].*;
    const inv_std: bf16.Quad = p.inv_std[ch..][0..4].*;
    const beta: bf16.Quad = p.beta[ch..][0..4].*;
    var v = gamma * (bf16.loadQuad(E, x, t) - mean) * inv_std + beta;
    if (l.flag & residual_flag != 0) v += bf16.loadQuad(E, residual, t);
    bf16.storeQuad(E, y, t, if (l.flag & relu_flag != 0) @max(v, @as(bf16.Quad, @splat(0))) else v);
}

pub inline fn gradInputQuad(comptime E: type, t: usize, io: GradIo(E), p: Affine, l: Layout) void {
    const ch = t % l.channels;
    const n: bf16.Quad = @splat(@floatFromInt(l.len / l.channels));
    const gamma: bf16.Quad = p.gamma[ch..][0..4].*;
    const mean: bf16.Quad = p.mean[ch..][0..4].*;
    const inv_std: bf16.Quad = p.inv_std[ch..][0..4].*;
    const sum_dy: bf16.Quad = io.sums[ch..][0..4].*;
    const sum_dyx: bf16.Quad = io.sums[l.channels + ch ..][0..4].*;
    const x_hat = (bf16.loadQuad(E, io.x, t) - mean) * inv_std;
    const dy = bf16.loadQuad(E, io.dy, t);
    const live = if (l.flag & mask_x_flag != 0) gamma * x_hat + @as(bf16.Quad, p.beta[ch..][0..4].*) > @as(bf16.Quad, @splat(0)) else bf16.loadQuad(E, io.mask, t) > @as(bf16.Quad, @splat(0));
    const g = if (l.flag & relu_flag != 0) @select(f32, live, dy, @as(bf16.Quad, @splat(0))) else dy;
    if (l.flag & residual_flag != 0) bf16.storeQuad(E, io.dr, t, bf16.loadQuad(E, io.dr, t) + g);
    const dx = gamma * inv_std / n * (n * g - sum_dy - x_hat * sum_dyx);
    bf16.storeQuad(E, io.dx, t, if (l.flag & store_flag != 0) dx else bf16.loadQuad(E, io.dx, t) + dx);
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
