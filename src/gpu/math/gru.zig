const act = @import("activation.zig");

// The element-wise part of a GRU step. The input projection gx has rows of 3 * hidden values (update | reset | candidate),
// the recurrent projection gh rows of 2 * hidden (update | reset), and everything else rows of hidden values.
// Both functions work on element i of a (rows, hidden) tensor. Backward recomputes the gates instead of storing them.

pub const needs_gx: u32 = 1;
pub const needs_gh: u32 = 2;
pub const needs_h: u32 = 4;
pub const needs_update: u32 = 8;
pub const needs_gc: u32 = 16;

inline fn gateInputs(gx: [*]const f32, gh: [*]const f32, i: usize, hidden: usize) struct { update: f32, reset: f32 } {
    const row = i / hidden;
    const j = i % hidden;
    return .{
        .update = act.forward(.sigmoid, gx[row * 3 * hidden + j] + gh[row * 2 * hidden + j]),
        .reset = act.forward(.sigmoid, gx[row * 3 * hidden + hidden + j] + gh[row * 2 * hidden + hidden + j]),
    };
}

pub inline fn gatesForward(gx: [*]const f32, gh: [*]const f32, h: [*]const f32, update: [*]f32, reset_h: [*]f32, i: usize, hidden: usize) void {
    const g = gateInputs(gx, gh, i, hidden);
    update[i] = g.update;
    reset_h[i] = g.reset * h[i];
}

pub const GatesGrad = extern struct { gx: [*]f32, gh: [*]f32, h: [*]f32 };

pub inline fn gatesBackward(gx: [*]const f32, gh: [*]const f32, h: [*]const f32, d_update: [*]const f32, d_reset_h: [*]const f32, grad: GatesGrad, flags: u32, i: usize, hidden: usize) void {
    const g = gateInputs(gx, gh, i, hidden);
    const row = i / hidden;
    const j = i % hidden;
    const d_update_pre = d_update[i] * g.update * (1 - g.update);
    const d_reset_pre = d_reset_h[i] * h[i] * g.reset * (1 - g.reset);
    if (flags & needs_gx != 0) {
        grad.gx[row * 3 * hidden + j] += d_update_pre;
        grad.gx[row * 3 * hidden + hidden + j] += d_reset_pre;
    }
    if (flags & needs_gh != 0) {
        grad.gh[row * 2 * hidden + j] += d_update_pre;
        grad.gh[row * 2 * hidden + hidden + j] += d_reset_pre;
    }
    if (flags & needs_h != 0) grad.h[i] += d_reset_h[i] * g.reset;
}

// next = c + u * (h - c) with c = tanh(candidate part of gx + gc).
pub inline fn nextForward(gx: [*]const f32, gc: [*]const f32, update: [*]const f32, h: [*]const f32, next: [*]f32, i: usize, hidden: usize) void {
    const row = i / hidden;
    const j = i % hidden;
    const c = act.forward(.tanh, gx[row * 3 * hidden + 2 * hidden + j] + gc[i]);
    next[i] = c + update[i] * (h[i] - c);
}

pub const NextGrad = extern struct { gx: [*]f32, gc: [*]f32, update: [*]f32, h: [*]f32 };

pub inline fn nextBackward(gx: [*]const f32, gc: [*]const f32, update: [*]const f32, h: [*]const f32, d_next: [*]const f32, grad: NextGrad, flags: u32, i: usize, hidden: usize) void {
    const row = i / hidden;
    const j = i % hidden;
    const c = act.forward(.tanh, gx[row * 3 * hidden + 2 * hidden + j] + gc[i]);
    const d_candidate_pre = d_next[i] * (1 - update[i]) * (1 - c * c);
    if (flags & needs_gx != 0) grad.gx[row * 3 * hidden + 2 * hidden + j] += d_candidate_pre;
    if (flags & needs_gc != 0) grad.gc[i] += d_candidate_pre;
    if (flags & needs_update != 0) grad.update[i] += d_next[i] * (h[i] - c);
    if (flags & needs_h != 0) grad.h[i] += d_next[i] * update[i];
}
