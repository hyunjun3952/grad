const std = @import("std");
const bf16 = @import("bf16.zig");

pub inline fn hash(seed: u64, t: usize) u64 {
    // 0x9E37... is 2^64 / golden ratio and 0xD1B5... a second odd constant: they spread index and seed over 64 bits.
    var z = (t +% 1) *% 0x9E3779B97F4A7C15 +% seed *% 0xD1B54A32D192ED03;
    // Two xor-shift-multiply rounds (0xD6E8... is a mixing multiplier from the xmxmx hash family) make every output bit depend on every input bit.
    z = (z ^ (z >> 32)) *% 0xD6E8FEB86659FD93;
    z = (z ^ (z >> 32)) *% 0xD6E8FEB86659FD93;
    return z ^ (z >> 32);
}

// The element type E may be f32 or bf16; the mask depends only on the seed and the index, so both types drop the same positions.
pub inline fn applyOf(comptime E: type, t: usize, src: [*]const E, dst: [*]E, seed: u64, keep: f32, accumulate: bool) void {
    // Top 24 bits give an exact f32 in [0, 1): 16777216 = 2^24.
    const u = @as(f32, @floatFromInt(hash(seed, t) >> 40)) * (1.0 / 16777216.0);
    const v = if (u < keep) bf16.load(E, src, t) / keep else 0;
    bf16.store(E, dst, t, if (accumulate) bf16.load(E, dst, t) + v else v);
}

pub inline fn apply(t: usize, src: [*]const f32, dst: [*]f32, seed: u64, keep: f32, accumulate: bool) void {
    applyOf(f32, t, src, dst, seed, keep, accumulate);
}
