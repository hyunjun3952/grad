const std = @import("std");

pub inline fn hash(seed: u64, t: usize) u64 {
    // 0x9E37... is 2^64 / golden ratio and 0xD1B5... a second odd constant: they spread index and seed over 64 bits.
    var z = (t +% 1) *% 0x9E3779B97F4A7C15 +% seed *% 0xD1B54A32D192ED03;
    // Two xor-shift-multiply rounds (0xD6E8... is a mixing multiplier from the xmxmx hash family) make every output bit depend on every input bit.
    z = (z ^ (z >> 32)) *% 0xD6E8FEB86659FD93;
    z = (z ^ (z >> 32)) *% 0xD6E8FEB86659FD93;
    return z ^ (z >> 32);
}

pub inline fn apply(src: [*]const f32, dst: [*]f32, t: usize, seed: u64, keep: f32, accumulate: bool) void {
    // Top 24 bits give an exact f32 in [0, 1): 16777216 = 2^24.
    const u = @as(f32, @floatFromInt(hash(seed, t) >> 40)) * (1.0 / 16777216.0);
    const v = if (u < keep) src[t] / keep else 0;
    dst[t] = if (accumulate) dst[t] + v else v;
}
test "the kept fraction follows the keep probability and kept values are scaled" {
    var src: [20000]f32 = undefined;
    var dst: [20000]f32 = undefined;
    @memset(&src, 1);
    for (0..src.len) |t| apply(&src, &dst, t, 7, 0.8, false);
    var kept: usize = 0;
    for (dst) |v| {
        if (v != 0) {
            kept += 1;
            try std.testing.expectApproxEqAbs(@as(f32, 1.25), v, 1e-6);
        }
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), @as(f32, @floatFromInt(kept)) / src.len, 0.01);
}
