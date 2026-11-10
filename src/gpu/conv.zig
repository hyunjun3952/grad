const std = @import("std");

pub const Op = enum { im2col, col2im, maxPool, maxPoolBackward };

pub const Dims = struct { n: usize, h: usize, w: usize, c: usize, k: usize };

pub fn workItems(op: Op, d: Dims) usize {
    return switch (op) {
        .im2col => d.n * (d.h - d.k + 1) * (d.w - d.k + 1) * d.k * d.k * d.c,
        .col2im => d.n * d.h * d.w * d.c,
        .maxPool, .maxPoolBackward => d.n * (d.h / 2) * (d.w / 2) * d.c,
    };
}

pub inline fn im2col(x: [*]const f32, col: [*]f32, t: usize, d: Dims) void {
    const kc = d.k * d.k * d.c;
    const ow = d.w - d.k + 1;
    const oh = d.h - d.k + 1;
    const row = t / kc;
    const cc = t % kc;
    const ch = cc % d.c;
    const kx = (cc / d.c) % d.k;
    const ky = cc / d.c / d.k;
    const ox = row % ow;
    const oy = row / ow % oh;
    const n = row / ow / oh;
    col[t] = x[((n * d.h + oy + ky) * d.w + ox + kx) * d.c + ch];
}

pub inline fn col2im(dcol: [*]const f32, dx: [*]f32, t: usize, d: Dims) void {
    const kc = d.k * d.k * d.c;
    const ow = d.w - d.k + 1;
    const oh = d.h - d.k + 1;
    const ch = t % d.c;
    const x = t / d.c % d.w;
    const y = t / d.c / d.w % d.h;
    const n = t / d.c / d.w / d.h;
    var total: f32 = 0;
    for (0..d.k) |ky| {
        if (y < ky or y - ky >= oh) continue;
        for (0..d.k) |kx| {
            if (x < kx or x - kx >= ow) continue;
            const row = (n * oh + y - ky) * ow + x - kx;
            total += dcol[row * kc + (ky * d.k + kx) * d.c + ch];
        }
    }
    dx[t] += total;
}

inline fn firstMax(x: [*]const f32, t: usize, d: Dims) usize {
    const ow = d.w / 2;
    const oh = d.h / 2;
    const ch = t % d.c;
    const ox = t / d.c % ow;
    const oy = t / d.c / ow % oh;
    const n = t / d.c / ow / oh;
    const base = ((n * d.h + 2 * oy) * d.w + 2 * ox) * d.c + ch;
    var best = base;
    for ([_]usize{ d.c, d.w * d.c, d.w * d.c + d.c }) |off| {
        if (x[base + off] > x[best]) best = base + off;
    }
    return best;
}

pub inline fn maxPool(x: [*]const f32, out: [*]f32, t: usize, d: Dims) void {
    out[t] = x[firstMax(x, t, d)];
}

pub inline fn maxPoolBackward(x: [*]const f32, gout: [*]const f32, gin: [*]f32, t: usize, d: Dims) void {
    gin[firstMax(x, t, d)] += gout[t];
}

test "max maxPool picks the first maximum, so a tied window routes its gradient to one element" {
    const x = [_]f32{ 1, 1, 1, 1 };
    const gout = [_]f32{5};
    var gin = [_]f32{ 0, 0, 0, 0 };
    const d: Dims = .{ .n = 1, .h = 2, .w = 2, .c = 1, .k = 2 };
    maxPoolBackward(&x, &gout, &gin, 0, d);
    try std.testing.expectEqualSlices(f32, &.{ 5, 0, 0, 0 }, &gin);

    const y = [_]f32{ 1, 3, 3, 2 };
    var out: [1]f32 = undefined;
    maxPool(&y, &out, 0, d);
    try std.testing.expectEqual(@as(f32, 3), out[0]);
    gin = .{ 0, 0, 0, 0 };
    maxPoolBackward(&y, &gout, &gin, 0, d);
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 0, 0 }, &gin);
}
