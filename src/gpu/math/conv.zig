const std = @import("std");

pub const Op = enum { im2col, col2im, maxPool, maxPoolBackward, avgPool, avgPoolBackward };

const Index = u32;

pub const Dims = extern struct { n: Index, h: Index, w: Index, c: Index, k: Index, stride: Index = 1, pad: Index = 0 };

pub fn outHeight(d: Dims) Index {
    return (d.h + 2 * d.pad - d.k) / d.stride + 1;
}

pub fn outWidth(d: Dims) Index {
    return (d.w + 2 * d.pad - d.k) / d.stride + 1;
}

pub fn workItems(op: Op, d: Dims) usize {
    return switch (op) {
        .im2col => @as(usize, d.n) * outHeight(d) * outWidth(d) * d.k * d.k * d.c,
        .col2im => @as(usize, d.n) * d.h * d.w * d.c,
        .maxPool, .maxPoolBackward, .avgPool, .avgPoolBackward => @as(usize, d.n) * (d.h / 2) * (d.w / 2) * d.c,
    };
}

pub const Patch = struct { n: Index, top: Index, left: Index };

pub inline fn patchOf(row: Index, d: Dims) Patch {
    const ow = outWidth(d);
    const oh = outHeight(d);
    return .{ .n = row / ow / oh, .top = row / ow % oh * d.stride, .left = row % ow * d.stride };
}

pub inline fn patchValue(x: [*]const f32, patch: Patch, cc: Index, d: Dims) f32 {
    const tap = cc / d.c;
    const py = patch.top + tap / d.k;
    const px = patch.left + tap % d.k;
    const inside = py >= d.pad and px >= d.pad and py - d.pad < d.h and px - d.pad < d.w;
    return if (inside) x[((patch.n * d.h + py - d.pad) * d.w + px - d.pad) * d.c + cc % d.c] else 0;
}

pub inline fn im2colValue(x: [*]const f32, row: Index, cc: Index, d: Dims) f32 {
    return patchValue(x, patchOf(row, d), cc, d);
}
pub inline fn im2col(t: Index, x: [*]const f32, col: [*]f32, d: Dims) void {
    const kc = d.k * d.k * d.c;
    col[t] = im2colValue(x, t / kc, t % kc, d);
}
pub inline fn col2im(t: Index, dcol: [*]const f32, dx: [*]f32, d: Dims) void {
    const kc = d.k * d.k * d.c;
    const ow = outWidth(d);
    const oh = outHeight(d);
    const ch = t % d.c;
    const x = t / d.c % d.w + d.pad;
    const y = t / d.c / d.w % d.h + d.pad;
    const n = t / d.c / d.w / d.h;
    var total: f32 = 0;
    var ky: Index = 0;
    while (ky < d.k) : (ky += 1) {
        if (y < ky or (d.stride != 1 and (y - ky) % d.stride != 0)) continue;
        const oy = if (d.stride == 1) y - ky else (y - ky) / d.stride;
        if (oy >= oh) continue;
        var kx: Index = 0;
        while (kx < d.k) : (kx += 1) {
            if (x < kx or (d.stride != 1 and (x - kx) % d.stride != 0)) continue;
            const ox = if (d.stride == 1) x - kx else (x - kx) / d.stride;
            if (ox >= ow) continue;
            const row = (n * oh + oy) * ow + ox;
            total += dcol[row * kc + (ky * d.k + kx) * d.c + ch];
        }
    }
    dx[t] += total;
}

inline fn windowOffsets(d: Dims) [4]Index {
    return .{ 0, d.c, d.w * d.c, d.w * d.c + d.c };
}

inline fn windowStart(t: Index, d: Dims) Index {
    const ow = d.w / 2;
    const oh = d.h / 2;
    const ch = t % d.c;
    const ox = t / d.c % ow;
    const oy = t / d.c / ow % oh;
    const n = t / d.c / ow / oh;
    return ((n * d.h + 2 * oy) * d.w + 2 * ox) * d.c + ch;
}

inline fn firstMax(x: [*]const f32, t: Index, d: Dims) Index {
    const base = windowStart(t, d);
    var best = base;
    for (windowOffsets(d)) |off| {
        if (x[base + off] > x[best]) best = base + off;
    }
    return best;
}

pub inline fn avgPool(t: Index, x: [*]const f32, out: [*]f32, d: Dims) void {
    const base = windowStart(t, d);
    var total: f32 = 0;
    for (windowOffsets(d)) |off| total += x[base + off];
    out[t] = total / 4;
}

pub inline fn avgPoolBackward(t: Index, gout: [*]const f32, gin: [*]f32, d: Dims) void {
    const base = windowStart(t, d);
    for (windowOffsets(d)) |off| gin[base + off] += gout[t] / 4;
}

pub inline fn maxPool(t: Index, x: [*]const f32, out: [*]f32, d: Dims) void {
    out[t] = x[firstMax(x, t, d)];
}

pub inline fn maxPoolBackward(t: Index, x: [*]const f32, gout: [*]const f32, gin: [*]f32, d: Dims) void {
    gin[firstMax(x, t, d)] += gout[t];
}

test "max maxPool picks the first maximum, so a tied window routes its gradient to one element" {
    const x = [_]f32{ 1, 1, 1, 1 };
    const gout = [_]f32{5};
    var gin = [_]f32{ 0, 0, 0, 0 };
    const d: Dims = .{ .n = 1, .h = 2, .w = 2, .c = 1, .k = 2 };
    maxPoolBackward(0, &x, &gout, &gin, d);
    try std.testing.expectEqualSlices(f32, &.{ 5, 0, 0, 0 }, &gin);

    const y = [_]f32{ 1, 3, 3, 2 };
    var out: [1]f32 = undefined;
    maxPool(0, &y, &out, d);
    try std.testing.expectEqual(@as(f32, 3), out[0]);
    gin = .{ 0, 0, 0, 0 };
    maxPoolBackward(0, &y, &gout, &gin, d);
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 0, 0 }, &gin);
}
