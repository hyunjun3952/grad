const std = @import("std");
const bf16 = @import("bf16.zig");

pub const Op = enum { im2col, col2im, col2imHalf, maxPool, maxPoolBackward, avgPool, avgPoolBackward, groupedForward, groupedInputGrad, groupedWeightGrad };

const Index = u32;

pub const Dims = extern struct { n: Index, h: Index, w: Index, c: Index, k: Index, stride: Index = 1, pad: Index = 0, dilation: Index = 1, groups: Index = 1, c_out: Index = 0 };

pub fn outHeight(d: Dims) Index {
    return (d.h + 2 * d.pad - d.dilation * (d.k - 1) - 1) / d.stride + 1;
}

pub fn outWidth(d: Dims) Index {
    return (d.w + 2 * d.pad - d.dilation * (d.k - 1) - 1) / d.stride + 1;
}

pub fn workItems(op: Op, d: Dims) usize {
    return switch (op) {
        .im2col => @as(usize, d.n) * outHeight(d) * outWidth(d) * d.k * d.k * d.c,
        .col2im, .col2imHalf => @as(usize, d.n) * d.h * d.w * d.c,
        .maxPool, .avgPool => @as(usize, d.n) * outHeight(d) * outWidth(d) * d.c,
        .maxPoolBackward => if (d.stride >= d.k) @as(usize, d.n) * outHeight(d) * outWidth(d) * d.c else @as(usize, d.n) * d.h * d.w * d.c,
        .avgPoolBackward => @as(usize, d.n) * d.h * d.w * d.c,
        .groupedForward => @as(usize, d.n) * outHeight(d) * outWidth(d) * d.c_out,
        .groupedInputGrad => @as(usize, d.n) * d.h * d.w * d.c,
        .groupedWeightGrad => @as(usize, d.k) * d.k * (d.c / d.groups) * d.c_out,
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

pub inline fn patchValues(x: [*]const f32, patch: Patch, cc: Index, d: Dims) @Vector(4, f32) {
    const tap = cc / d.c;
    const py = patch.top + tap / d.k;
    const px = patch.left + tap % d.k;
    const inside = py >= d.pad and px >= d.pad and py - d.pad < d.h and px - d.pad < d.w;
    if (!inside) return @splat(0);
    const at = ((patch.n * d.h + py - d.pad) * d.w + px - d.pad) * d.c + cc % d.c;
    const source: *align(16) const [4]f32 = @ptrCast(@alignCast(x + at));
    return source.*;
}

pub inline fn im2colValue(x: [*]const f32, row: Index, cc: Index, d: Dims) f32 {
    return patchValue(x, patchOf(row, d), cc, d);
}
pub inline fn im2col(t: Index, x: [*]const f32, col: [*]f32, d: Dims) void {
    const kc = d.k * d.k * d.c;
    col[t] = im2colValue(x, t / kc, t % kc, d);
}
inline fn col2imSum(comptime T: type, t: Index, dcol: [*]const T, d: Dims) f32 {
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
            const value = dcol[row * kc + (ky * d.k + kx) * d.c + ch];
            total += if (T == u16) bf16.decode(value) else value;
        }
    }
    return total;
}

pub inline fn col2im(t: Index, dcol: [*]const f32, dx: [*]f32, d: Dims) void {
    dx[t] += col2imSum(f32, t, dcol, d);
}

pub inline fn col2imHalf(t: Index, dcol: [*]const u16, dx: [*]f32, d: Dims) void {
    dx[t] += col2imSum(u16, t, dcol, d);
}
const Window = struct { n: Index, oy: Index, ox: Index, ch: Index };

inline fn windowOf(t: Index, d: Dims) Window {
    const ow = outWidth(d);
    const oh = outHeight(d);
    return .{ .ch = t % d.c, .ox = t / d.c % ow, .oy = t / d.c / ow % oh, .n = t / d.c / ow / oh };
}

const Bounds = struct { y0: Index, y1: Index, x0: Index, x1: Index };

inline fn boundsOf(w: Window, d: Dims) Bounds {
    const top = w.oy * d.stride;
    const left = w.ox * d.stride;
    return .{
        .y0 = if (top < d.pad) 0 else top - d.pad,
        .y1 = @min(top + d.k - d.pad, d.h),
        .x0 = if (left < d.pad) 0 else left - d.pad,
        .x1 = @min(left + d.k - d.pad, d.w),
    };
}

inline fn inputIndex(n: Index, y: Index, x: Index, ch: Index, d: Dims) Index {
    return ((n * d.h + y) * d.w + x) * d.c + ch;
}

inline fn firstMax(x: [*]const f32, w: Window, d: Dims) Index {
    const b = boundsOf(w, d);
    var best = inputIndex(w.n, b.y0, b.x0, w.ch, d);
    var y = b.y0;
    while (y < b.y1) : (y += 1) {
        var xx = b.x0;
        while (xx < b.x1) : (xx += 1) {
            const at = inputIndex(w.n, y, xx, w.ch, d);
            if (x[at] > x[best]) best = at;
        }
    }
    return best;
}

inline fn windowCount(w: Window, d: Dims) f32 {
    const b = boundsOf(w, d);
    return @floatFromInt((b.y1 - b.y0) * (b.x1 - b.x0));
}

pub inline fn avgPool(t: Index, x: [*]const f32, out: [*]f32, d: Dims) void {
    const w = windowOf(t, d);
    const b = boundsOf(w, d);
    var total: f32 = 0;
    var y = b.y0;
    while (y < b.y1) : (y += 1) {
        var xx = b.x0;
        while (xx < b.x1) : (xx += 1) total += x[inputIndex(w.n, y, xx, w.ch, d)];
    }
    out[t] = total / windowCount(w, d);
}

pub inline fn maxPool(t: Index, x: [*]const f32, out: [*]f32, index: [*]u32, d: Dims) void {
    const winner = firstMax(x, windowOf(t, d), d);
    out[t] = x[winner];
    index[t] = winner;
}

inline fn firstWindow(position: Index, d: Dims) Index {
    const shifted = position + d.pad + 1;
    return if (shifted > d.k) (shifted - d.k + d.stride - 1) / d.stride else 0;
}

inline fn pooledGradient(comptime winner: bool, t: Index, index: [*]const u32, gout: [*]const f32, d: Dims) f32 {
    const ch = t % d.c;
    const xx = t / d.c % d.w;
    const yy = t / d.c / d.w % d.h;
    const n = t / d.c / d.w / d.h;
    const oh = outHeight(d);
    const ow = outWidth(d);
    const last_y = @min((yy + d.pad) / d.stride, oh - 1);
    const last_x = @min((xx + d.pad) / d.stride, ow - 1);
    var total: f32 = 0;
    var oy = firstWindow(yy, d);
    while (oy <= last_y) : (oy += 1) {
        var ox = firstWindow(xx, d);
        while (ox <= last_x) : (ox += 1) {
            const at = ((n * oh + oy) * ow + ox) * d.c + ch;
            if (winner) {
                if (index[at] == t) total += gout[at];
            } else {
                total += gout[at] / windowCount(.{ .n = n, .oy = oy, .ox = ox, .ch = ch }, d);
            }
        }
    }
    return total;
}

pub inline fn avgPoolBackward(t: Index, gout: [*]const f32, gin: [*]f32, d: Dims) void {
    gin[t] += pooledGradient(false, t, undefined, gout, d);
}

pub inline fn maxPoolBackward(t: Index, index: [*]const u32, gout: [*]const f32, gin: [*]f32, d: Dims) void {
    if (d.stride >= d.k) {
        gin[index[t]] += gout[t];
        return;
    }
    gin[t] += pooledGradient(true, t, index, gout, d);
}
pub inline fn groupedForward(t: Index, x: [*]const f32, w: [*]const f32, out: [*]f32, d: Dims) void {
    const ow = outWidth(d);
    const oh = outHeight(d);
    const cg = d.c / d.groups;
    const oc = t % d.c_out;
    const ox = t / d.c_out % ow;
    const oy = t / d.c_out / ow % oh;
    const n = t / d.c_out / ow / oh;
    const base = oc / (d.c_out / d.groups) * cg;
    var total: f32 = 0;
    var ky: Index = 0;
    while (ky < d.k) : (ky += 1) {
        const py = oy * d.stride + ky * d.dilation;
        if (py < d.pad or py - d.pad >= d.h) continue;
        var kx: Index = 0;
        while (kx < d.k) : (kx += 1) {
            const px = ox * d.stride + kx * d.dilation;
            if (px < d.pad or px - d.pad >= d.w) continue;
            const source = ((n * d.h + py - d.pad) * d.w + px - d.pad) * d.c + base;
            const weights = (ky * d.k + kx) * cg;
            var ic: Index = 0;
            while (ic < cg) : (ic += 1) total += x[source + ic] * w[(weights + ic) * d.c_out + oc];
        }
    }
    out[t] = total;
}

pub inline fn groupedInputGrad(t: Index, dy: [*]const f32, w: [*]const f32, dx: [*]f32, d: Dims) void {
    const ow = outWidth(d);
    const oh = outHeight(d);
    const cg = d.c / d.groups;
    const per_group = d.c_out / d.groups;
    const ch = t % d.c;
    const xx = t / d.c % d.w + d.pad;
    const yy = t / d.c / d.w % d.h + d.pad;
    const n = t / d.c / d.w / d.h;
    const group = ch / cg;
    var total: f32 = 0;
    var ky: Index = 0;
    while (ky < d.k) : (ky += 1) {
        const shift_y = ky * d.dilation;
        if (yy < shift_y or (yy - shift_y) % d.stride != 0) continue;
        const oy = (yy - shift_y) / d.stride;
        if (oy >= oh) continue;
        var kx: Index = 0;
        while (kx < d.k) : (kx += 1) {
            const shift_x = kx * d.dilation;
            if (xx < shift_x or (xx - shift_x) % d.stride != 0) continue;
            const ox = (xx - shift_x) / d.stride;
            if (ox >= ow) continue;
            const grad = ((n * oh + oy) * ow + ox) * d.c_out + group * per_group;
            const weights = ((ky * d.k + kx) * cg + ch % cg) * d.c_out + group * per_group;
            var oc: Index = 0;
            while (oc < per_group) : (oc += 1) total += dy[grad + oc] * w[weights + oc];
        }
    }
    dx[t] += total;
}

pub inline fn groupedWeightGrad(t: Index, x: [*]const f32, dy: [*]const f32, dw: [*]f32, d: Dims) void {
    const ow = outWidth(d);
    const oh = outHeight(d);
    const cg = d.c / d.groups;
    const oc = t % d.c_out;
    const ic = t / d.c_out % cg;
    const kx = t / d.c_out / cg % d.k;
    const ky = t / d.c_out / cg / d.k;
    const channel = oc / (d.c_out / d.groups) * cg + ic;
    var total: f32 = 0;
    var n: Index = 0;
    while (n < d.n) : (n += 1) {
        var oy: Index = 0;
        while (oy < oh) : (oy += 1) {
            const py = oy * d.stride + ky * d.dilation;
            if (py < d.pad or py - d.pad >= d.h) continue;
            var ox: Index = 0;
            while (ox < ow) : (ox += 1) {
                const px = ox * d.stride + kx * d.dilation;
                if (px < d.pad or px - d.pad >= d.w) continue;
                total += x[((n * d.h + py - d.pad) * d.w + px - d.pad) * d.c + channel] * dy[((n * oh + oy) * ow + ox) * d.c_out + oc];
            }
        }
    }
    dw[t] += total;
}
test "max maxPool picks the first maximum, so a tied window routes its gradient to one element" {
    const d: Dims = .{ .n = 1, .h = 2, .w = 2, .c = 1, .k = 2 };
    const gout = [_]f32{5};
    var out: [1]f32 = undefined;
    var index: [1]u32 = undefined;
    const tied = [_]f32{ 1, 1, 1, 1 };
    var gin = [_]f32{ 0, 0, 0, 0 };
    maxPool(0, &tied, &out, &index, d);
    for (0..4) |i| maxPoolBackward(@intCast(i), &index, &gout, &gin, d);
    try std.testing.expectEqualSlices(f32, &.{ 5, 0, 0, 0 }, &gin);

    const y = [_]f32{ 1, 3, 3, 2 };
    maxPool(0, &y, &out, &index, d);
    try std.testing.expectEqual(@as(f32, 3), out[0]);
    gin = .{ 0, 0, 0, 0 };
    for (0..4) |i| maxPoolBackward(@intCast(i), &index, &gout, &gin, d);
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 0, 0 }, &gin);
}
