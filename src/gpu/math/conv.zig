const bf16 = @import("bf16.zig");

pub const Op = enum {
    im2col,
    col2im,
    col2imHalf,
    maxPool,
    maxPoolBackward,
    avgPool,
    avgPoolBackward,
    maxPoolHalf,
    maxPoolBackwardHalf,
    avgPoolHalf,
    avgPoolBackwardHalf,
    groupedForward,
    groupedInputGrad,
    groupedWeightGrad,
};

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
        .maxPool, .avgPool, .maxPoolHalf, .avgPoolHalf => @as(usize, d.n) * outHeight(d) * outWidth(d) * d.c,
        .maxPoolBackward, .maxPoolBackwardHalf => if (d.stride >= d.k) @as(usize, d.n) * outHeight(d) * outWidth(d) * d.c else @as(usize, d.n) * d.h * d.w * d.c,
        .avgPoolBackward, .avgPoolBackwardHalf => @as(usize, d.n) * d.h * d.w * d.c,
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

/// Length of one im2col row: the taps of a k x k window over all input channels.
pub inline fn patchLen(d: Dims) Index {
    return d.k * d.k * d.c;
}

// Position in x that column cc of a patch reads, or null when the tap falls into the padding.
inline fn patchSource(patch: Patch, cc: Index, d: Dims) ?Index {
    const tap = cc / d.c;
    const py = patch.top + tap / d.k * d.dilation;
    const px = patch.left + tap % d.k * d.dilation;
    if (py < d.pad or px < d.pad or py - d.pad >= d.h or px - d.pad >= d.w) return null;
    return ((patch.n * d.h + py - d.pad) * d.w + px - d.pad) * d.c + cc % d.c;
}

pub inline fn patchValue(x: [*]const f32, patch: Patch, cc: Index, d: Dims) f32 {
    const at = patchSource(patch, cc, d) orelse return 0;
    return x[at];
}

pub inline fn patchValues(x: [*]const f32, patch: Patch, cc: Index, d: Dims) @Vector(4, f32) {
    const at = patchSource(patch, cc, d) orelse return @splat(0);
    const source: *align(16) const [4]f32 = @ptrCast(@alignCast(x + at));
    return source.*;
}

pub inline fn im2colValue(x: [*]const f32, row: Index, cc: Index, d: Dims) f32 {
    return patchValue(x, patchOf(row, d), cc, d);
}
pub inline fn im2col(t: Index, x: [*]const f32, col: [*]f32, d: Dims) void {
    const kc = patchLen(d);
    col[t] = im2colValue(x, t / kc, t % kc, d);
}

// An input element split from its flat index.
const Pixel = struct { n: Index, y: Index, x: Index, ch: Index };

inline fn pixelOf(t: Index, d: Dims) Pixel {
    return .{ .ch = t % d.c, .x = t / d.c % d.w, .y = t / d.c / d.w % d.h, .n = t / d.c / d.w / d.h };
}

// Output coordinate whose window starts at pos - shift, or null when no window of this stride lands there. Stride 1 skips the
// division, which is slow on the device.
inline fn outputCoord(pos: Index, shift: Index, d: Dims) ?Index {
    if (pos < shift) return null;
    if (d.stride == 1) return pos - shift;
    if ((pos - shift) % d.stride != 0) return null;
    return (pos - shift) / d.stride;
}

inline fn col2imSum(comptime T: type, t: Index, dcol: [*]const T, d: Dims) f32 {
    const kc = patchLen(d);
    const ow = outWidth(d);
    const oh = outHeight(d);
    const pixel = pixelOf(t, d);
    var total: f32 = 0;
    var ky: Index = 0;
    while (ky < d.k) : (ky += 1) {
        const oy = outputCoord(pixel.y + d.pad, ky * d.dilation, d) orelse continue;
        if (oy >= oh) continue;
        var kx: Index = 0;
        while (kx < d.k) : (kx += 1) {
            const ox = outputCoord(pixel.x + d.pad, kx * d.dilation, d) orelse continue;
            if (ox >= ow) continue;
            const row = (pixel.n * oh + oy) * ow + ox;
            const value = dcol[row * kc + (ky * d.k + kx) * d.c + pixel.ch];
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

// An output element split from its flat index; channels is the channel count of the output (c for pooling, c_out for convolution).
inline fn outputOf(t: Index, channels: Index, d: Dims) Window {
    const ow = outWidth(d);
    const oh = outHeight(d);
    return .{ .ch = t % channels, .ox = t / channels % ow, .oy = t / channels / ow % oh, .n = t / channels / ow / oh };
}

inline fn windowOf(t: Index, d: Dims) Window {
    return outputOf(t, d.c, d);
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

inline fn firstMax(comptime E: type, x: [*]const E, w: Window, d: Dims) Index {
    const b = boundsOf(w, d);
    var best = inputIndex(w.n, b.y0, b.x0, w.ch, d);
    var y = b.y0;
    while (y < b.y1) : (y += 1) {
        var xx = b.x0;
        while (xx < b.x1) : (xx += 1) {
            const at = inputIndex(w.n, y, xx, w.ch, d);
            if (bf16.load(E, x, at) > bf16.load(E, x, best)) best = at;
        }
    }
    return best;
}

inline fn windowCount(w: Window, d: Dims) f32 {
    const b = boundsOf(w, d);
    return @floatFromInt((b.y1 - b.y0) * (b.x1 - b.x0));
}

inline fn avgPoolOf(comptime E: type, t: Index, x: [*]const E, out: [*]E, d: Dims) void {
    const w = windowOf(t, d);
    const b = boundsOf(w, d);
    var total: f32 = 0;
    var y = b.y0;
    while (y < b.y1) : (y += 1) {
        var xx = b.x0;
        while (xx < b.x1) : (xx += 1) total += bf16.load(E, x, inputIndex(w.n, y, xx, w.ch, d));
    }
    bf16.store(E, out, t, total / windowCount(w, d));
}

inline fn maxPoolOf(comptime E: type, t: Index, x: [*]const E, out: [*]E, index: [*]u32, d: Dims) void {
    const winner = firstMax(E, x, windowOf(t, d), d);
    out[t] = x[winner];
    index[t] = winner;
}

inline fn firstWindow(position: Index, d: Dims) Index {
    const shifted = position + d.pad + 1;
    return if (shifted > d.k) (shifted - d.k + d.stride - 1) / d.stride else 0;
}

inline fn pooledGradient(comptime E: type, comptime winner: bool, t: Index, index: [*]const u32, gout: [*]const E, d: Dims) f32 {
    const pixel = pixelOf(t, d);
    const oh = outHeight(d);
    const ow = outWidth(d);
    const last_y = @min((pixel.y + d.pad) / d.stride, oh - 1);
    const last_x = @min((pixel.x + d.pad) / d.stride, ow - 1);
    var total: f32 = 0;
    var oy = firstWindow(pixel.y, d);
    while (oy <= last_y) : (oy += 1) {
        var ox = firstWindow(pixel.x, d);
        while (ox <= last_x) : (ox += 1) {
            const at = ((pixel.n * oh + oy) * ow + ox) * d.c + pixel.ch;
            if (winner) {
                if (index[at] == t) total += bf16.load(E, gout, at);
            } else {
                total += bf16.load(E, gout, at) / windowCount(.{ .n = pixel.n, .oy = oy, .ox = ox, .ch = pixel.ch }, d);
            }
        }
    }
    return total;
}

inline fn avgPoolBackwardOf(comptime E: type, t: Index, gout: [*]const E, gin: [*]E, d: Dims) void {
    bf16.store(E, gin, t, bf16.load(E, gin, t) + pooledGradient(E, false, t, undefined, gout, d));
}

inline fn maxPoolBackwardOf(comptime E: type, t: Index, index: [*]const u32, gout: [*]const E, gin: [*]E, d: Dims) void {
    if (d.stride >= d.k) {
        bf16.store(E, gin, index[t], bf16.load(E, gin, index[t]) + bf16.load(E, gout, t));
        return;
    }
    bf16.store(E, gin, t, bf16.load(E, gin, t) + pooledGradient(E, true, t, index, gout, d));
}

pub inline fn avgPool(t: Index, x: [*]const f32, out: [*]f32, d: Dims) void {
    avgPoolOf(f32, t, x, out, d);
}

pub inline fn avgPoolHalf(t: Index, x: [*]const u16, out: [*]u16, d: Dims) void {
    avgPoolOf(u16, t, x, out, d);
}

pub inline fn maxPool(t: Index, x: [*]const f32, out: [*]f32, index: [*]u32, d: Dims) void {
    maxPoolOf(f32, t, x, out, index, d);
}

pub inline fn maxPoolHalf(t: Index, x: [*]const u16, out: [*]u16, index: [*]u32, d: Dims) void {
    maxPoolOf(u16, t, x, out, index, d);
}

pub inline fn avgPoolBackward(t: Index, gout: [*]const f32, gin: [*]f32, d: Dims) void {
    avgPoolBackwardOf(f32, t, gout, gin, d);
}

pub inline fn avgPoolBackwardHalf(t: Index, gout: [*]const u16, gin: [*]u16, d: Dims) void {
    avgPoolBackwardOf(u16, t, gout, gin, d);
}

pub inline fn maxPoolBackward(t: Index, index: [*]const u32, gout: [*]const f32, gin: [*]f32, d: Dims) void {
    maxPoolBackwardOf(f32, t, index, gout, gin, d);
}

pub inline fn maxPoolBackwardHalf(t: Index, index: [*]const u32, gout: [*]const u16, gin: [*]u16, d: Dims) void {
    maxPoolBackwardOf(u16, t, index, gout, gin, d);
}
pub inline fn groupedForward(t: Index, x: [*]const f32, w: [*]const f32, out: [*]f32, d: Dims) void {
    const cg = d.c / d.groups;
    const pos = outputOf(t, d.c_out, d);
    const base = pos.ch / (d.c_out / d.groups) * cg;
    var total: f32 = 0;
    var ky: Index = 0;
    while (ky < d.k) : (ky += 1) {
        const py = pos.oy * d.stride + ky * d.dilation;
        if (py < d.pad or py - d.pad >= d.h) continue;
        var kx: Index = 0;
        while (kx < d.k) : (kx += 1) {
            const px = pos.ox * d.stride + kx * d.dilation;
            if (px < d.pad or px - d.pad >= d.w) continue;
            const source = ((pos.n * d.h + py - d.pad) * d.w + px - d.pad) * d.c + base;
            const weights = (ky * d.k + kx) * cg;
            var ic: Index = 0;
            while (ic < cg) : (ic += 1) total += x[source + ic] * w[(weights + ic) * d.c_out + pos.ch];
        }
    }
    out[t] = total;
}

pub inline fn groupedInputGrad(t: Index, dy: [*]const f32, w: [*]const f32, dx: [*]f32, d: Dims) void {
    const ow = outWidth(d);
    const oh = outHeight(d);
    const cg = d.c / d.groups;
    const per_group = d.c_out / d.groups;
    const pixel = pixelOf(t, d);
    const first_out = pixel.ch / cg * per_group;
    var total: f32 = 0;
    var ky: Index = 0;
    while (ky < d.k) : (ky += 1) {
        const oy = outputCoord(pixel.y + d.pad, ky * d.dilation, d) orelse continue;
        if (oy >= oh) continue;
        var kx: Index = 0;
        while (kx < d.k) : (kx += 1) {
            const ox = outputCoord(pixel.x + d.pad, kx * d.dilation, d) orelse continue;
            if (ox >= ow) continue;
            const grad = ((pixel.n * oh + oy) * ow + ox) * d.c_out + first_out;
            const weights = ((ky * d.k + kx) * cg + pixel.ch % cg) * d.c_out + first_out;
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
