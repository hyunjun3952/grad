const std = @import("std");

const cv = @import("../gpu/math/conv.zig");
const kernels = @import("kernels.zig");

const Job = struct { src: []const f32, dst: []f32, d: cv.Dims, first: usize = 0 };

pub fn rowsOf(d: cv.Dims) usize {
    return @as(usize, d.n) * cv.outHeight(d) * cv.outWidth(d);
}

fn im2colRows(job: Job, lo: usize, hi: usize) void {
    const d = job.d;
    const c: usize = d.c;
    const k: usize = d.k;
    const kc = k * k * c;
    const w: i64 = d.w;
    const pad: i64 = d.pad;
    for (lo..hi) |index| {
        const patch = cv.patchOf(@intCast(job.first + index), d);
        const out = job.dst[index * kc ..][0..kc];
        const first: usize = @intCast(std.math.clamp(pad - @as(i64, patch.left), 0, @as(i64, @intCast(k))));
        const last: usize = @intCast(std.math.clamp(w + pad - @as(i64, patch.left), 0, @as(i64, @intCast(k))));
        for (0..k) |ky| {
            const line = out[ky * k * c ..][0 .. k * c];
            const y = @as(i64, patch.top) + @as(i64, @intCast(ky)) - pad;
            if (y < 0 or y >= d.h or last <= first) {
                @memset(line, 0);
                continue;
            }
            @memset(line[0 .. first * c], 0);
            const from = ((@as(usize, patch.n) * d.h + @as(usize, @intCast(y))) * d.w + patch.left + first - d.pad) * c;
            @memcpy(line[first * c .. last * c], job.src[from..][0 .. (last - first) * c]);
            @memset(line[last * c ..], 0);
        }
    }
}

pub fn im2col(par: kernels.Parallel, x: []const f32, col: []f32, d: cv.Dims) void {
    im2colRange(par, x, col, d, 0, rowsOf(d));
}

/// Writes rows first..first+count of the im2col matrix to the start of col.
pub fn im2colRange(par: kernels.Parallel, x: []const f32, col: []f32, d: cv.Dims, first: usize, count: usize) void {
    kernels.parallelFor(par, count, 32, Job{ .src = x, .dst = col, .d = d, .first = first }, im2colRows);
}

// Rows of the im2col matrix handled at once by the chunked convolution; the chunk buffer stays in cache.
const chunk_bytes = 1 << 21;

fn chunkRows(d: cv.Dims) usize {
    const kc = @as(usize, d.k) * d.k * d.c;
    return @min(rowsOf(d), @max(1024, chunk_bytes / (kc * @sizeOf(f32))));
}

// The matmul work buffer must fit both the full chunks and the shorter last one.
fn chunkWorkLen(comptime cfg: kernels.TileConfig, par: kernels.Parallel, d: cv.Dims, chunk: usize) usize {
    const kc = @as(usize, d.k) * d.k * d.c;
    const tail = rowsOf(d) % chunk;
    const full = if (cfg.trans_a)
        kernels.matmulTiledWorkLen(f32, cfg, par, kc, chunk, d.c_out)
    else
        kernels.matmulTiledWorkLen(f32, cfg, par, chunk, kc, d.c_out);
    if (tail == 0) return full;
    const last = if (cfg.trans_a)
        kernels.matmulTiledWorkLen(f32, cfg, par, kc, tail, d.c_out)
    else
        kernels.matmulTiledWorkLen(f32, cfg, par, tail, kc, d.c_out);
    return @max(full, last);
}

/// Computes out = im2col(x) * w for a convolution without materializing the whole im2col matrix: it is built and multiplied
/// in chunks of rows. w is (k*k*c) x c_out, out is positions x c_out.
pub fn convForward(gpa: std.mem.Allocator, par: kernels.Parallel, x: []const f32, w: []const f32, out: []f32, d: cv.Dims) !void {
    const kc = @as(usize, d.k) * d.k * d.c;
    const rows = rowsOf(d);
    const chunk = chunkRows(d);
    const cfg: kernels.TileConfig = .{};
    const col = try gpa.alloc(f32, chunk * kc);
    defer gpa.free(col);
    const work = try gpa.alloc(f32, chunkWorkLen(cfg, par, d, chunk));
    defer gpa.free(work);
    var first: usize = 0;
    while (first < rows) : (first += chunk) {
        const count = @min(chunk, rows - first);
        im2colRange(par, x, col, d, first, count);
        kernels.matmulTiled(f32, cfg, par, .{ .m = count, .k = kc, .n = d.c_out, .a = col[0 .. count * kc], .b = w, .out = out[first * d.c_out ..][0 .. count * d.c_out] }, work);
    }
}

/// Adds the weight gradient im2col(x)^T * grad to dw, rebuilding the im2col matrix chunk by chunk. grad is positions x c_out.
pub fn convWeightGrad(gpa: std.mem.Allocator, par: kernels.Parallel, x: []const f32, grad: []const f32, dw: []f32, d: cv.Dims) !void {
    const kc = @as(usize, d.k) * d.k * d.c;
    const rows = rowsOf(d);
    const chunk = chunkRows(d);
    const cfg: kernels.TileConfig = .{ .trans_a = true, .accumulate = true };
    const col = try gpa.alloc(f32, chunk * kc);
    defer gpa.free(col);
    const work = try gpa.alloc(f32, chunkWorkLen(cfg, par, d, chunk));
    defer gpa.free(work);
    var first: usize = 0;
    while (first < rows) : (first += chunk) {
        const count = @min(chunk, rows - first);
        im2colRange(par, x, col, d, first, count);
        kernels.matmulTiled(f32, cfg, par, .{ .m = kc, .k = count, .n = d.c_out, .a = col[0 .. count * kc], .b = grad[first * d.c_out ..][0 .. count * d.c_out], .out = dw }, work);
    }
}

fn col2imPixels(job: Job, lo: usize, hi: usize) void {
    const d = job.d;
    const c: usize = d.c;
    const kc = @as(usize, d.k) * d.k * c;
    const oh = cv.outHeight(d);
    const ow = cv.outWidth(d);
    for (lo..hi) |pixel| {
        const y: u32 = @intCast(pixel / d.w % d.h + d.pad);
        const x: u32 = @intCast(pixel % d.w + d.pad);
        const n: u32 = @intCast(pixel / d.w / d.h);
        const out = job.dst[pixel * c ..][0..c];
        var ky: u32 = 0;
        while (ky < d.k) : (ky += 1) {
            if (y < ky or (y - ky) % d.stride != 0) continue;
            const oy = (y - ky) / d.stride;
            if (oy >= oh) continue;
            var kx: u32 = 0;
            while (kx < d.k) : (kx += 1) {
                if (x < kx or (x - kx) % d.stride != 0) continue;
                const ox = (x - kx) / d.stride;
                if (ox >= ow) continue;
                const source = job.src[((@as(usize, n) * oh + oy) * ow + ox) * kc + (@as(usize, ky) * d.k + kx) * c ..][0..c];
                for (out, source) |*o, s| o.* += s;
            }
        }
    }
}

pub fn col2im(par: kernels.Parallel, dcol: []const f32, dx: []f32, d: cv.Dims) void {
    kernels.parallelFor(par, @as(usize, d.n) * d.h * d.w, 64, Job{ .src = dcol, .dst = dx, .d = d }, col2imPixels);
}

const norm = @import("../gpu/math/norm.zig");

const ApplyJob = struct { x: [*]const f32, y: [*]f32, p: norm.Affine, l: norm.Layout };

fn normApplyRows(job: ApplyJob, lo: usize, hi: usize) void {
    const c: usize = job.l.channels;
    const relu = job.l.flag != 0;
    for (lo..hi) |row| {
        const x = job.x[row * c ..][0..c];
        const y = job.y[row * c ..][0..c];
        for (0..c) |ch| {
            const v = job.p.gamma[ch] * (x[ch] - job.p.mean[ch]) * job.p.inv_std[ch] + job.p.beta[ch];
            y[ch] = if (relu) @max(v, 0) else v;
        }
    }
}

pub fn normApply(par: kernels.Parallel, x: []const f32, y: []f32, p: norm.Affine, l: norm.Layout) void {
    kernels.parallelFor(par, l.len / l.channels, 256, ApplyJob{ .x = x.ptr, .y = y.ptr, .p = p, .l = l }, normApplyRows);
}

const GradJob = struct { io: norm.GradIo(f32), p: norm.Affine, l: norm.Layout };

fn normGradRows(job: GradJob, lo: usize, hi: usize) void {
    const c: usize = job.l.channels;
    const n: f32 = @floatFromInt(job.l.len / job.l.channels);
    const masked = job.l.flag != 0;
    for (lo..hi) |row| {
        const x = job.io.x[row * c ..][0..c];
        const dy = job.io.dy[row * c ..][0..c];
        const mask = job.io.mask[row * c ..][0..c];
        const dx = job.io.dx[row * c ..][0..c];
        for (0..c) |ch| {
            const x_hat = (x[ch] - job.p.mean[ch]) * job.p.inv_std[ch];
            const g = if (masked and mask[ch] <= 0) 0 else dy[ch];
            dx[ch] += job.p.gamma[ch] * job.p.inv_std[ch] / n * (n * g - job.io.sums[ch] - x_hat * job.io.sums[c + ch]);
        }
    }
}

pub fn normGradInput(par: kernels.Parallel, io: norm.GradIo(f32), p: norm.Affine, l: norm.Layout) void {
    kernels.parallelFor(par, l.len / l.channels, 256, GradJob{ .io = io, .p = p, .l = l }, normGradRows);
}
const parts = 8;
const Wide = @Vector(4, f64);

const Pass = struct {
    x: [*]const f32,
    other: [*]const f32,
    mask: [*]const f32,
    centre: [*]const f64,
    inv_std: [*]const f32,
    rows: usize,
    channels: usize,
    partial: *[parts][2 * norm.max_channels]f64,
    masked: bool = false,
    kind: enum { sum, square, gradients },
};

fn widen(values: [*]const f32) Wide {
    const narrow: @Vector(4, f32) = values[0..4].*;
    return narrow;
}

fn accumulate(job: Pass, lo: usize, hi: usize) void {
    const c = job.channels;
    for (lo..hi) |part| {
        const first = part * job.rows / parts;
        const last = (part + 1) * job.rows / parts;
        const first_out = job.partial[part][0..c];
        const second_out = job.partial[part][norm.max_channels..][0..c];
        @memset(first_out, 0);
        @memset(second_out, 0);
        for (first..last) |row| {
            const x = job.x + row * c;
            const g = job.other + row * c;
            const m = job.mask + row * c;
            var ch: usize = 0;
            while (ch + 4 <= c) : (ch += 4) {
                const centre: Wide = job.centre[ch..][0..4].*;
                const v = widen(x + ch);
                switch (job.kind) {
                    .sum => first_out[ch..][0..4].* = @as(Wide, first_out[ch..][0..4].*) + v,
                    .square => second_out[ch..][0..4].* = @as(Wide, second_out[ch..][0..4].*) + (v - centre) * (v - centre),
                    .gradients => {
                        const kept = @as(@Vector(4, f32), m[ch..][0..4].*) > @as(@Vector(4, f32), @splat(0));
                        const gradient: Wide = if (job.masked) @select(f64, kept, widen(g + ch), @as(Wide, @splat(0))) else widen(g + ch);
                        const inverse: Wide = widen(job.inv_std + ch);
                        first_out[ch..][0..4].* = @as(Wide, first_out[ch..][0..4].*) + gradient;
                        second_out[ch..][0..4].* = @as(Wide, second_out[ch..][0..4].*) + gradient * ((v - centre) * inverse);
                    },
                }
            }
            while (ch < c) : (ch += 1) {
                const v: f64 = x[ch];
                switch (job.kind) {
                    .sum => first_out[ch] += v,
                    .square => second_out[ch] += (v - job.centre[ch]) * (v - job.centre[ch]),
                    .gradients => {
                        const gradient: f64 = if (job.masked and m[ch] <= 0) 0 else g[ch];
                        first_out[ch] += gradient;
                        second_out[ch] += gradient * ((v - job.centre[ch]) * job.inv_std[ch]);
                    },
                }
            }
        }
    }
}

fn reduceParts(par: kernels.Parallel, job: Pass, out: []f64, which: usize) void {
    kernels.parallelFor(par, parts, 1, job, accumulate);
    for (out[0..job.channels], 0..) |*total, ch| {
        total.* = 0;
        for (job.partial) |p| total.* += p[which * norm.max_channels + ch];
    }
}

pub fn normStats(par: kernels.Parallel, x: []const f32, st: norm.Stats, c: usize, training: bool) void {
    std.debug.assert(c <= norm.max_channels);
    if (!training) {
        for (0..c) |ch| {
            st.mean[ch] = st.running_mean[ch];
            st.inv_std[ch] = 1 / @sqrt(st.running_var[ch] + norm.eps);
        }
        return;
    }
    var partial: [parts][2 * norm.max_channels]f64 = undefined;
    var job: Pass = .{ .x = x.ptr, .other = x.ptr, .mask = x.ptr, .centre = undefined, .inv_std = st.inv_std, .rows = x.len / c, .channels = c, .partial = &partial, .kind = .sum };
    var mean: [norm.max_channels]f64 = undefined;
    var square: [norm.max_channels]f64 = undefined;
    reduceParts(par, job, &mean, 0);
    const count: f64 = @floatFromInt(job.rows);
    for (mean[0..c]) |*m| m.* /= count;
    job.centre = &mean;
    job.kind = .square;
    reduceParts(par, job, &square, 1);
    for (0..c) |ch| {
        const mu: f32 = @floatCast(mean[ch]);
        const variance: f32 = @floatCast(square[ch] / count);
        st.mean[ch] = mu;
        st.inv_std[ch] = 1 / @sqrt(variance + norm.eps);
        st.running_mean[ch] += norm.momentum * (mu - st.running_mean[ch]);
        st.running_var[ch] += norm.momentum * (variance - st.running_var[ch]);
    }
}

pub fn normGradSums(par: kernels.Parallel, x: []const f32, dy: []const f32, p: norm.Affine, mask: []const f32, out: norm.GradOut) void {
    const c = out.beta.len;
    std.debug.assert(c <= norm.max_channels);
    var partial: [parts][2 * norm.max_channels]f64 = undefined;
    var centre: [norm.max_channels]f64 = undefined;
    for (centre[0..c], p.mean) |*m, v| m.* = v;
    const job: Pass = .{ .x = x.ptr, .other = dy.ptr, .mask = if (mask.len != 0) mask.ptr else dy.ptr, .centre = &centre, .inv_std = p.inv_std, .rows = x.len / c, .channels = c, .partial = &partial, .masked = mask.len != 0, .kind = .gradients };
    var beta: [norm.max_channels]f64 = undefined;
    var gamma: [norm.max_channels]f64 = undefined;
    kernels.parallelFor(par, parts, 1, job, accumulate);
    for (0..c) |ch| {
        beta[ch] = 0;
        gamma[ch] = 0;
        for (partial) |q| {
            beta[ch] += q[ch];
            gamma[ch] += q[norm.max_channels + ch];
        }
        out.sums[ch] = @floatCast(beta[ch]);
        out.sums[c + ch] = @floatCast(gamma[ch]);
        out.beta[ch] += out.sums[ch];
        out.gamma[ch] += out.sums[c + ch];
    }
}
