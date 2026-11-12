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
        if (d.dilation != 1) {
            // NOTE: the memcpy fast path copies whole tap rows, which only holds for contiguous (dilated) taps.
            for (0..kc) |cc| out[cc] = cv.im2colValue(job.src.ptr, @intCast(job.first + index), @intCast(cc), d);
            continue;
        }
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

const DirectJob = struct { x: []const f32, w: []const f32, out: []f32, d: cv.Dims };

const direct_lanes = std.simd.suggestVectorLength(f32) orelse 1;

// NOTE: with a single input channel the im2col matrix is k*k times the input, and building it costs more than the product
// (0.26 ms against 0.08 ms for the MNIST 1->16 layer). Padding and dilation would need bounds checks per tap, so they stay on
// the GEMM path.
fn convIsDirect(d: cv.Dims) bool {
    return d.c == 1 and d.groups == 1 and d.pad == 0 and d.dilation == 1 and d.c_out % direct_lanes == 0;
}

// Output positions accumulated together: one position is a chain of k*k dependent multiply-adds, so a single chain waits on
// the FMA latency; independent positions fill the pipeline.
const direct_block = 8;

fn convDirectBlock(comptime block: usize, job: DirectJob, first: usize) void {
    const Vec = @Vector(direct_lanes, f32);
    const d = job.d;
    const k: usize = d.k;
    const c_out: usize = d.c_out;
    const out_w: usize = cv.outWidth(d);
    const out_h: usize = cv.outHeight(d);
    var origin: [block]usize = undefined;
    inline for (0..block) |b| {
        const row = first + b;
        origin[b] = ((row / (out_h * out_w)) * d.h + row / out_w % out_h * d.stride) * d.w + row % out_w * d.stride;
    }
    var v: usize = 0;
    while (v < c_out) : (v += direct_lanes) {
        var acc: [block]Vec = @splat(@splat(0));
        for (0..k) |ky| {
            for (0..k) |kx| {
                const weights: Vec = job.w[(ky * k + kx) * c_out + v ..][0..direct_lanes].*;
                inline for (0..block) |b| {
                    const tap: Vec = @splat(job.x[origin[b] + ky * d.w + kx]);
                    acc[b] = @mulAdd(Vec, tap, weights, acc[b]);
                }
            }
        }
        inline for (0..block) |b| job.out[(first + b) * c_out + v ..][0..direct_lanes].* = acc[b];
    }
}

fn convDirectRows(job: DirectJob, lo: usize, hi: usize) void {
    var row = lo;
    while (row + direct_block <= hi) : (row += direct_block) convDirectBlock(direct_block, job, row);
    while (row < hi) : (row += 1) convDirectBlock(1, job, row);
}

/// Computes out = im2col(x) * w for a convolution without materializing the whole im2col matrix: it is built and multiplied
/// in chunks of rows. w is (k*k*c) x c_out, out is positions x c_out. A single-channel convolution without padding skips the
/// matrix and accumulates the taps directly.
pub fn convForward(gpa: std.mem.Allocator, par: kernels.Parallel, x: []const f32, w: []const f32, out: []f32, d: cv.Dims) !void {
    if (convIsDirect(d)) return kernels.parallelFor(par, rowsOf(d), 64, DirectJob{ .x = x, .w = w, .out = out, .d = d }, convDirectRows);
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

/// Adds the input gradient of a convolution to dx: grad * w^T is formed and scattered back (col2im) for a few images at a
/// time, which is exact because images do not overlap. w is (k*k*c) x c_out, grad is positions x c_out.
pub fn convInputGrad(gpa: std.mem.Allocator, par: kernels.Parallel, grad: []const f32, w: []const f32, dx: []f32, d: cv.Dims) !void {
    const kc = @as(usize, d.k) * d.k * d.c;
    const per_image = cv.outHeight(d) * cv.outWidth(d);
    const images = @min(@as(usize, d.n), @max(1, chunk_bytes / (per_image * kc * @sizeOf(f32))));
    const cfg: kernels.TileConfig = .{ .trans_b = true };
    const dcol = try gpa.alloc(f32, images * per_image * kc);
    defer gpa.free(dcol);
    const work = try gpa.alloc(f32, kernels.matmulTiledWorkLen(f32, cfg, par, images * per_image, d.c_out, kc));
    defer gpa.free(work);
    var first: usize = 0;
    while (first < d.n) : (first += images) {
        const count = @min(images, d.n - first);
        const rows = count * per_image;
        kernels.matmulTiled(f32, cfg, par, .{ .m = rows, .k = d.c_out, .n = kc, .a = grad[first * per_image * d.c_out ..][0 .. rows * d.c_out], .b = w, .out = dcol[0 .. rows * kc] }, work);
        var part = d;
        part.n = @intCast(count);
        const pixels = count * @as(usize, d.h) * d.w * d.c;
        col2im(par, dcol[0 .. rows * kc], dx[first * @as(usize, d.h) * d.w * d.c ..][0..pixels], part);
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
            const shift_y = ky * d.dilation;
            if (y < shift_y or (y - shift_y) % d.stride != 0) continue;
            const oy = (y - shift_y) / d.stride;
            if (oy >= oh) continue;
            var kx: u32 = 0;
            while (kx < d.k) : (kx += 1) {
                const shift_x = kx * d.dilation;
                if (x < shift_x or (x - shift_x) % d.stride != 0) continue;
                const ox = (x - shift_x) / d.stride;
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

const PoolJob = struct { x: []const f32, out: []f32, index: []u32, d: cv.Dims };

fn maxPoolRows(job: PoolJob, lo: usize, hi: usize) void {
    const d = job.d;
    const c: usize = d.c;
    const oh: usize = cv.outHeight(d);
    const ow: usize = cv.outWidth(d);
    for (lo..hi) |row| {
        const n = row / oh;
        const top = row % oh * d.stride;
        const y0 = top -| d.pad;
        const y1 = @min(top + d.k - d.pad, d.h);
        for (0..ow) |ox| {
            const left = ox * d.stride;
            const x0 = left -| d.pad;
            const x1 = @min(left + d.k - d.pad, d.w);
            const at = (row * ow + ox) * c;
            const best = job.out[at..][0..c];
            const winner = job.index[at..][0..c];
            // NOTE: the window is walked row by row with every channel inside, so the inner loop is contiguous; a value
            // replaces the best only when strictly larger, which keeps the first maximum as the device kernel does.
            const first = ((n * d.h + y0) * d.w + x0) * c;
            @memcpy(best, job.x[first..][0..c]);
            for (winner, 0..) |*w, ch| w.* = @intCast(first + ch);
            for (y0..y1) |y| {
                for (x0..x1) |x| {
                    const from = ((n * d.h + y) * d.w + x) * c;
                    for (best, winner, job.x[from..][0..c], 0..) |*b, *w, v, ch| {
                        if (v > b.*) {
                            b.* = v;
                            w.* = @intCast(from + ch);
                        }
                    }
                }
            }
        }
    }
}

/// Max pooling of an NHWC tensor that also stores the flat input index of each winner, one output row per work item.
pub fn maxPool(par: kernels.Parallel, x: []const f32, out: []f32, index: []u32, d: cv.Dims) void {
    const per_row = @as(usize, cv.outWidth(d)) * d.k * d.k * d.c;
    kernels.parallelFor(par, @as(usize, d.n) * cv.outHeight(d), @max(1, 8192 / per_row), PoolJob{ .x = x, .out = out, .index = index, .d = d }, maxPoolRows);
}

const norm = @import("../gpu/math/norm.zig");

const ApplyJob = struct { x: [*]const f32, residual: [*]const f32, y: [*]f32, p: norm.Affine, l: norm.Layout };

fn normApplyRows(job: ApplyJob, lo: usize, hi: usize) void {
    const c: usize = job.l.channels;
    const relu = job.l.flag & norm.relu_flag != 0;
    const add = job.l.flag & norm.residual_flag != 0;
    for (lo..hi) |row| {
        const x = job.x[row * c ..][0..c];
        const r = job.residual[row * c ..][0..c];
        const y = job.y[row * c ..][0..c];
        for (0..c) |ch| {
            var v = job.p.gamma[ch] * (x[ch] - job.p.mean[ch]) * job.p.inv_std[ch] + job.p.beta[ch];
            if (add) v += r[ch];
            y[ch] = if (relu) @max(v, 0) else v;
        }
    }
}

/// Applies the batch-norm affine map row by row on the worker pool; residual is added before the ReLU when l.flag has residual_flag (pass x otherwise).
pub fn normApply(par: kernels.Parallel, x: []const f32, residual: []const f32, y: []f32, p: norm.Affine, l: norm.Layout) void {
    kernels.parallelFor(par, l.len / l.channels, 256, ApplyJob{ .x = x.ptr, .residual = residual.ptr, .y = y.ptr, .p = p, .l = l }, normApplyRows);
}

const GradJob = struct { io: norm.GradIo(f32), p: norm.Affine, l: norm.Layout };

fn normGradRows(job: GradJob, lo: usize, hi: usize) void {
    const c: usize = job.l.channels;
    const n: f32 = @floatFromInt(job.l.len / job.l.channels);
    const masked = job.l.flag & norm.relu_flag != 0;
    const add = job.l.flag & norm.residual_flag != 0;
    for (lo..hi) |row| {
        const x = job.io.x[row * c ..][0..c];
        const dy = job.io.dy[row * c ..][0..c];
        const mask = job.io.mask[row * c ..][0..c];
        const dx = job.io.dx[row * c ..][0..c];
        const dr = job.io.dr[row * c ..][0..c];
        for (0..c) |ch| {
            const x_hat = (x[ch] - job.p.mean[ch]) * job.p.inv_std[ch];
            const g = if (masked and mask[ch] <= 0) 0 else dy[ch];
            if (add) dr[ch] += g;
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
    const job: Pass = .{
        .x = x.ptr,
        .other = dy.ptr,
        .mask = if (mask.len != 0) mask.ptr else dy.ptr,
        .centre = &centre,
        .inv_std = p.inv_std,
        .rows = x.len / c,
        .channels = c,
        .partial = &partial,
        .masked = mask.len != 0,
        .kind = .gradients,
    };
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
