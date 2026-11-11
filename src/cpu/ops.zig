const std = @import("std");

const cv = @import("../gpu/math/conv.zig");
const kernels = @import("kernels.zig");

const Job = struct { src: []const f32, dst: []f32, d: cv.Dims };

fn rowsOf(d: cv.Dims) usize {
    return @as(usize, d.n) * cv.outHeight(d) * cv.outWidth(d);
}

fn im2colRows(job: Job, lo: usize, hi: usize) void {
    const d = job.d;
    const c: usize = d.c;
    const k: usize = d.k;
    const kc = k * k * c;
    const w: i64 = d.w;
    const pad: i64 = d.pad;
    for (lo..hi) |row| {
        const patch = cv.patchOf(@intCast(row), d);
        const out = job.dst[row * kc ..][0..kc];
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
    kernels.parallelFor(par, rowsOf(d), 32, Job{ .src = x, .dst = col, .d = d }, im2colRows);
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

const GradJob = struct { io: norm.GradIo, p: norm.Affine, l: norm.Layout };

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

pub fn normGradInput(par: kernels.Parallel, io: norm.GradIo, p: norm.Affine, l: norm.Layout) void {
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
const testing = std.testing;

test "row-wise im2col and col2im equal the per-element reference for strides, padding and channel counts" {
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    const gpa = testing.allocator;
    const par: kernels.Parallel = .{ .io = testing.io, .threads = 3 };
    for ([_]cv.Dims{
        .{ .n = 2, .h = 6, .w = 7, .c = 1, .k = 3, .stride = 1, .pad = 0 },
        .{ .n = 2, .h = 6, .w = 7, .c = 3, .k = 3, .stride = 1, .pad = 1 },
        .{ .n = 1, .h = 9, .w = 8, .c = 4, .k = 5, .stride = 2, .pad = 2 },
        .{ .n = 3, .h = 5, .w = 5, .c = 16, .k = 1, .stride = 1, .pad = 0 },
        .{ .n = 2, .h = 8, .w = 8, .c = 2, .k = 3, .stride = 2, .pad = 1 },
    }) |d| {
        const x = try gpa.alloc(f32, @as(usize, d.n) * d.h * d.w * d.c);
        defer gpa.free(x);
        for (x) |*v| v.* = rng.floatNorm(f32);
        const columns = rowsOf(d) * d.k * d.k * d.c;
        const want = try gpa.alloc(f32, columns);
        defer gpa.free(want);
        const got = try gpa.alloc(f32, columns);
        defer gpa.free(got);

        for (0..columns) |t| cv.im2col(@intCast(t), x.ptr, want.ptr, d);
        @memset(got, 99);
        im2col(par, x, got, d);
        try testing.expectEqualSlices(f32, want, got);

        const dcol = want;
        for (dcol) |*v| v.* = rng.floatNorm(f32);
        const want_x = try gpa.alloc(f32, x.len);
        defer gpa.free(want_x);
        const got_x = try gpa.alloc(f32, x.len);
        defer gpa.free(got_x);
        @memset(want_x, 0);
        @memset(got_x, 0);
        for (0..x.len) |t| cv.col2im(@intCast(t), dcol.ptr, want_x.ptr, d);
        col2im(par, dcol, got_x, d);
        for (want_x, got_x) |a, b| try testing.expectApproxEqAbs(a, b, 1e-6);
    }
}

test "row-wise batch-norm apply and input gradient equal the per-element reference" {
    var prng = std.Random.DefaultPrng.init(8);
    const rng = prng.random();
    const gpa = testing.allocator;
    const par: kernels.Parallel = .{ .io = testing.io, .threads = 3 };
    const c = 5;
    const rows = 301;
    const len = rows * c;
    const x = try gpa.alloc(f32, len);
    defer gpa.free(x);
    const dy = try gpa.alloc(f32, len);
    defer gpa.free(dy);
    const mask = try gpa.alloc(f32, len);
    defer gpa.free(mask);
    for (x, dy, mask) |*a, *b, *m| {
        a.* = rng.floatNorm(f32);
        b.* = rng.floatNorm(f32);
        m.* = rng.floatNorm(f32);
    }
    var channel: [5][c]f32 = undefined;
    for (&channel) |*row| {
        for (row) |*v| v.* = rng.float(f32) + 0.5;
    }
    var sums: [2 * c]f32 = undefined;
    for (&sums) |*v| v.* = rng.floatNorm(f32);
    const p: norm.Affine = .{ .mean = &channel[0], .inv_std = &channel[1], .gamma = &channel[2], .beta = &channel[3] };

    for ([_]u32{ 0, 1 }) |flag| {
        const l: norm.Layout = .{ .len = len, .channels = c, .flag = flag };
        const want = try gpa.alloc(f32, len);
        defer gpa.free(want);
        const got = try gpa.alloc(f32, len);
        defer gpa.free(got);
        for (0..len) |t| norm.apply(t, x.ptr, want.ptr, p, l);
        normApply(par, x, got, p, l);
        try testing.expectEqualSlices(f32, want, got);

        @memset(want, 0.25);
        @memset(got, 0.25);
        for (0..len) |t| norm.gradInput(t, .{ .x = x.ptr, .dy = dy.ptr, .dx = want.ptr, .sums = &sums, .mask = mask.ptr }, p, l);
        normGradInput(par, .{ .x = x.ptr, .dy = dy.ptr, .dx = got.ptr, .sums = &sums, .mask = mask.ptr }, p, l);
        for (want, got) |a, b| try testing.expectApproxEqAbs(a, b, 1e-6);
    }
}
