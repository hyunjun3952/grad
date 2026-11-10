const cv = @import("conv.zig");

const BM = 128;
const BN = 128;
const BK = 8;
const BM_PAD = BM + 1;
const BN_PAD = BN + 1;
const TM = BM / 16;
const TN = BN / 16;
const threads = 256;

const tiles = struct {
    var a: [BK * BM_PAD]f32 addrspace(.shared) = undefined;
    var b: [BK * BN_PAD]f32 addrspace(.shared) = undefined;
};

const reduction = struct {
    var partial: [threads]f32 addrspace(.shared) = undefined;
};

inline fn barrier() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

inline fn globalIndex() usize {
    return @as(usize, @workGroupId(0)) * @workGroupSize(0) + @workItemId(0);
}

inline fn expf(x: f32) f32 {
    return asm ("ex2.approx.f32 %[out], %[in];"
        : [out] "=f" (-> f32),
        : [in] "f" (x * 1.442695041),
    );
}

inline fn logf(x: f32) f32 {
    const l2 = asm ("lg2.approx.f32 %[out], %[in];"
        : [out] "=f" (-> f32),
        : [in] "f" (x),
    );
    return l2 * 0.6931471806;
}

inline fn gemm(
    comptime ta: bool,
    comptime tb: bool,
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    m: u32,
    n: u32,
    k: u32,
    accumulate: u32,
    chunk: u32,
) void {
    const tid: usize = @workItemId(0);
    const tx = tid % 16;
    const ty = tid / 16;
    const row0: usize = @as(usize, @workGroupId(1)) * BM;
    const col0: usize = @as(usize, @workGroupId(0)) * BN;
    const M: usize = m;
    const N: usize = n;
    const K: usize = k;

    var acc: [TM][TN]f32 = @splat(@splat(0));
    const k_begin = @as(usize, @workGroupId(2)) * chunk;
    const k_end = @min(K, k_begin + chunk);
    var kk = k_begin;
    while (kk < k_end) : (kk += BK) {
        inline for (0..BK * BM / threads) |i| {
            const idx = tid + i * threads;
            const r = if (ta) idx % BM else idx / BK;
            const cc = if (ta) idx / BM else idx % BK;
            const row = row0 + r;
            const col = kk + cc;
            const valid = row < M and col < K;
            tiles.a[cc * BM_PAD + r] = if (!valid) 0 else if (ta) a[col * M + row] else a[row * K + col];
        }
        inline for (0..BK * BN / threads) |i| {
            const idx = tid + i * threads;
            const r = if (tb) idx % BK else idx / BN;
            const cc = if (tb) idx / BK else idx % BN;
            const row = kk + r;
            const col = col0 + cc;
            const valid = row < K and col < N;
            tiles.b[r * BN_PAD + cc] = if (!valid) 0 else if (tb) b[col * K + row] else b[row * N + col];
        }
        barrier();

        inline for (0..BK) |p| {
            var av: [TM]f32 = undefined;
            var bv: [TN]f32 = undefined;
            inline for (0..TM) |i| av[i] = tiles.a[p * BM_PAD + ty + i * 16];
            inline for (0..TN) |j| bv[j] = tiles.b[p * BN_PAD + tx + j * 16];
            inline for (0..TM) |i| {
                inline for (0..TN) |j| acc[i][j] = @mulAdd(f32, av[i], bv[j], acc[i][j]);
            }
        }
        barrier();
    }

    inline for (0..TM) |i| {
        inline for (0..TN) |j| {
            const row = row0 + ty + i * 16;
            const col = col0 + tx + j * 16;
            if (row < M and col < N) {
                const at = row * N + col;
                if (chunk < K) {
                    _ = @atomicRmw(f32, &c[at], .Add, acc[i][j], .monotonic);
                } else {
                    c[at] = if (accumulate != 0) c[at] + acc[i][j] else acc[i][j];
                }
            }
        }
    }
}

export fn sgemmNN(a: [*]const f32, b: [*]const f32, c: [*]f32, m: u32, n: u32, k: u32, accumulate: u32, chunk: u32) callconv(.nvptx_kernel) void {
    gemm(false, false, a, b, c, m, n, k, accumulate, chunk);
}

export fn sgemmNT(a: [*]const f32, b: [*]const f32, c: [*]f32, m: u32, n: u32, k: u32, accumulate: u32, chunk: u32) callconv(.nvptx_kernel) void {
    gemm(false, true, a, b, c, m, n, k, accumulate, chunk);
}

export fn sgemmTN(a: [*]const f32, b: [*]const f32, c: [*]f32, m: u32, n: u32, k: u32, accumulate: u32, chunk: u32) callconv(.nvptx_kernel) void {
    gemm(true, false, a, b, c, m, n, k, accumulate, chunk);
}

export fn elementwise(
    a: [*]const f32,
    b: [*]const f32,
    out: [*]f32,
    n: u32,
    multiply: u32,
    d1: u32,
    d2: u32,
    d3: u32,
    sa0: u32,
    sa1: u32,
    sa2: u32,
    sa3: u32,
    sb0: u32,
    sb1: u32,
    sb2: u32,
    sb3: u32,
) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= n) return;
    var rest = t;
    const p3 = rest % d3;
    rest /= d3;
    const p2 = rest % d2;
    rest /= d2;
    const p1 = rest % d1;
    const p0 = rest / d1;
    const x = a[p0 * sa0 + p1 * sa1 + p2 * sa2 + p3 * sa3];
    const y = b[p0 * sb0 + p1 * sb1 + p2 * sb2 + p3 * sb3];
    out[t] = if (multiply != 0) x * y else x + y;
}

export fn relu(x: [*]const f32, out: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) out[t] = @max(x[t], 0);
}

export fn reluBackward(x: [*]const f32, g_out: [*]const f32, g_in: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) g_in[t] += if (x[t] > 0) g_out[t] else 0;
}

export fn addInPlace(dst: [*]f32, src: [*]const f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) dst[t] += src[t];
}

export fn sumTo(
    in: [*]const f32,
    out: [*]f32,
    out_len: u32,
    reduce_count: u32,
    rmask: u32,
    accumulate: u32,
    d0: u32,
    d1: u32,
    d2: u32,
    d3: u32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
) callconv(.nvptx_kernel) void {
    _ = out_len;
    const t: usize = @workGroupId(0);
    const tid: usize = @workItemId(0);
    const d = [4]usize{ d0, d1, d2, d3 };
    const s = [4]usize{ s0, s1, s2, s3 };

    var base: usize = 0;
    var rest: usize = t;
    inline for (0..4) |step| {
        const axis = 3 - step;
        if ((rmask >> axis) & 1 == 0) {
            base += (rest % d[axis]) * s[axis];
            rest /= d[axis];
        }
    }

    var total: f32 = 0;
    var q: usize = tid;
    while (q < reduce_count) : (q += threads) {
        var left = q;
        var at = base;
        inline for (0..4) |step| {
            const axis = 3 - step;
            if ((rmask >> axis) & 1 != 0) {
                at += (left % d[axis]) * s[axis];
                left /= d[axis];
            }
        }
        total += in[at];
    }
    reduction.partial[tid] = total;
    barrier();
    var half: usize = threads / 2;
    while (half > 0) : (half /= 2) {
        if (tid < half) reduction.partial[tid] += reduction.partial[tid + half];
        barrier();
    }
    if (tid == 0) out[t] = if (accumulate != 0) out[t] + reduction.partial[0] else reduction.partial[0];
}

export fn crossEntropyRows(x: [*]const f32, logp: [*]f32, labels: [*]const u8, row_loss: [*]f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    const r = globalIndex();
    if (r >= rows) return;
    const base = r * cols;
    var mx = x[base];
    var j: usize = 1;
    while (j < cols) : (j += 1) mx = @max(mx, x[base + j]);
    var total: f32 = 0;
    j = 0;
    while (j < cols) : (j += 1) total += expf(x[base + j] - mx);
    const shift = mx + logf(total);
    j = 0;
    while (j < cols) : (j += 1) logp[base + j] = x[base + j] - shift;
    row_loss[r] = -logp[base + labels[r]];
}

export fn crossEntropyMean(row_loss: [*]const f32, out: [*]f32, rows: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() != 0) return;
    var total: f32 = 0;
    var r: usize = 0;
    while (r < rows) : (r += 1) total += row_loss[r];
    out[0] = total / @as(f32, @floatFromInt(rows));
}

export fn crossEntropyBackward(logp: [*]const f32, labels: [*]const u8, grad: [*]f32, loss_grad: [*]const f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, rows) * cols) return;
    const r = t / cols;
    const j = t % cols;
    var p = expf(logp[t]);
    if (j == labels[r]) p -= 1;
    grad[t] += p * (loss_grad[0] / @as(f32, @floatFromInt(rows)));
}

fn convDims(n: u32, h: u32, w: u32, c: u32, k: u32) cv.Dims {
    return .{ .n = n, .h = h, .w = w, .c = c, .k = k };
}

export fn im2col(x: [*]const f32, col: [*]f32, len: u32, n: u32, h: u32, w: u32, c: u32, k: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.im2col(x, col, globalIndex(), convDims(n, h, w, c, k));
}

export fn col2im(dcol: [*]const f32, dx: [*]f32, len: u32, n: u32, h: u32, w: u32, c: u32, k: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.col2im(dcol, dx, globalIndex(), convDims(n, h, w, c, k));
}

export fn maxPool(x: [*]const f32, out: [*]f32, len: u32, n: u32, h: u32, w: u32, c: u32, k: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.maxPool(x, out, globalIndex(), convDims(n, h, w, c, k));
}

export fn maxPoolBackward(x: [*]const f32, gout: [*]const f32, gin: [*]f32, len: u32, n: u32, h: u32, w: u32, c: u32, k: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.maxPoolBackward(x, gout, gin, globalIndex(), convDims(n, h, w, c, k));
}

export fn sgd(w: [*]f32, g: [*]const f32, lr: f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) w[t] -= lr * g[t];
}

export fn adam(
    w: [*]f32,
    g: [*]const f32,
    m: [*]f32,
    v: [*]f32,
    n: u32,
    b1: f32,
    b2: f32,
    step: f32,
    inv_sqrt_c2: f32,
    eps: f32,
) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= n) return;
    const grad = g[t];
    const m_new = b1 * m[t] + (1 - b1) * grad;
    const v_new = b2 * v[t] + (1 - b2) * grad * grad;
    m[t] = m_new;
    v[t] = v_new;
    w[t] -= step * m_new / (@sqrt(v_new) * inv_sqrt_c2 + eps);
}

export fn gather(
    images: [*]const f32,
    labels_src: [*]const u8,
    order: [*]const u32,
    batch: [*]f32,
    labels_out: [*]u8,
    start: u32,
    rows: u32,
    pixels: u32,
) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, rows) * pixels) return;
    const r = t / pixels;
    const p = t % pixels;
    const idx: usize = order[start + r];
    batch[t] = images[idx * pixels + p];
    if (p == 0) labels_out[r] = labels_src[idx];
}
