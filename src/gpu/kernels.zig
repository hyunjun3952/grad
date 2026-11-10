const BM = 64;
const BN = 64;
const BK = 16;
const BM_PAD = BM + 1;
const BN_PAD = BN + 1;
const TM = 4;
const TN = 4;
const threads = 256;

const tiles = struct {
    var a: [BK * BM_PAD]f32 addrspace(.shared) = undefined;
    var b: [BK * BN_PAD]f32 addrspace(.shared) = undefined;
};

inline fn barrier() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

inline fn gid() usize {
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
    var kk: usize = 0;
    while (kk < K) : (kk += BK) {
        inline for (0..4) |i| {
            const idx = tid + i * threads;
            const r = if (ta) idx % BM else idx / BK;
            const cc = if (ta) idx / BM else idx % BK;
            const row = row0 + r;
            const col = kk + cc;
            const valid = row < M and col < K;
            tiles.a[cc * BM_PAD + r] = if (!valid) 0 else if (ta) a[col * M + row] else a[row * K + col];
        }
        inline for (0..4) |i| {
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
                c[at] = if (accumulate != 0) c[at] + acc[i][j] else acc[i][j];
            }
        }
    }
}

export fn sgemm_nn(a: [*]const f32, b: [*]const f32, c: [*]f32, m: u32, n: u32, k: u32, accumulate: u32) callconv(.nvptx_kernel) void {
    gemm(false, false, a, b, c, m, n, k, accumulate);
}

export fn sgemm_nt(a: [*]const f32, b: [*]const f32, c: [*]f32, m: u32, n: u32, k: u32, accumulate: u32) callconv(.nvptx_kernel) void {
    gemm(false, true, a, b, c, m, n, k, accumulate);
}

export fn sgemm_tn(a: [*]const f32, b: [*]const f32, c: [*]f32, m: u32, n: u32, k: u32, accumulate: u32) callconv(.nvptx_kernel) void {
    gemm(true, false, a, b, c, m, n, k, accumulate);
}

inline fn binary(
    comptime mul: bool,
    a: [*]const f32,
    b: [*]const f32,
    out: [*]f32,
    n: u32,
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
) void {
    const t = gid();
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
    out[t] = if (mul) x * y else x + y;
}

export fn bin_add(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32, d1: u32, d2: u32, d3: u32, sa0: u32, sa1: u32, sa2: u32, sa3: u32, sb0: u32, sb1: u32, sb2: u32, sb3: u32) callconv(.nvptx_kernel) void {
    binary(false, a, b, out, n, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);
}

export fn bin_mul(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32, d1: u32, d2: u32, d3: u32, sa0: u32, sa1: u32, sa2: u32, sa3: u32, sb0: u32, sb1: u32, sb2: u32, sb3: u32) callconv(.nvptx_kernel) void {
    binary(true, a, b, out, n, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);
}

export fn relu(x: [*]const f32, out: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = gid();
    if (t < n) out[t] = @max(x[t], 0);
}

export fn relu_bwd(x: [*]const f32, g_out: [*]const f32, g_in: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = gid();
    if (t < n) g_in[t] += if (x[t] > 0) g_out[t] else 0;
}

export fn add_inplace(dst: [*]f32, src: [*]const f32, n: u32) callconv(.nvptx_kernel) void {
    const t = gid();
    if (t < n) dst[t] += src[t];
}

export fn sum_to(
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
    const t = gid();
    if (t >= out_len) return;
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
    var q: usize = 0;
    while (q < reduce_count) : (q += 1) {
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
    out[t] = if (accumulate != 0) out[t] + total else total;
}

export fn ce_rows(x: [*]const f32, logp: [*]f32, labels: [*]const u8, row_loss: [*]f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    const r = gid();
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

export fn ce_mean(row_loss: [*]const f32, out: [*]f32, rows: u32) callconv(.nvptx_kernel) void {
    if (gid() != 0) return;
    var total: f32 = 0;
    var r: usize = 0;
    while (r < rows) : (r += 1) total += row_loss[r];
    out[0] = total / @as(f32, @floatFromInt(rows));
}

export fn ce_bwd(logp: [*]const f32, labels: [*]const u8, grad: [*]f32, loss_grad: [*]const f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    const t = gid();
    if (t >= @as(usize, rows) * cols) return;
    const r = t / cols;
    const j = t % cols;
    var p = expf(logp[t]);
    if (j == labels[r]) p -= 1;
    grad[t] += p * (loss_grad[0] / @as(f32, @floatFromInt(rows)));
}

export fn sgd(w: [*]f32, g: [*]const f32, lr: f32, n: u32) callconv(.nvptx_kernel) void {
    const t = gid();
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
    const t = gid();
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
    const t = gid();
    if (t >= @as(usize, rows) * pixels) return;
    const r = t / pixels;
    const p = t % pixels;
    const idx: usize = order[start + r];
    batch[t] = images[idx * pixels + p];
    if (p == 0) labels_out[r] = labels_src[idx];
}
