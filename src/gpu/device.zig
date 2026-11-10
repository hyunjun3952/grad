const aug = @import("math/augment.zig");
const ga = @import("args.zig");
const cv = @import("math/conv.zig");
const drop = @import("math/dropout.zig");
const nm = @import("math/norm.zig");

const BK = 8;
const threads = 256;

const reduction = struct {
    var partial: [threads]f32 addrspace(.shared) = undefined;
};

inline fn blockSum(value: f32, tid: usize) f32 {
    reduction.partial[tid] = value;
    barrier();
    var half: usize = threads / 2;
    while (half > 0) : (half /= 2) {
        if (tid < half) reduction.partial[tid] += reduction.partial[tid + half];
        barrier();
    }
    const total = reduction.partial[0];
    barrier();
    return total;
}

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

fn Gemm(comptime BM: usize, comptime BN: usize) type {
    return struct {
        const BM_PAD = BM + 1;
        const BN_PAD = BN + 1;
        const TM = BM / 16;
        const TN = BN / 16;

        var tile_a: [BK * BM_PAD]f32 addrspace(.shared) = undefined;
        var tile_b: [BK * BN_PAD]f32 addrspace(.shared) = undefined;

        fn run(comptime ta: bool, comptime tb: bool, a: [*]const f32, b: [*]const f32, c: [*]f32, args: ga.GemmShape) void {
            const tid: usize = @workItemId(0);
            const tx = tid % 16;
            const ty = tid / 16;
            const row0: usize = @as(usize, @workGroupId(1)) * BM;
            const col0: usize = @as(usize, @workGroupId(0)) * BN;
            const M: usize = args.m;
            const N: usize = args.n;
            const K: usize = args.k;

            var acc: [TM][TN]f32 = @splat(@splat(0));
            const k_begin = @as(usize, @workGroupId(2)) * args.chunk;
            const k_end = @min(K, k_begin + args.chunk);
            var kk = k_begin;
            while (kk < k_end) : (kk += BK) {
                inline for (0..BK * BM / threads) |i| {
                    const idx = tid + i * threads;
                    const r = if (ta) idx % BM else idx / BK;
                    const cc = if (ta) idx / BM else idx % BK;
                    const row = row0 + r;
                    const col = kk + cc;
                    const valid = row < M and col < K;
                    tile_a[cc * BM_PAD + r] = if (!valid) 0 else if (ta) a[col * M + row] else a[row * K + col];
                }
                inline for (0..BK * BN / threads) |i| {
                    const idx = tid + i * threads;
                    const r = if (tb) idx % BK else idx / BN;
                    const cc = if (tb) idx / BK else idx % BN;
                    const row = kk + r;
                    const col = col0 + cc;
                    const valid = row < K and col < N;
                    tile_b[r * BN_PAD + cc] = if (!valid) 0 else if (tb) b[col * K + row] else b[row * N + col];
                }
                barrier();

                inline for (0..BK) |p| {
                    var av: [TM]f32 = undefined;
                    var bv: [TN]f32 = undefined;
                    inline for (0..TM) |i| av[i] = tile_a[p * BM_PAD + ty + i * 16];
                    inline for (0..TN) |j| bv[j] = tile_b[p * BN_PAD + tx + j * 16];
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
                        if (args.chunk < K) {
                            _ = @atomicRmw(f32, &c[at], .Add, acc[i][j], .monotonic);
                        } else {
                            c[at] = if (args.accumulate != 0) c[at] + acc[i][j] else acc[i][j];
                        }
                    }
                }
            }
        }
    };
}

fn exportGemm(comptime name: []const u8, comptime BM: usize, comptime BN: usize, comptime ta: bool, comptime tb: bool) void {
    const entry = struct {
        fn kernel(a: [*]const f32, b: [*]const f32, c: [*]f32, shape: ga.GemmShape) callconv(.nvptx_kernel) void {
            Gemm(BM, BN).run(ta, tb, a, b, c, shape);
        }
    };
    @export(&entry.kernel, .{ .name = name });
}

comptime {
    @setEvalBranchQuota(100000);
    exportGemm("sgemmNN", 128, 128, false, false);
    exportGemm("sgemmNT", 128, 128, false, true);
    exportGemm("sgemmTN", 128, 128, true, false);
    exportGemm("sgemmTallNN", 128, 32, false, false);
    exportGemm("sgemmTallNT", 128, 32, false, true);
    exportGemm("sgemmTallTN", 128, 32, true, false);
    exportGemm("sgemmSmallNN", 32, 32, false, false);
    exportGemm("sgemmSmallNT", 32, 32, false, true);
    exportGemm("sgemmSmallTN", 32, 32, true, false);
}
export fn elementwise(a: [*]const f32, b: [*]const f32, out: [*]f32, args: ga.ElementwiseArgs) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= args.len) return;
    var rest = t;
    const p3 = rest % args.dims[3];
    rest /= args.dims[3];
    const p2 = rest % args.dims[2];
    rest /= args.dims[2];
    const p1 = rest % args.dims[1];
    const p0 = rest / args.dims[1];
    const x = a[p0 * args.a_strides[0] + p1 * args.a_strides[1] + p2 * args.a_strides[2] + p3 * args.a_strides[3]];
    const y = b[p0 * args.b_strides[0] + p1 * args.b_strides[1] + p2 * args.b_strides[2] + p3 * args.b_strides[3]];
    out[t] = if (args.multiply != 0) x * y else x + y;
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

export fn sumTo(in: [*]const f32, out: [*]f32, args: ga.SumArgs) callconv(.nvptx_kernel) void {
    const t: usize = @workGroupId(0);
    const tid: usize = @workItemId(0);
    const d = [4]usize{ args.dims[0], args.dims[1], args.dims[2], args.dims[3] };
    const s = [4]usize{ args.strides[0], args.strides[1], args.strides[2], args.strides[3] };
    const rmask = args.mask;

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
    while (q < args.reduce_count) : (q += threads) {
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
    const sum = blockSum(total, tid);
    if (tid == 0) out[t] = if (args.accumulate != 0) out[t] + sum else sum;
}

const column = struct {
    var first: [threads]f32 addrspace(.shared) = undefined;
    var second: [threads]f32 addrspace(.shared) = undefined;
};

export fn columnReduce(
    a: [*]const f32,
    b: [*]const f32,
    mean: [*]const f32,
    inv_std: [*]const f32,
    out: [*]f32,
    rows: u32,
    c: u32,
    row_stride: u32,
    kind: u32,
    scale: f32,
) callconv(.nvptx_kernel) void {
    const tid: usize = @workItemId(0);
    const lane = tid % 32;
    const group = tid / 32;
    const ch = @as(usize, @workGroupId(1)) * 32 + lane;
    const valid = ch < c;
    var first: f32 = 0;
    var second: f32 = 0;
    if (valid) {
        const m = mean[ch] * scale;
        const s = inv_std[ch];
        var r = @as(usize, @workGroupId(0)) * 8 + group;
        while (r < rows) : (r += row_stride) {
            const v = a[r * c + ch];
            if (kind == 0) {
                first += v;
            } else if (kind == 1) {
                first += (v - m) * (v - m);
            } else {
                first += v;
                second += v * ((b[r * c + ch] - m) * s);
            }
        }
    }
    column.first[tid] = first;
    column.second[tid] = second;
    barrier();
    if (group == 0 and valid) {
        var total_first: f32 = 0;
        var total_second: f32 = 0;
        inline for (0..8) |g| {
            total_first += column.first[lane + 32 * g];
            total_second += column.second[lane + 32 * g];
        }
        _ = @atomicRmw(f32, &out[ch], .Add, total_first, .monotonic);
        if (kind == 2) _ = @atomicRmw(f32, &out[c + ch], .Add, total_second, .monotonic);
    }
}

export fn batchNormFinish(sums: [*]const f32, mean: [*]f32, inv_std: [*]f32, running_mean: [*]f32, running_var: [*]f32, rows: u32, c: u32, training: u32) callconv(.nvptx_kernel) void {
    const ch = globalIndex();
    if (ch >= c) return;
    if (training == 0) {
        mean[ch] = running_mean[ch];
        inv_std[ch] = 1 / @sqrt(running_var[ch] + nm.eps);
        return;
    }
    const n: f32 = @floatFromInt(rows);
    const mu = sums[ch] / n;
    const variance = sums[c + ch] / n;
    mean[ch] = mu;
    inv_std[ch] = 1 / @sqrt(variance + nm.eps);
    running_mean[ch] += nm.momentum * (mu - running_mean[ch]);
    running_var[ch] += nm.momentum * (variance - running_var[ch]);
}

export fn batchNormGradFinish(sums: [*]const f32, gamma_grad: [*]f32, beta_grad: [*]f32, c: u32) callconv(.nvptx_kernel) void {
    const ch = globalIndex();
    if (ch >= c) return;
    beta_grad[ch] += sums[ch];
    gamma_grad[ch] += sums[c + ch];
}
export fn batchNormApply(x: [*]const f32, y: [*]f32, len: u32, c: u32, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) nm.apply(globalIndex(), x, y, c, mean, inv_std, gamma, beta);
}

export fn batchNormGradInput(x: [*]const f32, dy: [*]const f32, dx: [*]f32, len: u32, c: u32, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, sums: [*]const f32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) nm.gradInput(globalIndex(), x, dy, dx, c, len / c, mean, inv_std, gamma, sums);
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

export fn im2col(x: [*]const f32, col: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.im2col(@intCast(globalIndex()), x, col, d);
}

export fn col2im(dcol: [*]const f32, dx: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.col2im(@intCast(globalIndex()), dcol, dx, d);
}

export fn maxPool(x: [*]const f32, out: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.maxPool(@intCast(globalIndex()), x, out, d);
}

export fn avgPool(x: [*]const f32, out: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.avgPool(@intCast(globalIndex()), x, out, d);
}

export fn avgPoolBackward(gout: [*]const f32, gin: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.avgPoolBackward(@intCast(globalIndex()), gout, gin, d);
}

export fn maxPoolBackward(x: [*]const f32, gout: [*]const f32, gin: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.maxPoolBackward(@intCast(globalIndex()), x, gout, gin, d);
}

export fn dropout(src: [*]const f32, dst: [*]f32, len: u32, seed: u64, keep: f32, accumulate: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) drop.apply(globalIndex(), src, dst, seed, keep, accumulate != 0);
}

const Slot = struct { w: u64, g: u64, m: u64, v: u64, at: usize, valid: bool };

fn locate(table: ga.TensorTable, t: usize) Slot {
    var slot: Slot = .{ .w = 0, .g = 0, .m = 0, .v = 0, .at = t, .valid = false };
    inline for (0..ga.max_tensors) |i| {
        if (!slot.valid and i < table.count) {
            if (slot.at < table.len[i]) {
                slot.w = table.w[i];
                slot.g = table.g[i];
                slot.m = table.m[i];
                slot.v = table.v[i];
                slot.valid = true;
            } else {
                slot.at -= table.len[i];
            }
        }
    }
    return slot;
}

fn pointer(comptime T: type, address: u64) T {
    return @ptrFromInt(address);
}

export fn zeroGrads(table: ga.TensorTable) callconv(.nvptx_kernel) void {
    const slot = locate(table, globalIndex());
    if (slot.valid) pointer([*]f32, slot.g)[slot.at] = 0;
}

export fn sgd(table: ga.TensorTable, lr: f32) callconv(.nvptx_kernel) void {
    const slot = locate(table, globalIndex());
    if (slot.valid) pointer([*]f32, slot.w)[slot.at] -= lr * pointer([*]const f32, slot.g)[slot.at];
}

export fn adam(table: ga.TensorTable, hp: ga.AdamArgs) callconv(.nvptx_kernel) void {
    const slot = locate(table, globalIndex());
    if (!slot.valid) return;
    const w = pointer([*]f32, slot.w) + slot.at;
    const m = pointer([*]f32, slot.m) + slot.at;
    const v = pointer([*]f32, slot.v) + slot.at;
    const grad = pointer([*]const f32, slot.g)[slot.at];
    const m_new = hp.beta1 * m[0] + (1 - hp.beta1) * grad;
    const v_new = hp.beta2 * v[0] + (1 - hp.beta2) * grad * grad;
    m[0] = m_new;
    v[0] = v_new;
    w[0] -= hp.step * m_new / (@sqrt(v_new) * hp.inv_sqrt_c2 + hp.eps) + hp.decay * w[0];
}
export fn gather(images: [*]const f32, labels_src: [*]const u8, order: [*]const u32, batch: [*]f32, labels_out: [*]u8, args: ga.GatherArgs) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, args.rows) * args.pixels) return;
    const r = t / args.pixels;
    const p = t % args.pixels;
    const idx: usize = order[args.start + r];
    const shift = aug.shiftOf(args.seed, args.start + r, args.max_shift);
    batch[t] = aug.pixel(images + idx * args.pixels, p, aug.side(args.pixels), shift);
    if (p == 0) labels_out[r] = labels_src[idx];
}
