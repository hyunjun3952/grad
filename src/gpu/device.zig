const aug = @import("math/augment.zig");
const bf16 = @import("math/bf16.zig");
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

export fn addRelu(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) out[t] = @max(a[t] + b[t], 0);
}

export fn addReluBackward(out: [*]const f32, g: [*]const f32, grad_a: [*]f32, grad_b: [*]f32, n: u32, flags: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= n) return;
    const v = if (out[t] > 0) g[t] else 0;
    if (flags & 1 != 0) grad_a[t] += v;
    if (flags & 2 != 0) grad_b[t] += v;
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
    mask: [*]const f32,
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
            } else if (kind == 2) {
                first += v - m;
                second += (v - m) * (v - m);
            } else {
                const g = if (kind == 3 and mask[r * c + ch] <= 0) 0 else v;
                first += g;
                second += g * ((b[r * c + ch] - m) * s);
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
        if (kind != 0) _ = @atomicRmw(f32, &out[c + ch], .Add, total_second, .monotonic);
    }
}

export fn batchNormFinish(sums: [*]const f32, shift: [*]const f32, mean: [*]f32, inv_std: [*]f32, running_mean: [*]f32, running_var: [*]f32, rows: u32, c: u32, training: u32) callconv(.nvptx_kernel) void {
    const ch = globalIndex();
    if (ch >= c) return;
    if (training == 0) {
        mean[ch] = running_mean[ch];
        inv_std[ch] = 1 / @sqrt(running_var[ch] + nm.eps);
        return;
    }
    const n: f32 = @floatFromInt(rows);
    const offset = sums[ch] / n;
    const mu = shift[ch] + offset;
    const variance = @max(sums[c + ch] / n - offset * offset, 0);
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
export fn batchNormApply(x: [*]const f32, y: [*]f32, len: u32, c: u32, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32, fuse_relu: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) nm.apply(globalIndex(), x, y, c, mean, inv_std, gamma, beta, fuse_relu != 0);
}

export fn batchNormGradInput(x: [*]const f32, dy: [*]const f32, dx: [*]f32, len: u32, c: u32, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, sums: [*]const f32, mask: [*]const f32, use_mask: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) nm.gradInput(globalIndex(), x, dy, dx, c, len / c, mean, inv_std, gamma, sums, mask, use_mask != 0);
}

export fn crossEntropyRows(x: [*]const f32, logp: [*]f32, labels: [*]const u8, row_loss: [*]f32, rows: u32, cols: u32, smoothing: f32) callconv(.nvptx_kernel) void {
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
    var logp_sum: f32 = 0;
    j = 0;
    while (j < cols) : (j += 1) {
        logp[base + j] = x[base + j] - shift;
        logp_sum += logp[base + j];
    }
    row_loss[r] = -(1 - smoothing) * logp[base + labels[r]] - smoothing / @as(f32, @floatFromInt(cols)) * logp_sum;
}

export fn crossEntropyMean(row_loss: [*]const f32, out: [*]f32, rows: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() != 0) return;
    var total: f32 = 0;
    var r: usize = 0;
    while (r < rows) : (r += 1) total += row_loss[r];
    out[0] = total / @as(f32, @floatFromInt(rows));
}

export fn crossEntropyBackward(logp: [*]const f32, labels: [*]const u8, grad: [*]f32, loss_grad: [*]const f32, rows: u32, cols: u32, smoothing: f32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, rows) * cols) return;
    const r = t / cols;
    const j = t % cols;
    var p = expf(logp[t]) - smoothing / @as(f32, @floatFromInt(cols));
    if (j == labels[r]) p -= 1 - smoothing;
    grad[t] += p * (loss_grad[0] / @as(f32, @floatFromInt(rows)));
}

export fn im2col(x: [*]const f32, col: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.im2col(@intCast(globalIndex()), x, col, d);
}

export fn col2im(dcol: [*]const f32, dx: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.col2im(@intCast(globalIndex()), dcol, dx, d);
}

export fn col2imHalf(dcol: [*]const u16, dx: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.col2imHalf(@intCast(globalIndex()), dcol, dx, d);
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
    const config: aug.Config = .{ .max_shift = args.max_shift, .max_angle = args.max_angle, .max_scale = args.max_scale, .flip = args.flip != 0, .channels = args.channels, .seed = args.seed };
    const source = images + idx * args.pixels;
    batch[t] = if (config.active()) aug.pixel(source, p, aug.side(args.pixels, args.channels), args.channels, aug.transformOf(args.start + r, config)) else source[p];
    if (p == 0) labels_out[r] = labels_src[idx];
}

const tensor_core = struct {
    const rows = 128;
    const cols = 64;
    const depth = 32;
    const stages = 3;
    const a_elems = rows * depth;
    const b_elems = depth * cols;
    const stage_elems = a_elems + b_elems;

    var smem: [stages * stage_elems]u16 align(128) addrspace(.shared) = undefined;

    inline fn aSlot(row: usize, chunk: usize) usize {
        return row * depth + ((chunk ^ ((row / 2) % 4)) * 8);
    }

    inline fn bSlot(k: usize, chunk: usize) usize {
        return k * cols + ((chunk ^ (k % 8)) * 8);
    }

    inline fn sharedAddress(index: usize) u32 {
        return @truncate(@intFromPtr(&smem[index]));
    }

    inline fn copyAsync(dst: u32, src: [*]const u16) void {
        asm volatile ("cp.async.cg.shared.global [%[dst]], [%[src]], 16;"
            :
            : [dst] "r" (dst),
              [src] "l" (@intFromPtr(src)),
            : .{ .memory = true });
    }

    inline fn commit() void {
        asm volatile ("cp.async.commit_group;" ::: .{ .memory = true });
    }

    inline fn waitForOlderGroups() void {
        asm volatile ("cp.async.wait_group 1;" ::: .{ .memory = true });
    }

    inline fn ldmatrix(address: u32) [4]u32 {
        var r0: u32 = undefined;
        var r1: u32 = undefined;
        var r2: u32 = undefined;
        var r3: u32 = undefined;
        asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%[r0], %[r1], %[r2], %[r3]}, [%[address]];"
            : [r0] "=r" (r0),
              [r1] "=r" (r1),
              [r2] "=r" (r2),
              [r3] "=r" (r3),
            : [address] "r" (address),
        );
        return .{ r0, r1, r2, r3 };
    }

    inline fn ldmatrixTransposed(address: u32) [4]u32 {
        var r0: u32 = undefined;
        var r1: u32 = undefined;
        var r2: u32 = undefined;
        var r3: u32 = undefined;
        asm volatile ("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%[r0], %[r1], %[r2], %[r3]}, [%[address]];"
            : [r0] "=r" (r0),
              [r1] "=r" (r1),
              [r2] "=r" (r2),
              [r3] "=r" (r3),
            : [address] "r" (address),
        );
        return .{ r0, r1, r2, r3 };
    }

    inline fn mma(acc: *[4]f32, a: [4]u32, b: [2]u32) void {
        var d0 = acc[0];
        var d1 = acc[1];
        var d2 = acc[2];
        var d3 = acc[3];
        asm volatile ("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%[d0], %[d1], %[d2], %[d3]}, {%[a0], %[a1], %[a2], %[a3]}, {%[b0], %[b1]}, {%[d0], %[d1], %[d2], %[d3]};"
            : [d0] "+f" (d0),
              [d1] "+f" (d1),
              [d2] "+f" (d2),
              [d3] "+f" (d3),
            : [a0] "r" (a[0]),
              [a1] "r" (a[1]),
              [a2] "r" (a[2]),
              [a3] "r" (a[3]),
              [b0] "r" (b[0]),
              [b1] "r" (b[1]),
        );
        acc.* = .{ d0, d1, d2, d3 };
    }
};
export fn packHalf(src: [*]const f32, dst: [*]u16, args: ga.PackArgs) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, args.rows_padded) * args.cols_padded) return;
    const i = t / args.cols_padded;
    const j = t % args.cols_padded;
    const inside = i < args.rows and j < args.cols;
    const source = if (args.transpose != 0) j * args.src_stride + i else i * args.src_stride + j;
    dst[t] = if (inside) bf16.encode(src[source]) else 0;
}

const transpose_tile = struct {
    var values: [32 * 33]f32 addrspace(.shared) = undefined;
};

export fn im2colHalf(x: [*]const f32, dst: [*]u16, d: cv.Dims, cols_padded: u32) callconv(.nvptx_kernel) void {
    const row: u32 = @workGroupId(0);
    const real = row < d.n * cv.outHeight(d) * cv.outWidth(d);
    const patch = cv.patchOf(row, d);
    const kc = d.k * d.k * d.c;
    const line = dst + @as(usize, row) * cols_padded;
    if (d.c % 4 == 0) {
        var cc: u32 = @workItemId(0) * 4;
        while (cc < cols_padded) : (cc += threads * 4) {
            const values: @Vector(4, f32) = if (real and cc < kc) cv.patchValues(x, patch, cc, d) else @splat(0);
            const packed_pair: *align(8) u64 = @ptrCast(@alignCast(line + cc));
            packed_pair.* = @as(u64, bf16.encode(values[0])) | @as(u64, bf16.encode(values[1])) << 16 | @as(u64, bf16.encode(values[2])) << 32 | @as(u64, bf16.encode(values[3])) << 48;
        }
    } else {
        var cc: u32 = @workItemId(0);
        while (cc < cols_padded) : (cc += threads) {
            line[cc] = if (real and cc < kc) bf16.encode(cv.patchValue(x, patch, cc, d)) else 0;
        }
    }
}
export fn packHalfTransposed(src: [*]const f32, dst: [*]u16, args: ga.PackArgs) callconv(.nvptx_kernel) void {
    const tid: usize = @workItemId(0);
    const lane = tid % 32;
    const group = tid / 32;
    const row_base = @as(usize, @workGroupId(1)) * 32;
    const col_base = @as(usize, @workGroupId(0)) * 32;
    inline for (0..4) |step| {
        const j = col_base + group + 8 * step;
        const i = row_base + lane;
        transpose_tile.values[(group + 8 * step) * 33 + lane] = if (i < args.rows and j < args.cols) src[j * args.src_stride + i] else 0;
    }
    barrier();
    inline for (0..4) |step| {
        const i = row_base + group + 8 * step;
        dst[i * args.cols_padded + col_base + lane] = bf16.encode(transpose_tile.values[lane * 33 + group + 8 * step]);
    }
}

const GemmMode = enum { dense, gather_a, gather_b };

const zero_geometry: ga.ConvGeometry = .{ .n = 0, .h = 0, .w = 0, .c = 0, .k = 0, .stride = 0, .pad = 0, .out_h = 0, .out_w = 0, .positions = 0 };

inline fn copyAsyncZfill(dst: u32, src: [*]const u16, valid: bool) void {
    asm volatile ("cp.async.cg.shared.global [%[dst]], [%[src]], 16, %[size];"
        :
        : [dst] "r" (dst),
          [src] "l" (@intFromPtr(src)),
          [size] "r" (@as(u32, if (valid) 16 else 0)),
        : .{ .memory = true });
}

inline fn tensorGemm(comptime mode: GemmMode, a: [*]const u16, b: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) void {
    const tc = tensor_core;
    const tid: usize = @workItemId(0);
    const lane = tid % 32;
    const warp = tid / 32;
    const warp_row = warp / 2;
    const warp_col = warp % 2;
    const row0 = @as(usize, @workGroupId(1)) * tc.rows;
    const col0 = @as(usize, @workGroupId(0)) * tc.cols;
    const M: usize = shape.m;
    const N: usize = shape.n;
    const K: usize = shape.k_padded;
    const NP: usize = shape.n_padded;
    const k_begin = @as(usize, @workGroupId(2)) * shape.chunk;
    const k_end = @min(K, k_begin + shape.chunk);
    const tiles = (k_end -| k_begin) / tc.depth;

    const Loader = struct {
        a: [*]const u16,
        b: [*]const u16,
        geo: ga.ConvGeometry,
        rows_a: [2]usize,
        chunk_a: usize,
        dst_a: [2]usize,
        dst_b: usize,
        k_b: usize,
        chunk_b: usize,
        first_col: usize,
        depth_k: usize,
        stride_n: usize,
        row_base: usize,
        origin_n: [2]u32,
        origin_y: [2]i32,
        origin_x: [2]i32,
        origin_valid: [2]bool,
        cursor_ky: i32,
        cursor_kx: i32,
        cursor_channel: usize,

        inline fn issue(self: *@This(), stage: usize, kk: usize) void {
            const base = stage * tc.stage_elems;
            const geometry = self.geo;
            if (mode == .gather_a) {
                inline for (0..2) |i| {
                    const y = self.origin_y[i] + self.cursor_ky;
                    const x = self.origin_x[i] + self.cursor_kx;
                    const valid = self.origin_valid[i] and y >= 0 and y < @as(i32, @intCast(geometry.h)) and x >= 0 and x < @as(i32, @intCast(geometry.w));
                    const at = if (valid) ((@as(usize, self.origin_n[i]) * geometry.h + @as(usize, @intCast(y))) * geometry.w + @as(usize, @intCast(x))) * geometry.c + self.cursor_channel + self.chunk_a * 8 else 0;
                    copyAsyncZfill(tc.sharedAddress(base + self.dst_a[i]), self.a + at, valid);
                }
                self.cursor_channel += tc.depth;
                if (self.cursor_channel >= geometry.c) {
                    self.cursor_channel -= geometry.c;
                    self.cursor_kx += 1;
                    if (self.cursor_kx == @as(i32, @intCast(geometry.k))) {
                        self.cursor_kx = 0;
                        self.cursor_ky += 1;
                    }
                }
            } else {
                inline for (0..2) |i| tc.copyAsync(tc.sharedAddress(base + self.dst_a[i]), self.a + (self.row_base + self.rows_a[i]) * self.depth_k + self.chunk_a * 8 + kk);
            }
            if (mode == .gather_b) {
                const tap: u32 = @intCast(self.first_col / geometry.c);
                const channel: usize = self.first_col % geometry.c + self.chunk_b * 8;
                const ky: i32 = @intCast(tap / geometry.k);
                const kx: i32 = @intCast(tap % geometry.k);
                const p: u32 = @intCast(kk + self.k_b);
                const y = @as(i32, @intCast(p / geometry.out_w % geometry.out_h * geometry.stride)) + ky - @as(i32, @intCast(geometry.pad));
                const x = @as(i32, @intCast(p % geometry.out_w * geometry.stride)) + kx - @as(i32, @intCast(geometry.pad));
                const valid = p < geometry.positions and y >= 0 and y < @as(i32, @intCast(geometry.h)) and x >= 0 and x < @as(i32, @intCast(geometry.w));
                const at = if (valid) ((@as(usize, p / (geometry.out_h * geometry.out_w)) * geometry.h + @as(usize, @intCast(y))) * geometry.w + @as(usize, @intCast(x))) * geometry.c + channel else 0;
                copyAsyncZfill(tc.sharedAddress(base + self.dst_b), self.b + at, valid);
            } else {
                tc.copyAsync(tc.sharedAddress(base + self.dst_b), self.b + (kk + self.k_b) * self.stride_n + self.first_col + self.chunk_b * 8);
            }
        }
    };

    const a_rows = [2]usize{ tid / 4, tid / 4 + 64 };
    const a_chunk = tid % 4;
    const b_k = tid / 8;
    const b_chunk = tid % 8;
    var loader: Loader = .{
        .a = a,
        .b = b,
        .geo = geo,
        .rows_a = a_rows,
        .chunk_a = a_chunk,
        .dst_a = .{ tc.aSlot(a_rows[0], a_chunk), tc.aSlot(a_rows[1], a_chunk) },
        .dst_b = tc.a_elems + tc.bSlot(b_k, b_chunk),
        .k_b = b_k,
        .chunk_b = b_chunk,
        .first_col = col0,
        .depth_k = K,
        .stride_n = NP,
        .row_base = row0,
        .origin_n = undefined,
        .origin_y = undefined,
        .origin_x = undefined,
        .origin_valid = undefined,
        .cursor_ky = 0,
        .cursor_kx = 0,
        .cursor_channel = 0,
    };
    if (mode == .gather_a) {
        inline for (0..2) |i| {
            const row: u32 = @intCast(row0 + a_rows[i]);
            loader.origin_valid[i] = row < shape.m;
            loader.origin_n[i] = row / (geo.out_h * geo.out_w);
            loader.origin_y[i] = @as(i32, @intCast(row / geo.out_w % geo.out_h * geo.stride)) - @as(i32, @intCast(geo.pad));
            loader.origin_x[i] = @as(i32, @intCast(row % geo.out_w * geo.stride)) - @as(i32, @intCast(geo.pad));
        }
        const tap: u32 = @intCast(k_begin / geo.c);
        loader.cursor_ky = @intCast(tap / geo.k);
        loader.cursor_kx = @intCast(tap % geo.k);
        loader.cursor_channel = k_begin % geo.c;
    }

    inline for (0..tc.stages - 1) |stage| {
        if (stage < tiles) loader.issue(stage, k_begin + stage * tc.depth);
        tc.commit();
    }
    var acc: [2][4][4]f32 = @splat(@splat(@splat(0)));
    var tile: usize = 0;
    while (tile < tiles) : (tile += 1) {
        tc.waitForOlderGroups();
        barrier();

        const ahead = tile + tc.stages - 1;
        if (ahead < tiles) loader.issue(ahead % tc.stages, k_begin + ahead * tc.depth);
        tc.commit();

        const base = (tile % tc.stages) * tc.stage_elems;
        inline for (0..tc.depth / 16) |step| {
            var a_frag: [2][4]u32 = undefined;
            inline for (0..2) |mi| {
                const row = warp_row * 32 + mi * 16 + lane % 8 + 8 * ((lane / 8) % 2);
                a_frag[mi] = tc.ldmatrix(tc.sharedAddress(base + tc.aSlot(row, step * 2 + lane / 16)));
            }
            inline for (0..2) |pair| {
                const k = step * 16 + lane % 8 + 8 * ((lane / 8) % 2);
                const frag = tc.ldmatrixTransposed(tc.sharedAddress(base + tc.a_elems + tc.bSlot(k, warp_col * 4 + pair * 2 + lane / 16)));
                inline for (0..2) |mi| {
                    tc.mma(&acc[mi][pair * 2], a_frag[mi], .{ frag[0], frag[1] });
                    tc.mma(&acc[mi][pair * 2 + 1], a_frag[mi], .{ frag[2], frag[3] });
                }
            }
        }
    }

    inline for (0..2) |mi| {
        inline for (0..4) |ni| {
            inline for (0..4) |q| {
                const r = row0 + warp_row * 32 + mi * 16 + lane / 4 + (q / 2) * 8;
                const col = col0 + warp_col * 32 + ni * 8 + (lane % 4) * 2 + q % 2;
                if (r < M and col < N) {
                    const at = if (shape.transpose_out != 0) col * M + r else r * N + col;
                    if (shape.output_half != 0) {
                        const half_out: [*]u16 = @ptrCast(@alignCast(c));
                        half_out[at] = bf16.encode(acc[mi][ni][q]);
                    } else if (shape.chunk < K) {
                        _ = @atomicRmw(f32, &c[at], .Add, acc[mi][ni][q], .monotonic);
                    } else {
                        c[at] = if (shape.accumulate != 0) c[at] + acc[mi][ni][q] else acc[mi][ni][q];
                    }
                }
            }
        }
    }
}

export fn sgemmTensor(a: [*]const u16, b: [*]const u16, c: [*]f32, shape: ga.TensorShape) callconv(.nvptx_kernel) void {
    tensorGemm(.dense, a, b, c, shape, zero_geometry);
}

export fn convGemmForward(x: [*]const u16, w: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tensorGemm(.gather_a, x, w, c, shape, geo);
}

export fn convGemmWeightGrad(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tensorGemm(.gather_b, dy_t, x, c, shape, geo);
}

export fn packWeightsRotated(src: [*]const f32, dst: [*]u16, args: ga.RotateArgs) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, args.rows_padded) * args.cols_padded) return;
    const i: u32 = @intCast(t / args.cols_padded);
    const j: u32 = @intCast(t % args.cols_padded);
    const tap = i / args.c_out;
    const co = i % args.c_out;
    const source_tap = (args.k - 1 - tap / args.k) * args.k + (args.k - 1 - tap % args.k);
    dst[t] = if (j < args.c_in) bf16.encode(src[(source_tap * args.c_in + j) * args.c_out + co]) else 0;
}
