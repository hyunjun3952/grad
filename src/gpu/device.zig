const act = @import("math/activation.zig");
const aug = @import("math/augment.zig");
const bf16 = @import("math/bf16.zig");
const gru = @import("math/gru.zig");
const ga = @import("args.zig");
const cv = @import("math/conv.zig");
const drop = @import("math/dropout.zig");
const nm = @import("math/norm.zig");
const soft = @import("math/softmax.zig");
const ln = @import("math/layernorm.zig");

// Device code. Every export fn is a kernel that gpu.zig launches by name (see its Kernel enum),
// and per-element math shared with the CPU lives in math/*.zig.

// ***** helpers *****

const BK = 8;
const threads = 256;

const reduction = struct {
    var partial: [threads]f32 addrspace(.shared) = undefined;
};

// Adds value across the threads of a block with a shared-memory tree and returns the total to every thread.
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

// Waits until every thread of the block has reached this point.
inline fn barrier() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}

inline fn globalIndex() usize {
    return @as(usize, @workGroupId(0)) * @workGroupSize(0) + @workItemId(0);
}

// NOTE: ex2.approx is much faster than a libm exp and accurate to about two ulp, which softmax and the losses tolerate.
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

// ***** f32 GEMM *****

// A register-tiled f32 matrix product: a block of 16 x 16 threads computes a BM x BN tile of C,
// each thread a (BM / 16) x (BN / 16) patch. The tiles are padded by one column to avoid shared-memory bank conflicts.
fn Gemm(comptime BM: usize, comptime BN: usize) type {
    return struct {
        const BM_PAD = BM + 1;
        const BN_PAD = BN + 1;
        const TM = BM / 16;
        const TN = BN / 16;

        var tile_a: [BK * BM_PAD]f32 addrspace(.shared) = undefined;
        var tile_b: [BK * BN_PAD]f32 addrspace(.shared) = undefined;

        fn run(comptime ta: bool, comptime tb: bool, a_all: [*]const f32, b_all: [*]const f32, c_all: [*]f32, args: ga.GemmShape) void {
            const tid: usize = @workItemId(0);
            const tx = tid % 16;
            const ty = tid / 16;
            // NOTE: a batched product never splits K, so work group z is free to pick the matrix.
            const z: usize = @workGroupId(2);
            const batched = args.batch != 0;
            const a = if (batched) a_all + z * args.stride_a else a_all;
            const b = if (batched) b_all + z * args.stride_b else b_all;
            const c = if (batched) c_all + z * args.stride_c else c_all;
            const row0: usize = @as(usize, @workGroupId(1)) * BM;
            const col0: usize = @as(usize, @workGroupId(0)) * BN;
            const M: usize = args.m;
            const N: usize = args.n;
            const K: usize = args.k;

            var acc: [TM][TN]f32 = @splat(@splat(0));
            const k_begin = if (batched) 0 else z * args.chunk;
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
// ***** elementwise and reductions *****

inline fn elementwiseOf(comptime E: type, a: [*]const E, b: [*]const E, out: [*]E, args: ga.ElementwiseArgs) void {
    const t = globalIndex();
    if (t >= args.len) return;
    var rest = t;
    var from_a: usize = 0;
    var from_b: usize = 0;
    inline for (0..ga.max_rank) |step| {
        const axis = ga.max_rank - 1 - step;
        const position = rest % args.dims[axis];
        rest /= args.dims[axis];
        from_a += position * args.a_strides[axis];
        from_b += position * args.b_strides[axis];
    }
    const lhs = bf16.load(E, a, from_a);
    const rhs = bf16.load(E, b, from_b);
    bf16.store(E, out, t, if (args.multiply != 0) lhs * rhs else lhs + rhs);
}

export fn elementwise(a: [*]const f32, b: [*]const f32, out: [*]f32, args: ga.ElementwiseArgs) callconv(.nvptx_kernel) void {
    elementwiseOf(f32, a, b, out, args);
}

export fn elementwiseHalf(a: [*]const u16, b: [*]const u16, out: [*]u16, args: ga.ElementwiseArgs) callconv(.nvptx_kernel) void {
    elementwiseOf(u16, a, b, out, args);
}

export fn relu(x: [*]const f32, out: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) out[t] = @max(x[t], 0);
}

export fn reluHalf(x: [*]const u16, out: [*]u16, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) bf16.store(u16, out, t, @max(bf16.load(u16, x, t), 0));
}

export fn reluBackward(x: [*]const f32, g_out: [*]const f32, g_in: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) g_in[t] += if (x[t] > 0) g_out[t] else 0;
}

export fn reluBackwardHalf(x: [*]const u16, g_out: [*]const u16, g_in: [*]u16, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) bf16.store(u16, g_in, t, bf16.load(u16, g_in, t) + if (bf16.load(u16, x, t) > 0) bf16.load(u16, g_out, t) else 0);
}

export fn toHalf(x: [*]const f32, y: [*]u16, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) y[t] = bf16.encode(x[t]);
}

export fn fromHalf(x: [*]const u16, y: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) y[t] = bf16.decode(x[t]);
}

export fn addFromHalf(x: [*]const u16, y: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) y[t] += bf16.decode(x[t]);
}

export fn addToHalf(x: [*]const f32, y: [*]u16, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) y[t] = bf16.encode(bf16.decode(y[t]) + x[t]);
}

inline fn layerNormRowsOf(comptime E: type, x: [*]const E, y: [*]E, gamma: [*]const f32, beta: [*]const f32, mean: [*]f32, inv_std: [*]f32, rows: u32, cols: u32) void {
    const r = globalIndex();
    if (r >= rows) return;
    const row = ln.forwardRow(E, x + r * cols, y + r * cols, gamma, beta, cols);
    mean[r] = row.mean;
    inv_std[r] = row.inv_std;
}

export fn layerNormRows(x: [*]const f32, y: [*]f32, gamma: [*]const f32, beta: [*]const f32, mean: [*]f32, inv_std: [*]f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    layerNormRowsOf(f32, x, y, gamma, beta, mean, inv_std, rows, cols);
}

export fn layerNormRowsHalf(x: [*]const u16, y: [*]u16, gamma: [*]const f32, beta: [*]const f32, mean: [*]f32, inv_std: [*]f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    layerNormRowsOf(u16, x, y, gamma, beta, mean, inv_std, rows, cols);
}

inline fn layerNormBackwardOf(comptime E: type, x: [*]const E, dy: [*]const E, dx: [*]E, gamma: [*]const f32, mean: [*]const f32, inv_std: [*]const f32, rows: u32, cols: u32) void {
    const r = globalIndex();
    if (r < rows) ln.backwardRow(E, x + r * cols, dy + r * cols, dx + r * cols, gamma, .{ .mean = mean[r], .inv_std = inv_std[r] }, cols);
}

export fn layerNormBackward(x: [*]const f32, dy: [*]const f32, dx: [*]f32, gamma: [*]const f32, mean: [*]const f32, inv_std: [*]const f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    layerNormBackwardOf(f32, x, dy, dx, gamma, mean, inv_std, rows, cols);
}

export fn layerNormBackwardHalf(x: [*]const u16, dy: [*]const u16, dx: [*]u16, gamma: [*]const f32, mean: [*]const f32, inv_std: [*]const f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    layerNormBackwardOf(u16, x, dy, dx, gamma, mean, inv_std, rows, cols);
}

export fn gruGates(gx: [*]const f32, gh: [*]const f32, h: [*]const f32, update: [*]f32, reset_h: [*]f32, n: u32, hidden: u32) callconv(.nvptx_kernel) void {
    const i = globalIndex();
    if (i < n) gru.gatesForward(gx, gh, h, update, reset_h, i, hidden);
}

export fn gruGatesBackward(gx: [*]const f32, gh: [*]const f32, h: [*]const f32, d_update: [*]const f32, d_reset_h: [*]const f32, grad: gru.GatesGrad, flags: u32, n: u32, hidden: u32) callconv(.nvptx_kernel) void {
    const i = globalIndex();
    if (i < n) gru.gatesBackward(gx, gh, h, d_update, d_reset_h, grad, flags, i, hidden);
}

export fn gruNext(gx: [*]const f32, gc: [*]const f32, update: [*]const f32, h: [*]const f32, next: [*]f32, n: u32, hidden: u32) callconv(.nvptx_kernel) void {
    const i = globalIndex();
    if (i < n) gru.nextForward(gx, gc, update, h, next, i, hidden);
}

export fn gruNextBackward(gx: [*]const f32, gc: [*]const f32, update: [*]const f32, h: [*]const f32, d_next: [*]const f32, grad: gru.NextGrad, flags: u32, n: u32, hidden: u32) callconv(.nvptx_kernel) void {
    const i = globalIndex();
    if (i < n) gru.nextBackward(gx, gc, update, h, d_next, grad, flags, i, hidden);
}

// The table and its gradient are always f32; only the gathered rows (out) and their gradient (dy) may be bf16 (E).
inline fn embeddingGatherOf(comptime E: type, table: [*]const f32, ids: [*]const u32, out: [*]E, n: u32, width: u32) void {
    const t = globalIndex();
    if (t < @as(usize, n) * width) bf16.store(E, out, t, table[@as(usize, ids[t / width]) * width + t % width]);
}

export fn embeddingGather(table: [*]const f32, ids: [*]const u32, out: [*]f32, n: u32, width: u32) callconv(.nvptx_kernel) void {
    embeddingGatherOf(f32, table, ids, out, n, width);
}

export fn embeddingGatherHalf(table: [*]const f32, ids: [*]const u32, out: [*]u16, n: u32, width: u32) callconv(.nvptx_kernel) void {
    embeddingGatherOf(u16, table, ids, out, n, width);
}

inline fn embeddingScatterOf(comptime E: type, dy: [*]const E, ids: [*]const u32, dtable: [*]f32, n: u32, width: u32) void {
    const t = globalIndex();
    if (t < @as(usize, n) * width) _ = @atomicRmw(f32, &dtable[@as(usize, ids[t / width]) * width + t % width], .Add, bf16.load(E, dy, t), .monotonic);
}

export fn embeddingScatter(dy: [*]const f32, ids: [*]const u32, dtable: [*]f32, n: u32, width: u32) callconv(.nvptx_kernel) void {
    embeddingScatterOf(f32, dy, ids, dtable, n, width);
}

export fn embeddingScatterHalf(dy: [*]const u16, ids: [*]const u32, dtable: [*]f32, n: u32, width: u32) callconv(.nvptx_kernel) void {
    embeddingScatterOf(u16, dy, ids, dtable, n, width);
}

inline fn softmaxRowsOf(comptime E: type, x: [*]const E, y: [*]E, rows: u32, cols: u32, period: u32) void {
    const r = globalIndex();
    if (r < rows) soft.forwardRow(E, x + r * cols, y + r * cols, cols, soft.limitOf(r, cols, period));
}

export fn softmaxRows(x: [*]const f32, y: [*]f32, rows: u32, cols: u32, period: u32) callconv(.nvptx_kernel) void {
    softmaxRowsOf(f32, x, y, rows, cols, period);
}

export fn softmaxRowsHalf(x: [*]const u16, y: [*]u16, rows: u32, cols: u32, period: u32) callconv(.nvptx_kernel) void {
    softmaxRowsOf(u16, x, y, rows, cols, period);
}

inline fn softmaxBackwardOf(comptime E: type, y: [*]const E, dy: [*]const E, dx: [*]E, rows: u32, cols: u32) void {
    const r = globalIndex();
    if (r < rows) soft.backwardRow(E, y + r * cols, dy + r * cols, dx + r * cols, cols);
}

export fn softmaxBackward(y: [*]const f32, dy: [*]const f32, dx: [*]f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    softmaxBackwardOf(f32, y, dy, dx, rows, cols);
}

export fn softmaxBackwardHalf(y: [*]const u16, dy: [*]const u16, dx: [*]u16, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
    softmaxBackwardOf(u16, y, dy, dx, rows, cols);
}

inline fn activationOf(comptime E: type, x: [*]const E, out: [*]E, n: u32, kind: u32) void {
    const t = globalIndex();
    if (t < n) bf16.store(E, out, t, act.forward(@fromBackingInt(@intCast(kind)), bf16.load(E, x, t)));
}

export fn activation(x: [*]const f32, out: [*]f32, n: u32, kind: u32) callconv(.nvptx_kernel) void {
    activationOf(f32, x, out, n, kind);
}

export fn activationHalf(x: [*]const u16, out: [*]u16, n: u32, kind: u32) callconv(.nvptx_kernel) void {
    activationOf(u16, x, out, n, kind);
}

inline fn activationBackwardOf(comptime E: type, x: [*]const E, y: [*]const E, g_out: [*]const E, g_in: [*]E, n: u32, kind: u32) void {
    const t = globalIndex();
    if (t < n) bf16.store(E, g_in, t, bf16.load(E, g_in, t) + bf16.load(E, g_out, t) * act.derivative(@fromBackingInt(@intCast(kind)), bf16.load(E, x, t), bf16.load(E, y, t)));
}

export fn activationBackward(x: [*]const f32, y: [*]const f32, g_out: [*]const f32, g_in: [*]f32, n: u32, kind: u32) callconv(.nvptx_kernel) void {
    activationBackwardOf(f32, x, y, g_out, g_in, n, kind);
}

export fn activationBackwardHalf(x: [*]const u16, y: [*]const u16, g_out: [*]const u16, g_in: [*]u16, n: u32, kind: u32) callconv(.nvptx_kernel) void {
    activationBackwardOf(u16, x, y, g_out, g_in, n, kind);
}

inline fn addReluOf(comptime E: type, a: [*]const E, b: [*]const E, out: [*]E, n: u32) void {
    const t = globalIndex();
    if (t < n) bf16.store(E, out, t, @max(bf16.load(E, a, t) + bf16.load(E, b, t), 0));
}

inline fn addReluBackwardOf(comptime E: type, out: [*]const E, g: [*]const E, grad_a: [*]E, grad_b: [*]E, n: u32, flags: u32) void {
    const t = globalIndex();
    if (t >= n) return;
    const v = if (bf16.load(E, out, t) > 0) bf16.load(E, g, t) else 0;
    if (flags & 1 != 0) bf16.store(E, grad_a, t, bf16.load(E, grad_a, t) + v);
    if (flags & 2 != 0) bf16.store(E, grad_b, t, bf16.load(E, grad_b, t) + v);
}

export fn addReluQuad(a: [*]const u16, b: [*]const u16, out: [*]u16, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t * 4 < n) bf16.storeQuad(u16, out, t * 4, @max(bf16.loadQuad(u16, a, t * 4) + bf16.loadQuad(u16, b, t * 4), @as(bf16.Quad, @splat(0))));
}

export fn addReluBackwardQuad(out: [*]const u16, g: [*]const u16, grad_a: [*]u16, grad_b: [*]u16, n: u32, flags: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t * 4 >= n) return;
    const zero: bf16.Quad = @splat(0);
    const v = @select(f32, bf16.loadQuad(u16, out, t * 4) > zero, bf16.loadQuad(u16, g, t * 4), zero);
    if (flags & 1 != 0) bf16.storeQuad(u16, grad_a, t * 4, bf16.loadQuad(u16, grad_a, t * 4) + v);
    if (flags & 2 != 0) bf16.storeQuad(u16, grad_b, t * 4, bf16.loadQuad(u16, grad_b, t * 4) + v);
}

export fn addRelu(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) callconv(.nvptx_kernel) void {
    addReluOf(f32, a, b, out, n);
}

export fn addReluHalf(a: [*]const u16, b: [*]const u16, out: [*]u16, n: u32) callconv(.nvptx_kernel) void {
    addReluOf(u16, a, b, out, n);
}

export fn addReluBackward(out: [*]const f32, g: [*]const f32, grad_a: [*]f32, grad_b: [*]f32, n: u32, flags: u32) callconv(.nvptx_kernel) void {
    addReluBackwardOf(f32, out, g, grad_a, grad_b, n, flags);
}

export fn addReluBackwardHalf(out: [*]const u16, g: [*]const u16, grad_a: [*]u16, grad_b: [*]u16, n: u32, flags: u32) callconv(.nvptx_kernel) void {
    addReluBackwardOf(u16, out, g, grad_a, grad_b, n, flags);
}

export fn addInPlace(dst: [*]f32, src: [*]const f32, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) dst[t] += src[t];
}

export fn addInPlaceHalf(dst: [*]u16, src: [*]const u16, n: u32) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t < n) bf16.store(u16, dst, t, bf16.load(u16, dst, t) + bf16.load(u16, src, t));
}

export fn sumTo(in: [*]const f32, out: [*]f32, args: ga.SumArgs) callconv(.nvptx_kernel) void {
    sumToOf(f32, in, out, args);
}

export fn sumToHalf(in: [*]const u16, out: [*]u16, args: ga.SumArgs) callconv(.nvptx_kernel) void {
    sumToOf(u16, in, out, args);
}

inline fn sumToOf(comptime E: type, in: [*]const E, out: [*]E, args: ga.SumArgs) void {
    const t: usize = @workGroupId(0);
    const tid: usize = @workItemId(0);
    var d: [ga.max_rank]usize = undefined;
    var s: [ga.max_rank]usize = undefined;
    inline for (0..ga.max_rank) |i| {
        d[i] = args.dims[i];
        s[i] = args.strides[i];
    }
    const rmask = args.mask;

    var base: usize = 0;
    var rest: usize = t;
    inline for (0..ga.max_rank) |step| {
        const axis = ga.max_rank - 1 - step;
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
        inline for (0..ga.max_rank) |step| {
            const axis = ga.max_rank - 1 - step;
            if ((rmask >> axis) & 1 != 0) {
                at += (left % d[axis]) * s[axis];
                left /= d[axis];
            }
        }
        total += bf16.load(E, in, at);
    }
    const sum = blockSum(total, tid);
    if (tid == 0) bf16.store(E, out, t, if (args.accumulate != 0) bf16.load(E, out, t) + sum else sum);
}

// ***** column reductions and batch norm *****

const column = struct {
    var first: [threads]f32 addrspace(.shared) = undefined;
    var second: [threads]f32 addrspace(.shared) = undefined;
};

export fn columnReduce(args: ga.ColumnArgs) callconv(.nvptx_kernel) void {
    columnReduceOf(f32, args);
}

export fn columnReduceHalf(args: ga.ColumnArgs) callconv(.nvptx_kernel) void {
    columnReduceOf(u16, args);
}

inline fn columnReduceOf(comptime E: type, args: ga.ColumnArgs) void {
    const a: [*]const E = @ptrCast(args.a);
    const b: [*]const E = @ptrCast(args.b);
    const mask: [*]const E = @ptrCast(args.mask);
    const mean, const inv_std, const out = .{ args.mean, args.inv_std, args.out };
    const rows, const c, const row_stride, const kind, const scale = .{ args.rows, args.channels, args.row_stride, args.kind, args.scale };
    const tid: usize = @workItemId(0);
    const lane = tid % 32;
    const group = tid / 32;
    const ch = @as(usize, @workGroupId(1)) * 32 + lane;
    const valid = ch < c;
    var first: f32 = 0;
    var second: f32 = 0;
    if (valid) {
        // NOTE: kind 4 (layer norm) has one mean and inverse deviation per row instead of per channel, so they are read inside the loop.
        const m = if (kind == 2) bf16.load(E, a, ch) else if (kind == 4) 0 else mean[ch] * scale;
        const s = if (kind == 4) 0 else inv_std[ch];
        var r = @as(usize, @workGroupId(0)) * 8 + group;
        while (r < rows) : (r += row_stride) {
            const v = bf16.load(E, a, r * c + ch);
            if (kind == 0) {
                first += v;
            } else if (kind == 2) {
                first += v - m;
                second += (v - m) * (v - m);
            } else if (kind == 4) {
                first += v;
                second += v * ((bf16.load(E, b, r * c + ch) - mean[r]) * inv_std[r]);
            } else {
                const g = if (kind == 3 and bf16.load(E, mask, r * c + ch) <= 0) 0 else v;
                first += g;
                second += g * ((bf16.load(E, b, r * c + ch) - m) * s);
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
        if (args.ordered != 0) {
            const mine = args.partial + @as(usize, @workGroupId(0)) * 2 * c;
            mine[ch] = total_first;
            mine[c + ch] = total_second;
        } else {
            _ = @atomicRmw(f32, &out[ch], .Add, total_first, .monotonic);
            if (kind != 0) _ = @atomicRmw(f32, &out[c + ch], .Add, total_second, .monotonic);
        }
    }
}

export fn splitCombine(ws: [*]const f32, out: [*]f32, splits: u32, stride: u32, accumulate: u32) callconv(.nvptx_kernel) void {
    const i = globalIndex();
    if (i >= stride) return;
    var total: f32 = 0;
    for (0..splits) |z| total += ws[z * stride + i];
    out[i] = (if (accumulate != 0) out[i] else 0) + total;
}

export fn columnCombine(partial: [*]const f32, out: [*]f32, blocks: u32, width: u32, stride: u32) callconv(.nvptx_kernel) void {
    const i = globalIndex();
    if (i >= width) return;
    var total: f32 = 0;
    for (0..blocks) |b| total += partial[b * stride + i];
    out[i] += total;
}

export fn batchNormFinish(sums: [*]const f32, shift: [*]const f32, st: nm.Stats, l: nm.Layout) callconv(.nvptx_kernel) void {
    batchNormFinishOf(f32, sums, shift, st, l);
}

export fn batchNormFinishHalf(sums: [*]const f32, shift: [*]const u16, st: nm.Stats, l: nm.Layout) callconv(.nvptx_kernel) void {
    batchNormFinishOf(u16, sums, shift, st, l);
}

inline fn batchNormFinishOf(comptime E: type, sums: [*]const f32, shift: [*]const E, st: nm.Stats, l: nm.Layout) void {
    const mean, const inv_std, const running_mean, const running_var = .{ st.mean, st.inv_std, st.running_mean, st.running_var };
    const c = l.channels;
    const rows = l.len / c;
    const ch = globalIndex();
    if (ch >= c) return;
    if (l.flag == 0) {
        mean[ch] = running_mean[ch];
        inv_std[ch] = 1 / @sqrt(running_var[ch] + nm.eps);
        return;
    }
    const n: f32 = @floatFromInt(rows);
    const offset = sums[ch] / n;
    const mu = bf16.load(E, shift, ch) + offset;
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
export fn batchNormApply(x: [*]const f32, y: [*]f32, p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() < l.len) nm.apply(f32, globalIndex(), x, y, p, l);
}

export fn batchNormApplyQuad(x: [*]const u16, y: [*]u16, p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() * 4 < l.len) nm.applyQuad(u16, globalIndex() * 4, x, y, p, l);
}

export fn batchNormGradInputQuad(io: nm.GradIo(u16), p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() * 4 < l.len) nm.gradInputQuad(u16, globalIndex() * 4, io, p, l);
}

export fn batchNormApplyHalf(x: [*]const u16, y: [*]u16, p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() < l.len) nm.apply(u16, globalIndex(), x, y, p, l);
}

export fn batchNormGradInput(io: nm.GradIo(f32), p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() < l.len) nm.gradInput(f32, globalIndex(), io, p, l);
}

export fn batchNormGradInputHalf(io: nm.GradIo(u16), p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() < l.len) nm.gradInput(u16, globalIndex(), io, p, l);
}

// ***** losses *****

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

// ***** convolution helpers and pooling *****

export fn im2col(x: [*]const f32, col: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.im2col(@intCast(globalIndex()), x, col, d);
}

export fn col2im(dcol: [*]const f32, dx: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.col2im(@intCast(globalIndex()), dcol, dx, d);
}

export fn col2imHalf(dcol: [*]const u16, dx: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.col2imHalf(@intCast(globalIndex()), dcol, dx, d);
}

export fn groupedForward(x: [*]const f32, w: [*]const f32, out: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.groupedForward(@intCast(globalIndex()), x, w, out, d);
}

export fn groupedInputGrad(dy: [*]const f32, w: [*]const f32, dx: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.groupedInputGrad(@intCast(globalIndex()), dy, w, dx, d);
}

export fn groupedWeightGrad(x: [*]const f32, dy: [*]const f32, dw: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.groupedWeightGrad(@intCast(globalIndex()), x, dy, dw, d);
}

export fn maxPool(x: [*]const f32, out: [*]f32, index: [*]u32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.maxPool(@intCast(globalIndex()), x, out, index, d);
}

export fn avgPool(x: [*]const f32, out: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.avgPool(@intCast(globalIndex()), x, out, d);
}

export fn avgPoolBackward(gout: [*]const f32, gin: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.avgPoolBackward(@intCast(globalIndex()), gout, gin, d);
}

export fn maxPoolBackward(index: [*]const u32, gout: [*]const f32, gin: [*]f32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.maxPoolBackward(@intCast(globalIndex()), index, gout, gin, d);
}

export fn maxPoolHalf(x: [*]const u16, out: [*]u16, index: [*]u32, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.maxPoolHalf(@intCast(globalIndex()), x, out, index, d);
}

export fn avgPoolHalf(x: [*]const u16, out: [*]u16, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.avgPoolHalf(@intCast(globalIndex()), x, out, d);
}

export fn avgPoolBackwardHalf(gout: [*]const u16, gin: [*]u16, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.avgPoolBackwardHalf(@intCast(globalIndex()), gout, gin, d);
}

export fn maxPoolBackwardHalf(index: [*]const u32, gout: [*]const u16, gin: [*]u16, len: u32, d: cv.Dims) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) cv.maxPoolBackwardHalf(@intCast(globalIndex()), index, gout, gin, d);
}

export fn dropout(src: [*]const f32, dst: [*]f32, len: u32, seed: u64, keep: f32, accumulate: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) drop.applyOf(f32, globalIndex(), src, dst, seed, keep, accumulate != 0);
}

export fn dropoutHalf(src: [*]const u16, dst: [*]u16, len: u32, seed: u64, keep: f32, accumulate: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) drop.applyOf(u16, globalIndex(), src, dst, seed, keep, accumulate != 0);
}

// ***** optimizers and data *****

// NOTE: one launch updates every tensor of a model. The host passes a table of addresses, each block handles
// tensor_chunk elements of one tensor, and tableBlock finds which tensor that is.
const tensor_chunk = 1024;

const TensorBlock = struct { w: u64, g: u64, m: u64, v: u64, at: usize, n: usize };

const current_block = struct {
    var words: [6]u64 addrspace(.shared) = undefined;
};

fn pointer(comptime T: type, address: u64) T {
    return @ptrFromInt(address);
}

// Thread 0 looks up the tensor that this block works on and shares it through shared memory.
inline fn tableBlock(table: ga.TensorTable) TensorBlock {
    if (@workItemId(0) == 0) {
        const block: u32 = @workGroupId(0);
        inline for (0..ga.max_tensors) |i| {
            const last = i + 1 >= table.count or block < table.first_block[@min(i + 1, ga.max_tensors - 1)];
            if (i < table.count and block >= table.first_block[i] and last) {
                const offset = (block - table.first_block[i]) * tensor_chunk;
                current_block.words = .{ table.w[i], table.g[i], table.m[i], table.v[i], offset, @min(tensor_chunk, table.len[i] - offset) };
            }
        }
    }
    barrier();
    const w = current_block.words;
    barrier();
    return .{ .w = w[0], .g = w[1], .m = w[2], .v = w[3], .at = w[4], .n = w[5] };
}

export fn zeroGrads(table: ga.TensorTable) callconv(.nvptx_kernel) void {
    const b = tableBlock(table);
    var e: usize = @workItemId(0);
    while (e < b.n) : (e += threads) pointer([*]f32, b.g)[b.at + e] = 0;
}

export fn sgd(table: ga.TensorTable, lr: f32) callconv(.nvptx_kernel) void {
    const b = tableBlock(table);
    var e: usize = @workItemId(0);
    while (e < b.n) : (e += threads) pointer([*]f32, b.w)[b.at + e] -= lr * pointer([*]const f32, b.g)[b.at + e];
}

export fn ema(table: ga.TensorTable, decay: f32) callconv(.nvptx_kernel) void {
    const b = tableBlock(table);
    var e: usize = @workItemId(0);
    while (e < b.n) : (e += threads) {
        const shadow = pointer([*]f32, b.g) + b.at + e;
        shadow[0] = decay * shadow[0] + (1 - decay) * pointer([*]const f32, b.w)[b.at + e];
    }
}

export fn swapWeights(table: ga.TensorTable) callconv(.nvptx_kernel) void {
    const b = tableBlock(table);
    var e: usize = @workItemId(0);
    while (e < b.n) : (e += threads) {
        const live = pointer([*]f32, b.w) + b.at + e;
        const shadow = pointer([*]f32, b.g) + b.at + e;
        const kept = live[0];
        live[0] = shadow[0];
        shadow[0] = kept;
    }
}

export fn adam(table: ga.TensorTable, hp: ga.AdamArgs) callconv(.nvptx_kernel) void {
    const b = tableBlock(table);
    var e: usize = @workItemId(0);
    while (e < b.n) : (e += threads) {
        const w = pointer([*]f32, b.w) + b.at + e;
        const m = pointer([*]f32, b.m) + b.at + e;
        const v = pointer([*]f32, b.v) + b.at + e;
        const grad = pointer([*]const f32, b.g)[b.at + e];
        const m_new = hp.beta1 * m[0] + (1 - hp.beta1) * grad;
        const v_new = hp.beta2 * v[0] + (1 - hp.beta2) * grad * grad;
        m[0] = m_new;
        v[0] = v_new;
        w[0] -= hp.step * m_new / (@sqrt(v_new) * hp.inv_sqrt_c2 + hp.eps) + hp.decay * w[0];
    }
}
export fn gather(images: [*]const f32, labels_src: [*]const u8, order: [*]const u32, batch: [*]f32, labels_out: [*]u8, labels_other: [*]u8, args: ga.GatherArgs) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, args.rows) * args.pixels) return;
    const r = t / args.pixels;
    const p = t % args.pixels;
    const box: aug.Box = .{ .x0 = args.x0, .y0 = args.y0, .x1 = args.x1, .y1 = args.y1 };
    const partner = (r + 1) % args.rows;
    const side = aug.side(args.pixels, args.channels);
    const from = if (box.contains(p, side, args.channels)) partner else r;
    const idx: usize = order[args.start + from];
    const config: aug.Config = .{ .max_shift = args.max_shift, .max_angle = args.max_angle, .max_scale = args.max_scale, .flip = args.flip != 0, .mirror = args.mirror != 0, .cutout = args.cutout, .channels = args.channels, .seed = args.seed };
    const source = images + idx * args.pixels;
    batch[t] = if (config.active()) aug.pixel(source, p, side, args.channels, aug.transformOf(args.start + from, config)) else source[p];
    if (p == 0) {
        labels_out[r] = labels_src[order[args.start + r]];
        labels_other[r] = labels_src[order[args.start + partner]];
    }
}

// ***** tensor core GEMM *****

fn TensorTile(comptime tile_rows: usize, comptime tile_cols: usize) type {
    return struct {
        const rows = tile_rows;
        const cols = tile_cols;
        const warp_m = rows * cols / 256;
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

        inline fn aDestination(row: usize, chunk: usize) usize {
            return aSlot(row, chunk);
        }

        inline fn bDestination(k: usize, chunk: usize) usize {
            return a_elems + bSlot(k, chunk);
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
}

inline fn packHalfOf(comptime E: type, src: [*]const E, dst: [*]u16, args: ga.PackArgs) void {
    const t = globalIndex();
    if (t >= @as(usize, args.rows_padded) * args.cols_padded) return;
    const i = t / args.cols_padded;
    const j = t % args.cols_padded;
    const inside = i < args.rows and j < args.cols;
    const source = if (args.transpose != 0) j * args.src_stride + i else i * args.src_stride + j;
    dst[t] = if (inside) bf16.encode(bf16.load(E, src, source)) else 0;
}

export fn packHalf(src: [*]const f32, dst: [*]u16, args: ga.PackArgs) callconv(.nvptx_kernel) void {
    packHalfOf(f32, src, dst, args);
}

export fn packHalfCopy(src: [*]const u16, dst: [*]u16, args: ga.PackArgs) callconv(.nvptx_kernel) void {
    packHalfOf(u16, src, dst, args);
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
    packTransposedOf(f32, src, dst, args);
}

export fn transposeHalf(src: [*]const u16, dst: [*]u16, args: ga.PackArgs) callconv(.nvptx_kernel) void {
    packTransposedOf(u16, src, dst, args);
}

inline fn packTransposedOf(comptime E: type, src: [*]const E, dst: [*]u16, args: ga.PackArgs) void {
    const tid: usize = @workItemId(0);
    const lane = tid % 32;
    const group = tid / 32;
    const row_base = @as(usize, @workGroupId(1)) * 32;
    const col_base = @as(usize, @workGroupId(0)) * 32;
    inline for (0..4) |step| {
        const j = col_base + group + 8 * step;
        const i = row_base + lane;
        transpose_tile.values[(group + 8 * step) * 33 + lane] = if (i < args.rows and j < args.cols) bf16.load(E, src, j * args.src_stride + i) else 0;
    }
    barrier();
    inline for (0..4) |step| {
        const i = row_base + group + 8 * step;
        dst[i * args.cols_padded + col_base + lane] = bf16.encode(transpose_tile.values[lane * 33 + group + 8 * step]);
    }
}

// How the operands of a tensor core GEMM are read. dense reads both matrices as stored. gather_a builds the left
// operand from the NHWC input on the fly (implicit GEMM for the forward pass and the input gradient), and gather_b
// does the same for the right operand (weight gradient), so no im2col buffer is ever written.
const GemmMode = enum { dense, gather_a, gather_b };

const zero_geometry: ga.ConvGeometry = .{ .n = 0, .h = 0, .w = 0, .c = 0, .k = 0, .stride = 0, .pad = 0, .out_h = 0, .out_w = 0, .positions = 0, .c_out = 0 };

// Copies 16 bytes from global to shared memory without passing through registers. An invalid copy writes zeros, which pads the matrix edges.
inline fn copyAsyncZfill(dst: u32, src: anytype, valid: bool) void {
    asm volatile ("cp.async.cg.shared.global [%[dst]], [%[src]], 16, %[size];"
        :
        : [dst] "r" (dst),
          [src] "l" (@intFromPtr(src)),
          [size] "r" (@as(u32, if (valid) 16 else 0)),
        : .{ .memory = true });
}

inline fn commitCopies() void {
    asm volatile ("cp.async.commit_group;" ::: .{ .memory = true });
}

inline fn waitCopies() void {
    asm volatile ("cp.async.wait_group 1;" ::: .{ .memory = true });
}

const Acc = [4][4][4]f32;
const Warp = struct { lane: usize, row: usize, col: usize };
const Block = struct { tid: usize, row0: usize, col0: usize, k_begin: usize };

// Chooses a tensor core GEMM at compile time: bf16 or TF32 inputs, the block tile size and the operand mode.
const GemmKind = struct {
    tf32: bool = false,
    rows: usize = 128,
    cols: usize = 64,
    mode: GemmMode,

    fn Elem(comptime self: GemmKind) type {
        return if (self.tf32) f32 else u16;
    }

    fn Tile(comptime self: GemmKind) type {
        return if (self.tf32) Tf32Tile(self.rows, self.cols) else TensorTile(self.rows, self.cols);
    }
};

fn GemmLoader(comptime kind: GemmKind) type {
    const tc = kind.Tile();
    const Elem = kind.Elem();
    const mode = kind.mode;
    const a_per = tc.rows / 64;
    const b_per = tc.cols / 64;
    const lane_elems = 16 / @sizeOf(Elem);

    return struct {
        a: [*]const Elem,
        b: [*]const Elem,
        geo: ga.ConvGeometry,
        rows_a: [a_per]usize,
        chunk_a: usize,
        dst_a: [a_per]usize,
        dst_b: [b_per]usize,
        k_b: [b_per]usize,
        chunk_b: [b_per]usize,
        first_col: usize,
        depth_k: usize,
        stride_n: usize,
        row_base: usize,
        rows_total: usize,
        origin_n: [a_per]u32,
        origin_y: [a_per]i32,
        origin_x: [a_per]i32,
        origin_valid: [a_per]bool,
        cursor_ky: i32,
        cursor_kx: i32,
        cursor_channel: usize,

        inline fn init(a: [*]const Elem, b: [*]const Elem, geo: ga.ConvGeometry, shape: ga.TensorShape, block: Block) @This() {
            const tid, const row0, const col0, const k_begin = .{ block.tid, block.row0, block.col0, block.k_begin };
            var self: @This() = undefined;
            self.a = a;
            self.b = b;
            self.geo = geo;
            self.chunk_a = tid % 4;
            self.first_col = col0;
            self.depth_k = shape.k_padded;
            self.stride_n = shape.n_padded;
            self.row_base = row0;
            self.rows_total = shape.m;
            self.cursor_ky = 0;
            self.cursor_kx = 0;
            self.cursor_channel = 0;
            inline for (0..a_per) |i| {
                self.rows_a[i] = tid / 4 + 64 * i;
                self.dst_a[i] = tc.aDestination(self.rows_a[i], self.chunk_a);
            }
            inline for (0..b_per) |j| {
                self.k_b[j] = (tid + j * 256) / (tc.cols / lane_elems);
                self.chunk_b[j] = (tid + j * 256) % (tc.cols / lane_elems);
                self.dst_b[j] = tc.bDestination(self.k_b[j], self.chunk_b[j]);
            }
            if (mode == .gather_a) {
                inline for (0..a_per) |i| {
                    const row: u32 = @intCast(row0 + self.rows_a[i]);
                    self.origin_valid[i] = row < shape.m;
                    self.origin_n[i] = row / (geo.out_h * geo.out_w);
                    self.origin_y[i] = @as(i32, @intCast(row / geo.out_w % geo.out_h * geo.stride)) - @as(i32, @intCast(geo.pad));
                    self.origin_x[i] = @as(i32, @intCast(row % geo.out_w * geo.stride)) - @as(i32, @intCast(geo.pad));
                }
                const tap: u32 = @intCast(k_begin / geo.c);
                self.cursor_ky = @intCast(tap / geo.k);
                self.cursor_kx = @intCast(tap % geo.k);
                self.cursor_channel = k_begin % geo.c;
            }
            return self;
        }

        inline fn issue(self: *@This(), stage: usize, kk: usize) void {
            const base = stage * tc.stage_elems;
            const geometry = self.geo;
            if (mode == .gather_a) {
                inline for (0..a_per) |i| {
                    const y = self.origin_y[i] + self.cursor_ky;
                    const x = self.origin_x[i] + self.cursor_kx;
                    const valid = self.origin_valid[i] and y >= 0 and y < @as(i32, @intCast(geometry.h)) and x >= 0 and x < @as(i32, @intCast(geometry.w));
                    const at = if (valid) ((@as(usize, self.origin_n[i]) * geometry.h + @as(usize, @intCast(y))) * geometry.w + @as(usize, @intCast(x))) * geometry.c + self.cursor_channel + self.chunk_a * lane_elems else 0;
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
                inline for (0..a_per) |i| {
                    const valid = self.row_base + self.rows_a[i] < self.rows_total;
                    const at = if (valid) (self.row_base + self.rows_a[i]) * self.depth_k + self.chunk_a * lane_elems + kk else 0;
                    copyAsyncZfill(tc.sharedAddress(base + self.dst_a[i]), self.a + at, valid);
                }
            }
            inline for (0..b_per) |j| {
                const col = self.first_col + self.chunk_b[j] * lane_elems;
                if (mode == .gather_b) {
                    const tap: u32 = @intCast(col / geometry.c);
                    const channel: usize = col % geometry.c;
                    const ky: i32 = @intCast(tap / geometry.k);
                    const kx: i32 = @intCast(tap % geometry.k);
                    const p: u32 = @intCast(kk + self.k_b[j]);
                    const y = @as(i32, @intCast(p / geometry.out_w % geometry.out_h * geometry.stride)) + ky - @as(i32, @intCast(geometry.pad));
                    const x = @as(i32, @intCast(p % geometry.out_w * geometry.stride)) + kx - @as(i32, @intCast(geometry.pad));
                    const valid = p < geometry.positions and y >= 0 and y < @as(i32, @intCast(geometry.h)) and x >= 0 and x < @as(i32, @intCast(geometry.w));
                    const at = if (valid) ((@as(usize, p / (geometry.out_h * geometry.out_w)) * geometry.h + @as(usize, @intCast(y))) * geometry.w + @as(usize, @intCast(x))) * geometry.c + channel else 0;
                    copyAsyncZfill(tc.sharedAddress(base + self.dst_b[j]), self.b + at, valid);
                } else {
                    const valid = col < self.stride_n;
                    const at = if (valid) (kk + self.k_b[j]) * self.stride_n + col else 0;
                    copyAsyncZfill(tc.sharedAddress(base + self.dst_b[j]), self.b + at, valid);
                }
            }
        }
    };
}

// Overlaps loading and computing: the copies for the next stages are in flight while the current tile is multiplied.
inline fn pipeline(comptime kind: GemmKind, loader: anytype, block: Block, tiles: usize, warp: Warp, acc: *Acc) void {
    const tc = kind.Tile();
    inline for (0..tc.stages - 1) |stage| {
        if (stage < tiles) loader.issue(stage, block.k_begin + stage * tc.depth);
        commitCopies();
    }
    var tile: usize = 0;
    while (tile < tiles) : (tile += 1) {
        waitCopies();
        barrier();
        const ahead = tile + tc.stages - 1;
        if (ahead < tiles) loader.issue(ahead % tc.stages, block.k_begin + ahead * tc.depth);
        commitCopies();
        const base = (tile % tc.stages) * tc.stage_elems;
        if (kind.tf32) tf32Compute(tc, warp, base, acc) else bf16Compute(tc, warp, base, acc);
    }
}

// Writes one accumulator tile. Depending on the shape flags the result is rounded to bf16, stored in its own slice of a
// split-K workspace (deterministic mode), added atomically (split-K), or stored or accumulated directly.
inline fn storeTile(comptime kind: GemmKind, c: [*]f32, acc: *const Acc, shape: ga.TensorShape, block: Block, warp: Warp) void {
    const tc = kind.Tile();
    const half_output = !kind.tf32;
    inline for (0..tc.warp_m / 16) |mi| {
        inline for (0..4) |ni| {
            inline for (0..4) |q| {
                const r = block.row0 + warp.row * tc.warp_m + mi * 16 + warp.lane / 4 + (q / 2) * 8;
                const col = block.col0 + warp.col * 32 + ni * 8 + (warp.lane % 4) * 2 + q % 2;
                if (r < shape.m and col < shape.n) {
                    const at = if (shape.transpose_out != 0) col * shape.m + r else r * shape.n + col;
                    if (half_output and shape.output_half != 0) {
                        const half_out: [*]u16 = @ptrCast(@alignCast(c));
                        half_out[at] = bf16.encode((if (shape.accumulate != 0) bf16.decode(half_out[at]) else 0) + acc[mi][ni][q]);
                    } else if (shape.split_stride != 0) {
                        c[@as(usize, @workGroupId(2)) * shape.split_stride + at] = acc[mi][ni][q];
                    } else if (shape.chunk < shape.k_padded) {
                        _ = @atomicRmw(f32, &c[at], .Add, acc[mi][ni][q], .monotonic);
                    } else {
                        c[at] = if (shape.accumulate != 0) c[at] + acc[mi][ni][q] else acc[mi][ni][q];
                    }
                }
            }
        }
    }
}

// One thread block computes one tile of C. Work group x and y pick the tile and work group z picks the range of K for split-K.
inline fn tileGemm(comptime kind: GemmKind, a: [*]const kind.Elem(), b: [*]const kind.Elem(), c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) void {
    const tc = kind.Tile();
    const tid: usize = @workItemId(0);
    const warp: Warp = .{ .lane = tid % 32, .row = tid / 32 / (tc.cols / 32), .col = tid / 32 % (tc.cols / 32) };
    const block: Block = .{ .tid = tid, .row0 = @as(usize, @workGroupId(1)) * tc.rows, .col0 = @as(usize, @workGroupId(0)) * tc.cols, .k_begin = @as(usize, @workGroupId(2)) * shape.chunk };
    const tiles = (@min(shape.k_padded, block.k_begin + shape.chunk) -| block.k_begin) / tc.depth;

    var loader = GemmLoader(kind).init(a, b, geo, shape, block);
    var acc: Acc = @splat(@splat(@splat(0)));
    pipeline(kind, &loader, block, tiles, warp, &acc);
    storeTile(kind, c, &acc, shape, block, warp);
}
inline fn bf16Compute(comptime tc: type, warp: Warp, base: usize, acc: *Acc) void {
    inline for (0..tc.depth / 16) |step| {
        var a_frag: [4][4]u32 = undefined;
        inline for (0..tc.warp_m / 16) |mi| {
            const row = warp.row * tc.warp_m + mi * 16 + warp.lane % 8 + 8 * ((warp.lane / 8) % 2);
            a_frag[mi] = tc.ldmatrix(tc.sharedAddress(base + tc.aSlot(row, step * 2 + warp.lane / 16)));
        }
        inline for (0..2) |pair| {
            const k = step * 16 + warp.lane % 8 + 8 * ((warp.lane / 8) % 2);
            const frag = tc.ldmatrixTransposed(tc.sharedAddress(base + tc.a_elems + tc.bSlot(k, warp.col * 4 + pair * 2 + warp.lane / 16)));
            inline for (0..tc.warp_m / 16) |mi| {
                tc.mma(&acc[mi][pair * 2], a_frag[mi], .{ frag[0], frag[1] });
                tc.mma(&acc[mi][pair * 2 + 1], a_frag[mi], .{ frag[2], frag[3] });
            }
        }
    }
}

inline fn tf32Compute(comptime tc: type, warp: Warp, base: usize, acc: *Acc) void {
    inline for (0..tc.depth / 8) |step| {
        const k0 = step * 8 + warp.lane % 4;
        var a_frag: [4][4]u32 = undefined;
        inline for (0..tc.warp_m / 16) |mi| {
            const r = warp.row * tc.warp_m + mi * 16 + warp.lane / 4;
            a_frag[mi] = .{
                tc.rounded(tc.smem[base + r * tc.a_pitch + k0]),
                tc.rounded(tc.smem[base + (r + 8) * tc.a_pitch + k0]),
                tc.rounded(tc.smem[base + r * tc.a_pitch + k0 + 4]),
                tc.rounded(tc.smem[base + (r + 8) * tc.a_pitch + k0 + 4]),
            };
        }
        inline for (0..4) |ni| {
            const n = warp.col * 32 + ni * 8 + warp.lane / 4;
            const b_frag: [2]u32 = .{
                tc.rounded(tc.smem[base + tc.a_elems + k0 * tc.b_pitch + n]),
                tc.rounded(tc.smem[base + tc.a_elems + (k0 + 4) * tc.b_pitch + n]),
            };
            inline for (0..tc.warp_m / 16) |mi| tc.mma(&acc[mi][ni], a_frag[mi], b_frag);
        }
    }
}

export fn sgemmTensor(a: [*]const u16, b: [*]const u16, c: [*]f32, shape: ga.TensorShape) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .dense }, a, b, c, shape, zero_geometry);
}

export fn convGemmForward(x: [*]const u16, w: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .gather_a }, x, w, c, shape, geo);
}

export fn convGemmWeightGrad(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .gather_b }, dy_t, x, c, shape, geo);
}

export fn convGemmForwardWide(x: [*]const u16, w: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 128, .cols = 128, .mode = .gather_a }, x, w, c, shape, geo);
}

export fn convGemmWeightGradWide(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 128, .cols = 128, .mode = .gather_b }, dy_t, x, c, shape, geo);
}

export fn convGemmWeightGradShort(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 64, .cols = 128, .mode = .gather_b }, dy_t, x, c, shape, geo);
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

export fn weightsToKrsc(src: [*]const f32, dst: [*]f32, args: ga.LayoutArgs) callconv(.nvptx_kernel) void {
    const t: u32 = @intCast(globalIndex());
    const total = args.k * args.r * args.r * args.c;
    if (t >= total) return;
    const ci = t % args.c;
    const kx = t / args.c % args.r;
    const ky = t / args.c / args.r % args.r;
    const o = t / args.c / args.r / args.r;
    dst[t] = src[((ky * args.r + kx) * args.c + ci) * args.k + o];
}

export fn addKrscGradient(src: [*]const f32, dst: [*]f32, args: ga.LayoutArgs) callconv(.nvptx_kernel) void {
    const t: u32 = @intCast(globalIndex());
    const total = args.k * args.r * args.r * args.c;
    if (t >= total) return;
    const ci = t % args.c;
    const kx = t / args.c % args.r;
    const ky = t / args.c / args.r % args.r;
    const o = t / args.c / args.r / args.r;
    dst[((ky * args.r + kx) * args.c + ci) * args.k + o] += src[t];
}

fn Tf32Tile(comptime tile_rows: usize, comptime tile_cols: usize) type {
    return struct {
        const rows = tile_rows;
        const cols = tile_cols;
        const warp_m = rows * cols / 256;
        const depth = 16;
        const stages = 3;
        const a_pitch = depth + 4;
        const b_pitch = cols + 8;
        const a_elems = rows * a_pitch;
        const b_elems = depth * b_pitch;
        const stage_elems = a_elems + b_elems;

        var smem: [stages * stage_elems]f32 align(128) addrspace(.shared) = undefined;

        inline fn sharedAddress(index: usize) u32 {
            return @truncate(@intFromPtr(&smem[index]));
        }

        inline fn aDestination(row: usize, chunk: usize) usize {
            return row * a_pitch + chunk * 4;
        }

        inline fn bDestination(k: usize, chunk: usize) usize {
            return a_elems + k * b_pitch + chunk * 4;
        }

        inline fn rounded(x: f32) u32 {
            return asm ("cvt.rna.tf32.f32 %[out], %[in];"
                : [out] "=r" (-> u32),
                : [in] "f" (x),
            );
        }

        inline fn mma(acc: *[4]f32, a: [4]u32, b: [2]u32) void {
            var d0 = acc[0];
            var d1 = acc[1];
            var d2 = acc[2];
            var d3 = acc[3];
            asm volatile ("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%[d0], %[d1], %[d2], %[d3]}, {%[a0], %[a1], %[a2], %[a3]}, {%[b0], %[b1]}, {%[d0], %[d1], %[d2], %[d3]};"
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
}

export fn convTf32Forward(x: [*]const f32, w: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .mode = .gather_a }, x, w, c, shape, geo);
}

export fn convTf32WeightGrad(dy_t: [*]const f32, x: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .mode = .gather_b }, dy_t, x, c, shape, geo);
}

export fn convTf32WeightGradShort(dy_t: [*]const f32, x: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .rows = 64, .cols = 128, .mode = .gather_b }, dy_t, x, c, shape, geo);
}

export fn rotateWeightsF32(src: [*]const f32, dst: [*]f32, args: ga.RotateArgs) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, args.rows_padded) * args.cols_padded) return;
    const i: u32 = @intCast(t / args.cols_padded);
    const j: u32 = @intCast(t % args.cols_padded);
    const tap = i / args.c_out;
    const co = i % args.c_out;
    const source_tap = (args.k - 1 - tap / args.k) * args.k + (args.k - 1 - tap % args.k);
    dst[t] = if (j < args.c_in) src[(source_tap * args.c_in + j) * args.c_out + co] else 0;
}

export fn transposeF32(src: [*]const f32, dst: [*]f32, args: ga.PackArgs) callconv(.nvptx_kernel) void {
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
        dst[i * args.cols_padded + col_base + lane] = transpose_tile.values[lane * 33 + group + 8 * step];
    }
}
