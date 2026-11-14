const std = @import("std");
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

// Adds a value across the 32 lanes of a warp; every lane gets the total. All lanes of the warp must call it together.
inline fn warpSum(value: f32) f32 {
    var total = value;
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        const other = asm volatile ("shfl.sync.bfly.b32 %[out], %[in], %[offset], 0x1f, 0xffffffff;"
            : [out] "=r" (-> u32),
            : [in] "r" (@as(u32, @bitCast(total))),
              [offset] "r" (@as(u32, offset)),
        );
        total += @as(f32, @bitCast(other));
    }
    return total;
}

// Adds value across the lanes that are offset apart; every lane gets the same total. All lanes of the warp must call it together.
inline fn shflXor(value: f32, comptime offset: u32) f32 {
    const other = asm volatile ("shfl.sync.bfly.b32 %[out], %[in], %[offset], 0x1f, 0xffffffff;"
        : [out] "=r" (-> u32),
        : [in] "r" (@as(u32, @bitCast(value))),
          [offset] "r" (@as(u32, offset)),
    );
    return value + @as(f32, @bitCast(other));
}

// The largest value across the 32 lanes of a warp; every lane gets it. All lanes of the warp must call it together.
inline fn warpMax(value: f32) f32 {
    var best = value;
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        const other = asm volatile ("shfl.sync.bfly.b32 %[out], %[in], %[offset], 0x1f, 0xffffffff;"
            : [out] "=r" (-> u32),
            : [in] "r" (@as(u32, @bitCast(best))),
              [offset] "r" (@as(u32, offset)),
        );
        best = @max(best, @as(f32, @bitCast(other)));
    }
    return best;
}

// Writes 16 bytes of bf16 quiet NaNs to shared memory. A gathered copy that falls outside the input taps uses this
// instead of a zero-fill when the operand is transformed on load: the transform maps NaN to exactly 0, while a
// zero-filled value would wrongly come out of relu(scale * 0 + shift) as max(shift, 0).
inline fn storeMagicShared(dst: u32) void {
    asm volatile ("st.shared.v4.b32 [%[dst]], {%[m], %[m], %[m], %[m]};"
        :
        : [dst] "r" (dst),
          [m] "r" (@as(u32, 0x7fc07fc0)),
        : .{ .memory = true });
}

// The row a warp works on when a kernel runs one warp per row (eight rows per block of 256 threads), and the lane within it.
inline fn warpRow() struct { row: usize, lane: usize } {
    const tid: usize = @workItemId(0);
    return .{ .row = @as(usize, @workGroupId(0)) * 8 + tid / 32, .lane = tid % 32 };
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

// Exports a kernel for f32 under name and for bf16 (u16) under name plus Half; make takes the element type and returns a struct with the entry run.
fn exportPair(comptime name: []const u8, comptime make: fn (comptime type) type) void {
    @export(&make(f32).run, .{ .name = name });
    @export(&make(u16).run, .{ .name = name ++ "Half" });
}

comptime {
    @setEvalBranchQuota(100000);
    exportPair("elementwise", elementwiseKernel);
    exportPair("relu", reluKernel);
    exportPair("reluBackward", reluBackwardKernel);
    exportPair("layerNormRows", layerNormRowsKernel);
    exportPair("layerNormBackward", layerNormBackwardKernel);
    exportPair("embeddingGather", embeddingGatherKernel);
    exportPair("embeddingScatter", embeddingScatterKernel);
    exportPair("softmaxRows", softmaxRowsKernel);
    exportPair("softmaxBackward", softmaxBackwardKernel);
    exportPair("activation", activationKernel);
    exportPair("activationBackward", activationBackwardKernel);
    exportPair("pointwise", pointwiseKernel);
    exportPair("addRelu", addReluKernel);
    exportPair("addReluBackward", addReluBackwardKernel);
    exportPair("addInPlace", addInPlaceKernel);
    exportPair("sumTo", sumToKernel);
    exportPair("columnReduce", columnReduceKernel);
    @export(&ColumnReduce(u16, 2).run, .{ .name = "columnReduceHalfPair" });
    exportPair("batchNormFinish", batchNormFinishKernel);
    exportGemm("sgemmNN", 128, 128, false, false);
    exportGemm("sgemmNT", 128, 128, false, true);
    exportGemm("sgemmTN", 128, 128, true, false);
    exportGemm("sgemmTallNN", 128, 32, false, false);
    exportGemm("sgemmTallNT", 128, 32, false, true);
    exportGemm("sgemmTallTN", 128, 32, true, false);
    exportGemm("sgemmSmallNN", 32, 32, false, false);
    exportGemm("sgemmSmallNT", 32, 32, false, true);
    exportGemm("sgemmSmallTN", 32, 32, true, false);
    exportGemm("sgemmMidNN", 128, 64, false, false);
    exportGemm("sgemmMidNT", 128, 64, false, true);
    exportGemm("sgemmMidTN", 128, 64, true, false);
}
// ***** elementwise and reductions *****

fn elementwiseKernel(comptime E: type) type {
    return struct {
        fn run(a: [*]const E, b: [*]const E, out: [*]E, args: ga.ElementwiseArgs) callconv(.nvptx_kernel) void {
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
    };
}

fn reluKernel(comptime E: type) type {
    return struct {
        fn run(x: [*]const E, out: [*]E, n: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < n) bf16.store(E, out, t, @max(bf16.load(E, x, t), 0));
        }
    };
}

fn reluBackwardKernel(comptime E: type) type {
    return struct {
        fn run(x: [*]const E, g_out: [*]const E, g_in: [*]E, n: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < n) bf16.store(E, g_in, t, bf16.load(E, g_in, t) + if (bf16.load(E, x, t) > 0) bf16.load(E, g_out, t) else 0);
        }
    };
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

// NOTE: layer norm uses one warp per row (eight rows per block), the lanes striding over the columns and meeting in warpSum,
// because one thread per row left most of the device idle. The sums run in another order than the CPU loop, so results agree only to rounding.
fn layerNormRowsKernel(comptime E: type) type {
    return struct {
        fn run(x: [*]const E, y: [*]E, gamma: [*]const f32, beta: [*]const f32, mean: [*]f32, inv_std: [*]f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
            const tid: usize = @workItemId(0);
            const lane = tid % 32;
            const r = @as(usize, @workGroupId(0)) * 8 + tid / 32;
            if (r >= rows) return;
            const xr = x + r * cols;
            var sum: f32 = 0;
            var j = lane;
            while (j < cols) : (j += 32) sum += bf16.load(E, xr, j);
            const mu = warpSum(sum) / @as(f32, @floatFromInt(cols));
            var square: f32 = 0;
            j = lane;
            while (j < cols) : (j += 32) square += (bf16.load(E, xr, j) - mu) * (bf16.load(E, xr, j) - mu);
            const inverse = 1 / @sqrt(warpSum(square) / @as(f32, @floatFromInt(cols)) + ln.eps);
            j = lane;
            while (j < cols) : (j += 32) bf16.store(E, y + r * cols, j, gamma[j] * (bf16.load(E, xr, j) - mu) * inverse + beta[j]);
            if (lane == 0) {
                mean[r] = mu;
                inv_std[r] = inverse;
            }
        }
    };
}
fn layerNormBackwardKernel(comptime E: type) type {
    return struct {
        fn run(x: [*]const E, dy: [*]const E, dx: [*]E, gamma: [*]const f32, mean: [*]const f32, inv_std: [*]const f32, rows: u32, cols: u32) callconv(.nvptx_kernel) void {
            const tid: usize = @workItemId(0);
            const lane = tid % 32;
            const r = @as(usize, @workGroupId(0)) * 8 + tid / 32;
            if (r >= rows) return;
            const xr = x + r * cols;
            const dyr = dy + r * cols;
            const mu = mean[r];
            const inverse = inv_std[r];
            var sum_g: f32 = 0;
            var sum_gx: f32 = 0;
            var j = lane;
            while (j < cols) : (j += 32) {
                const g = bf16.load(E, dyr, j) * gamma[j];
                sum_g += g;
                sum_gx += g * (bf16.load(E, xr, j) - mu) * inverse;
            }
            sum_g = warpSum(sum_g);
            sum_gx = warpSum(sum_gx);
            const n: f32 = @floatFromInt(cols);
            j = lane;
            while (j < cols) : (j += 32) {
                const g = bf16.load(E, dyr, j) * gamma[j];
                bf16.store(E, dx + r * cols, j, bf16.load(E, dx + r * cols, j) + inverse * (g - sum_g / n - (bf16.load(E, xr, j) - mu) * inverse * sum_gx / n));
            }
        }
    };
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
fn embeddingGatherKernel(comptime E: type) type {
    return struct {
        fn run(table: [*]const f32, ids: [*]const u32, out: [*]E, n: u32, width: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < @as(usize, n) * width) bf16.store(E, out, t, table[@as(usize, ids[t / width]) * width + t % width]);
        }
    };
}

fn embeddingScatterKernel(comptime E: type) type {
    return struct {
        fn run(dy: [*]const E, ids: [*]const u32, dtable: [*]f32, n: u32, width: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < @as(usize, n) * width) _ = @atomicRmw(f32, &dtable[@as(usize, ids[t / width]) * width + t % width], .Add, bf16.load(E, dy, t), .monotonic);
        }
    };
}

// NOTE: one warp per row, like layer norm. The masked columns past the causal limit are written as zero.
fn softmaxRowsKernel(comptime E: type) type {
    return struct {
        fn run(x: [*]const E, y: [*]E, rows: u32, cols: u32, period: u32, scale: f32) callconv(.nvptx_kernel) void {
            const at = warpRow();
            if (at.row >= rows) return;
            const xr = x + at.row * cols;
            const yr = y + at.row * cols;
            const limit = soft.limitOf(at.row, cols, period);
            var largest: f32 = -std.math.inf(f32);
            var j = at.lane;
            while (j < limit) : (j += 32) largest = @max(largest, bf16.load(E, xr, j));
            largest = warpMax(largest);
            var total: f32 = 0;
            j = at.lane;
            while (j < limit) : (j += 32) total += act.exponential(scale * (bf16.load(E, xr, j) - largest));
            const inverse = 1 / warpSum(total);
            j = at.lane;
            while (j < cols) : (j += 32) bf16.store(E, yr, j, if (j < limit) act.exponential(scale * (bf16.load(E, xr, j) - largest)) * inverse else 0);
        }
    };
}
fn softmaxBackwardKernel(comptime E: type) type {
    return struct {
        fn run(y: [*]const E, dy: [*]const E, dx: [*]E, rows: u32, cols: u32, scale: f32) callconv(.nvptx_kernel) void {
            const at = warpRow();
            if (at.row >= rows) return;
            const base = at.row * cols;
            var dot: f32 = 0;
            var j = at.lane;
            while (j < cols) : (j += 32) dot += bf16.load(E, y + base, j) * bf16.load(E, dy + base, j);
            dot = warpSum(dot);
            j = at.lane;
            while (j < cols) : (j += 32) bf16.store(E, dx + base, j, bf16.load(E, dx + base, j) + scale * bf16.load(E, y + base, j) * (bf16.load(E, dy + base, j) - dot));
        }
    };
}
fn activationKernel(comptime E: type) type {
    return struct {
        fn run(x: [*]const E, out: [*]E, n: u32, kind: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < n) bf16.store(E, out, t, act.forward(@fromBackingInt(@intCast(kind)), bf16.load(E, x, t)));
        }
    };
}

fn pointwiseKernel(comptime E: type) type {
    return struct {
        fn run(x: [*]const E, out: [*]E, n: u32, kind: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < n) bf16.store(E, out, t, act.pointwise(@fromBackingInt(@intCast(kind)), bf16.load(E, x, t)));
        }
    };
}

fn activationBackwardKernel(comptime E: type) type {
    return struct {
        fn run(x: [*]const E, y: [*]const E, g_out: [*]const E, g_in: [*]E, n: u32, kind: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < n) bf16.store(E, g_in, t, bf16.load(E, g_in, t) + bf16.load(E, g_out, t) * act.derivative(@fromBackingInt(@intCast(kind)), bf16.load(E, x, t), bf16.load(E, y, t)));
        }
    };
}

fn addReluKernel(comptime E: type) type {
    return struct {
        fn run(a: [*]const E, b: [*]const E, out: [*]E, n: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < n) bf16.store(E, out, t, @max(bf16.load(E, a, t) + bf16.load(E, b, t), 0));
        }
    };
}

fn addReluBackwardKernel(comptime E: type) type {
    return struct {
        fn run(out: [*]const E, g: [*]const E, grad_a: [*]E, grad_b: [*]E, n: u32, flags: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t >= n) return;
            const v = if (bf16.load(E, out, t) > 0) bf16.load(E, g, t) else 0;
            if (flags & 1 != 0) bf16.store(E, grad_a, t, bf16.load(E, grad_a, t) + v);
            if (flags & 2 != 0) bf16.store(E, grad_b, t, bf16.load(E, grad_b, t) + v);
        }
    };
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

fn addInPlaceKernel(comptime E: type) type {
    return struct {
        fn run(dst: [*]E, src: [*]const E, n: u32) callconv(.nvptx_kernel) void {
            const t = globalIndex();
            if (t < n) bf16.store(E, dst, t, bf16.load(E, dst, t) + bf16.load(E, src, t));
        }
    };
}

fn sumToKernel(comptime E: type) type {
    return struct {
        fn run(in: [*]const E, out: [*]E, args: ga.SumArgs) callconv(.nvptx_kernel) void {
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
    };
}

// ***** column reductions and batch norm *****

const column = struct {
    var first: [2 * threads]f32 addrspace(.shared) = undefined;
    var second: [2 * threads]f32 addrspace(.shared) = undefined;
};

fn columnReduceKernel(comptime E: type) type {
    return ColumnReduce(E, 1);
}

// Each lane reduces width adjacent channels. With width 2 a bf16 row segment is read as one 4-byte word per lane, so a warp
// reads 128 contiguous bytes per row instead of 64. Every channel still adds its rows in the same order, so the result is
// bitwise the same as with width 1; the host uses width 2 only for an even channel count.
fn ColumnReduce(comptime E: type, comptime width: usize) type {
    return struct {
        const V = @Vector(width, f32);

        inline fn loadChannels(p: [*]const E, i: usize) V {
            if (width == 1) return @splat(bf16.load(E, p, i));
            if (E == u16) {
                const raw: *align(2 * width) const [width]u16 = @ptrCast(@alignCast(p + i));
                var out: V = undefined;
                inline for (0..width) |j| out[j] = bf16.decode(raw[j]);
                return out;
            }
            const raw: *align(4 * width) const [width]f32 = @ptrCast(@alignCast(p + i));
            return raw.*;
        }

        inline fn loadPerChannel(p: [*]const f32, i: usize) V {
            if (width == 1) return @splat(p[i]);
            const raw: *align(4 * width) const [width]f32 = @ptrCast(@alignCast(p + i));
            return raw.*;
        }

        fn run(args: ga.ColumnArgs) callconv(.nvptx_kernel) void {
            const a: [*]const E = @ptrCast(args.a);
            const b: [*]const E = @ptrCast(args.b);
            const mask: [*]const E = @ptrCast(args.mask);
            const mean, const inv_std, const out = .{ args.mean, args.inv_std, args.out };
            const rows, const c, const row_stride, const kind, const scale = .{ args.rows, args.channels, args.row_stride, args.kind, args.scale };
            const zero: V = @splat(0);
            const tid: usize = @workItemId(0);
            const lane = tid % 32;
            const group = tid / 32;
            const ch = @as(usize, @workGroupId(1)) * 32 * width + lane * width;
            const valid = ch < c;
            var first: V = zero;
            var second: V = zero;
            if (valid) {
                // NOTE: kind 4 (layer norm) has one mean and inverse deviation per row instead of per channel, so they are read inside the loop.
                const m: V = if (kind == 2) loadChannels(a, ch) else if (kind == 4) zero else loadPerChannel(mean, ch) * @as(V, @splat(scale));
                const s: V = if (kind == 4) zero else loadPerChannel(inv_std, ch);
                var r = @as(usize, @workGroupId(0)) * 8 + group;
                while (r < rows) : (r += row_stride) {
                    const v = loadChannels(a, r * c + ch);
                    if (kind == 0) {
                        first += v;
                    } else if (kind == 2) {
                        first += v - m;
                        second += (v - m) * (v - m);
                    } else if (kind == 4) {
                        first += v;
                        second += v * ((loadChannels(b, r * c + ch) - @as(V, @splat(mean[r]))) * @as(V, @splat(inv_std[r])));
                    } else {
                        const g = if (kind == 3) @select(f32, loadChannels(mask, r * c + ch) <= zero, zero, v) else v;
                        first += g;
                        second += g * ((loadChannels(b, r * c + ch) - m) * s);
                    }
                }
            }
            inline for (0..width) |j| {
                column.first[tid * width + j] = first[j];
                column.second[tid * width + j] = second[j];
            }
            barrier();
            if (group == 0 and valid) {
                inline for (0..width) |j| {
                    var total_first: f32 = 0;
                    var total_second: f32 = 0;
                    inline for (0..8) |g| {
                        total_first += column.first[(lane + 32 * g) * width + j];
                        total_second += column.second[(lane + 32 * g) * width + j];
                    }
                    if (args.ordered != 0) {
                        const mine = args.partial + @as(usize, @workGroupId(0)) * 2 * c;
                        mine[ch + j] = total_first;
                        mine[c + ch + j] = total_second;
                    } else {
                        _ = @atomicRmw(f32, &out[ch + j], .Add, total_first, .monotonic);
                        if (kind != 0) _ = @atomicRmw(f32, &out[c + ch + j], .Add, total_second, .monotonic);
                    }
                }
            }
        }
    };
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

fn batchNormFinishKernel(comptime E: type) type {
    return struct {
        fn run(sums: [*]const f32, shift: [*]const E, st: nm.Stats, l: nm.Layout) callconv(.nvptx_kernel) void {
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
    };
}

export fn batchNormGradFinish(sums: [*]const f32, gamma_grad: [*]f32, beta_grad: [*]f32, c: u32) callconv(.nvptx_kernel) void {
    const ch = globalIndex();
    if (ch >= c) return;
    beta_grad[ch] += sums[ch];
    gamma_grad[ch] += sums[c + ch];
}
export fn batchNormApply(x: [*]const f32, residual: [*]const f32, y: [*]f32, p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() < l.len) nm.apply(f32, globalIndex(), x, residual, y, p, l);
}

export fn batchNormApplyQuad(x: [*]const u16, residual: [*]const u16, y: [*]u16, p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() * 4 < l.len) nm.applyQuad(u16, globalIndex() * 4, x, residual, y, p, l);
}

export fn batchNormGradInputQuad(io: nm.GradIo(u16), p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() * 4 < l.len) nm.gradInputQuad(u16, globalIndex() * 4, io, p, l);
}

export fn batchNormApplyHalf(x: [*]const u16, residual: [*]const u16, y: [*]u16, p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() < l.len) nm.apply(u16, globalIndex(), x, residual, y, p, l);
}

export fn batchNormGradInput(io: nm.GradIo(f32), p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() < l.len) nm.gradInput(f32, globalIndex(), io, p, l);
}

export fn batchNormGradInputHalf(io: nm.GradIo(u16), p: nm.Affine, l: nm.Layout) callconv(.nvptx_kernel) void {
    if (globalIndex() < l.len) nm.gradInput(u16, globalIndex(), io, p, l);
}

// ***** losses *****

// One warp per row: the log-probabilities of the row and its loss, with label smoothing.
export fn crossEntropyRows(x: [*]const f32, logp: [*]f32, labels: [*]const u8, row_loss: [*]f32, rows: u32, cols: u32, smoothing: f32) callconv(.nvptx_kernel) void {
    const at = warpRow();
    if (at.row >= rows) return;
    const base = at.row * cols;
    var largest: f32 = -std.math.inf(f32);
    var j = at.lane;
    while (j < cols) : (j += 32) largest = @max(largest, x[base + j]);
    largest = warpMax(largest);
    var total: f32 = 0;
    j = at.lane;
    while (j < cols) : (j += 32) total += expf(x[base + j] - largest);
    const shift = largest + logf(warpSum(total));
    var logp_sum: f32 = 0;
    j = at.lane;
    while (j < cols) : (j += 32) {
        logp[base + j] = x[base + j] - shift;
        logp_sum += logp[base + j];
    }
    logp_sum = warpSum(logp_sum);
    if (at.lane == 0) row_loss[at.row] = -(1 - smoothing) * (x[base + labels[at.row]] - shift) - smoothing / @as(f32, @floatFromInt(cols)) * logp_sum;
}

// One block of 256 threads sums the row losses.
export fn crossEntropyMean(row_loss: [*]const f32, out: [*]f32, rows: u32) callconv(.nvptx_kernel) void {
    const tid: usize = @workItemId(0);
    var total: f32 = 0;
    var r = tid;
    while (r < rows) : (r += threads) total += row_loss[r];
    const sum = blockSum(total, tid);
    if (tid == 0) out[0] = sum / @as(f32, @floatFromInt(rows));
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

export fn dropout(src: [*]const f32, dst: [*]f32, len: u32, seed: u64, step_seed: [*]const u64, keep: f32, accumulate: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) drop.applyOf(f32, globalIndex(), src, dst, seed +% step_seed[0], keep, accumulate != 0);
}

export fn dropoutHalf(src: [*]const u16, dst: [*]u16, len: u32, seed: u64, step_seed: [*]const u64, keep: f32, accumulate: u32) callconv(.nvptx_kernel) void {
    if (globalIndex() < len) drop.applyOf(u16, globalIndex(), src, dst, seed +% step_seed[0], keep, accumulate != 0);
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
    const config: aug.Config = .{
        .max_shift = args.max_shift,
        .max_angle = args.max_angle,
        .max_scale = args.max_scale,
        .flip = args.flip != 0,
        .mirror = args.mirror != 0,
        .cutout = args.cutout,
        .channels = args.channels,
        .seed = args.seed,
    };
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
        const a_elems = rows * depth;
        const b_elems = depth * cols;
        const stage_elems = a_elems + b_elems;
        // NOTE: static shared memory is limited to 48 KiB per block, so a 128 x 128 tile fits only three stages.
        const stages = if (4 * stage_elems * 2 <= 48 * 1024) 4 else 3;

        var smem: [stages * stage_elems]u16 align(128) addrspace(.shared) = undefined;

        inline fn aSlot(row: usize, chunk: usize) usize {
            return row * depth + ((chunk ^ ((row / 2) % 4)) * 8);
        }

        inline fn bSlot(k: usize, chunk: usize) usize {
            return k * cols + ((chunk ^ (k % 8)) * 8);
        }

        // The A tile stored by k instead of by row (depth rows of rows elements), swizzled like B; the weight gradient
        // loads the output gradient this way because it is contiguous along the rows of A.
        inline fn aTransposedSlot(k: usize, chunk: usize) usize {
            return k * rows + ((chunk ^ (k % 8)) * 8);
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

// NOTE: one thread per output element (per four elements when the channel count is divisible by four). A block per row left
// most of its 256 threads idle for the narrow rows of the stem convolution (27 values per position), and launched one block
// per position. len is the element count of the padded matrix; cols_padded is a multiple of four, so a quad never crosses a row.
export fn im2colHalf(x: [*]const f32, dst: [*]u16, d: cv.Dims, cols_padded: u32, len: u32) callconv(.nvptx_kernel) void {
    const wide = d.c % 4 == 0;
    const first = globalIndex() * @as(usize, if (wide) 4 else 1);
    if (first >= len) return;
    const row: u32 = @intCast(first / cols_padded);
    const cc: u32 = @intCast(first % cols_padded);
    const real = row < d.n * cv.outHeight(d) * cv.outWidth(d);
    const kc = d.k * d.k * d.c;
    if (!wide) {
        dst[first] = if (real and cc < kc) bf16.encode(cv.patchValue(x, cv.patchOf(row, d), cc, d)) else 0;
        return;
    }
    const values: @Vector(4, f32) = if (real and cc < kc) cv.patchValues(x, cv.patchOf(row, d), cc, d) else @splat(0);
    const packed_quad: *align(8) u64 = @ptrCast(@alignCast(dst + first));
    packed_quad.* = @as(u64, bf16.encode(values[0])) | @as(u64, bf16.encode(values[1])) << 16 | @as(u64, bf16.encode(values[2])) << 32 | @as(u64, bf16.encode(values[3])) << 48;
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
// does the same for the right operand (weight gradient), so no im2col buffer is ever written. In bf16, gather_b also
// reads its left operand, the transposed output gradient, straight from the (positions, c_out) gradient.
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

// Waits until at most pending groups of copies are still in flight; the pipeline keeps stages - 2 tiles loading ahead.
inline fn waitCopies(comptime pending: usize) void {
    asm volatile (std.fmt.comptimePrint("cp.async.wait_group {d};", .{pending}) ::: .{ .memory = true });
}

// One gathered operand copy: an asynchronous 16-byte copy with zero-fill, or, when the operand is transformed on
// load (magic), a NaN marker for out-of-bounds taps that the transform turns back into an exact zero.
inline fn fillSlot(dst: u32, src: anytype, valid: bool, comptime magic: bool) void {
    if (magic and !valid) return storeMagicShared(dst);
    copyAsyncZfill(dst, src, valid);
}

const Acc = [4][4][4]f32;
const Warp = struct { lane: usize, row: usize, col: usize };
const Block = struct { tid: usize, row0: usize, col0: usize, k_begin: usize };

// Chooses a tensor core GEMM at compile time: bf16 or TF32 inputs, the block tile size and the operand mode.
const GemmKind = struct {
    tf32: bool = false,
    // With tf32 set: multiply in full f32 on the CUDA cores instead of rounding to TF32 for the tensor cores. The tile, the
    // gathers and the pipeline are the same, so a strict f32 convolution needs no im2col matrix either.
    exact: bool = false,
    rows: usize = 128,
    cols: usize = 64,
    mode: GemmMode,
    // Adds per-channel batch-norm sums of the output to the epilogue (see statsTile).
    bn_stats: bool = false,
    // gather_a only: the left operand is a virtual batch-norm-plus-ReLU output, recomputed on load from the raw
    // values (see affinePair). Out-of-bounds taps carry NaN markers so the transform yields exact zeros.
    affine_a: bool = false,
    // gather_b only: the same transform applied to the gathered right operand of a weight gradient.
    affine_b: bool = false,
    // Adds the masked per-channel sums a batch-norm backward needs to the epilogue (see dxStatsTile).
    dx_stats: bool = false,

    fn Elem(comptime self: GemmKind) type {
        return if (self.tf32) f32 else u16;
    }

    fn Tile(comptime self: GemmKind) type {
        return if (self.tf32) Tf32Tile(self.rows, self.cols) else TensorTile(self.rows, self.cols);
    }

    // True when the left operand is stored transposed, as (k, rows), and loaded with ldmatrix.trans.
    fn transposedA(comptime self: GemmKind) bool {
        return self.mode == .gather_b and !self.tf32;
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
        chunk_t: [a_per]usize,
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
        // gather_b: the column (tap and channel) of each copy is fixed, and its output position p advances by depth per issue.
        tap_ky: [b_per]i32,
        tap_kx: [b_per]i32,
        tap_channel: [b_per]usize,
        p: [b_per]u32,
        p_n: [b_per]u32,
        p_y: [b_per]u32,
        p_x: [b_per]u32,
        step_y: u32,
        step_x: u32,

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
            if (comptime kind.transposedA()) {
                // NOTE: here rows_a holds the k of each copy and chunk_t its group of eight rows; a tile row of A is a
                // contiguous run of the gradient row of one position.
                inline for (0..a_per) |i| {
                    const index = tid + 256 * i;
                    self.rows_a[i] = index / (tc.rows / 8);
                    self.chunk_t[i] = index % (tc.rows / 8);
                    self.dst_a[i] = tc.aTransposedSlot(self.rows_a[i], self.chunk_t[i]);
                }
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
            if (mode == .gather_b) {
                // NOTE: the divisions by run-time sizes are done once here; issue only adds, which matters because the
                // weight gradient issues one gathered copy per thread and tile.
                self.step_y = @intCast(tc.depth / geo.out_w);
                self.step_x = @intCast(tc.depth % geo.out_w);
                inline for (0..b_per) |j| {
                    const col = col0 + self.chunk_b[j] * lane_elems;
                    const tap: u32 = @intCast(col / geo.c);
                    self.tap_channel[j] = col % geo.c;
                    self.tap_ky[j] = @intCast(tap / geo.k);
                    self.tap_kx[j] = @intCast(tap % geo.k);
                    const p: u32 = @intCast(k_begin + self.k_b[j]);
                    self.p[j] = p;
                    self.p_n[j] = p / (geo.out_h * geo.out_w);
                    self.p_y[j] = p / geo.out_w % geo.out_h;
                    self.p_x[j] = p % geo.out_w;
                }
            }
            return self;
        }

        inline fn issue(self: *@This(), stage: usize, kk: usize) void {
            const base = stage * tc.stage_elems;
            const geometry = self.geo;
            // NOTE: the taps of a dilated convolution sit dilation apart, and a grouped launch reads its channels out of a
            // tensor whose rows hold all the groups, so the memory stride differs from the logical channel count.
            const dil: i32 = @intCast(geometry.dilation);
            const channel_step = if (geometry.c_step != 0) geometry.c_step else geometry.c;
            if (mode == .gather_a) {
                inline for (0..a_per) |i| {
                    const y = self.origin_y[i] + self.cursor_ky * dil;
                    const x = self.origin_x[i] + self.cursor_kx * dil;
                    const valid = self.origin_valid[i] and y >= 0 and y < @as(i32, @intCast(geometry.h)) and x >= 0 and x < @as(i32, @intCast(geometry.w));
                    const at = if (valid)
                        ((@as(usize, self.origin_n[i]) * geometry.h + @as(usize, @intCast(y))) * geometry.w +
                            @as(usize, @intCast(x))) * channel_step + self.cursor_channel + self.chunk_a * lane_elems
                    else
                        0;
                    fillSlot(tc.sharedAddress(base + self.dst_a[i]), self.a + at, valid, kind.affine_a);
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
            } else if (comptime kind.transposedA()) {
                inline for (0..a_per) |i| {
                    const p = kk + self.rows_a[i];
                    const row = self.row_base + self.chunk_t[i] * lane_elems;
                    const valid = p < geometry.positions and row < self.rows_total;
                    copyAsyncZfill(tc.sharedAddress(base + self.dst_a[i]), self.a + if (valid) p * self.rows_total + row else 0, valid);
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
                    const y = @as(i32, @intCast(self.p_y[j] * geometry.stride)) + self.tap_ky[j] * dil - @as(i32, @intCast(geometry.pad));
                    const x = @as(i32, @intCast(self.p_x[j] * geometry.stride)) + self.tap_kx[j] * dil - @as(i32, @intCast(geometry.pad));
                    const valid = self.p[j] < geometry.positions and y >= 0 and y < @as(i32, @intCast(geometry.h)) and x >= 0 and x < @as(i32, @intCast(geometry.w));
                    const at = if (valid) ((@as(usize, self.p_n[j]) * geometry.h + @as(usize, @intCast(y))) * geometry.w + @as(usize, @intCast(x))) * channel_step + self.tap_channel[j] else 0;
                    fillSlot(tc.sharedAddress(base + self.dst_b[j]), self.b + at, valid, kind.affine_b);
                    // NOTE: issue is called for consecutive tiles of K, so the position moves by exactly depth each time.
                    self.p[j] += tc.depth;
                    self.p_x[j] += self.step_x;
                    self.p_y[j] += self.step_y;
                    if (self.p_x[j] >= geometry.out_w) {
                        self.p_x[j] -= geometry.out_w;
                        self.p_y[j] += 1;
                    }
                    while (self.p_y[j] >= geometry.out_h) {
                        self.p_y[j] -= geometry.out_h;
                        self.p_n[j] += 1;
                    }
                } else {
                    const valid = col < self.stride_n;
                    const at = if (valid) (kk + self.k_b[j]) * self.stride_n + col else 0;
                    copyAsyncZfill(tc.sharedAddress(base + self.dst_b[j]), self.b + at, valid);
                }
            }
        }
    };
}

// Per-position scale and shift of a virtual batch-norm-plus-ReLU operand, expanded over the (tap, channel) index
// and packed as bf16 pairs: scale2 and shift2 hold the pairs (2j, 2j+1) for the A fragments, scalec and shiftc
// hold one duplicated value per column for the B fragments.
const AffineLoad = struct { scale2: [*]const u32, shift2: [*]const u32, scalec: [*]const u32, shiftc: [*]const u32 };

// Overlaps loading and computing: the copies for the next stages are in flight while the current tile is multiplied.
inline fn pipeline(comptime kind: GemmKind, loader: anytype, block: Block, tiles: usize, warp: Warp, aff: AffineLoad, acc: *Acc) void {
    const tc = kind.Tile();
    inline for (0..tc.stages - 1) |stage| {
        if (stage < tiles) loader.issue(stage, block.k_begin + stage * tc.depth);
        commitCopies();
    }
    var tile: usize = 0;
    while (tile < tiles) : (tile += 1) {
        waitCopies(tc.stages - 2);
        barrier();
        const ahead = tile + tc.stages - 1;
        if (ahead < tiles) loader.issue(ahead % tc.stages, block.k_begin + ahead * tc.depth);
        commitCopies();
        const base = (tile % tc.stages) * tc.stage_elems;
        if (kind.exact) f32Compute(tc, warp, base, acc) else if (kind.tf32) tf32Compute(tc, warp, base, acc) else bf16Compute(tc, kind, warp, base, block.k_begin + tile * tc.depth, block.col0, aff, acc);
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
                    // NOTE: a grouped launch stores its slice of C with c_pitch as the row stride and c_offset as the column
                    // base; both are zero for a dense product.
                    const pitch = if (shape.c_pitch != 0) shape.c_pitch else if (shape.transpose_out != 0) shape.m else shape.n;
                    const at = (if (shape.transpose_out != 0) col * pitch + r else r * pitch + col) + shape.c_offset;
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

// The per-channel sums a batch norm needs, accumulated in the GEMM epilogue: shift centers the values (the host
// passes the running mean, a close estimate, so the squared sums keep their precision), sums takes the results
// atomically, and in an ordered (deterministic) run partial takes one entry per block row for an ordered combine.
const BnStats = struct { shift: [*]const f32, sums: [*]f32, partial: [*]f32, ordered: u32 };

// The per-channel sums a batch-norm backward needs, accumulated in the epilogue of the GEMM that produces the
// output gradient of a virtual normalized map: x is the raw map the norm ran on, the affine parameters rebuild
// the ReLU mask and the normalized values, and sums takes the masked sums of dy and of dy * x_hat.
const DxStats = struct { x: [*]const u16, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32, sums: [*]f32, partial: [*]f32, ordered: u32 };

// Folds the per-lane column sums across the warp and the warp rows of the tile, then flushes them to the global
// sums (atomically, or to the per-block-row partial buffer of an ordered run).
inline fn flushStats(comptime kind: GemmKind, sums: *const [4][2]f32, squares: *const [4][2]f32, shape: ga.TensorShape, block: Block, warp: Warp, out_sums: [*]f32, out_partial: [*]f32, ordered: u32) void {
    const tc = kind.Tile();
    var total_sums = sums.*;
    var total_squares = squares.*;
    // NOTE: the lanes that share a column are four apart (the column comes from lane % 4, the row from lane / 4),
    // so folding those three lane bits leaves the warp's column sums in every lane.
    inline for (.{ 4, 8, 16 }) |offset| {
        inline for (0..4) |ni| {
            inline for (0..2) |qp| {
                total_sums[ni][qp] = shflXor(total_sums[ni][qp], offset);
                total_squares[ni][qp] = shflXor(total_squares[ni][qp], offset);
            }
        }
    }
    // The pipeline is done, so its shared memory hosts the reduction across the warp rows of the tile.
    barrier();
    const red: [*]addrspace(.shared) f32 = @ptrCast(&tc.smem[0]);
    if (warp.lane < 4) {
        inline for (0..4) |ni| {
            inline for (0..2) |qp| {
                const at = (warp.row * tc.cols + warp.col * 32 + ni * 8 + warp.lane * 2 + qp) * 2;
                red[at] = total_sums[ni][qp];
                red[at + 1] = total_squares[ni][qp];
            }
        }
    }
    barrier();
    const warp_rows = 8 / (tc.cols / 32);
    if (block.tid < tc.cols) {
        const ch = block.col0 + block.tid;
        if (ch < shape.n) {
            var total: f32 = 0;
            var total_sq: f32 = 0;
            inline for (0..warp_rows) |wr| {
                total += red[(wr * tc.cols + block.tid) * 2];
                total_sq += red[(wr * tc.cols + block.tid) * 2 + 1];
            }
            if (ordered != 0) {
                const at = @as(usize, @workGroupId(1)) * 2 * shape.n + ch;
                out_partial[at] = total;
                out_partial[at + shape.n] = total_sq;
            } else {
                _ = @atomicRmw(f32, &out_sums[ch], .Add, total, .monotonic);
                _ = @atomicRmw(f32, &out_sums[shape.n + ch], .Add, total_sq, .monotonic);
            }
        }
    }
}

// Accumulates the sums of v - shift and (v - shift)^2 per output channel of the tile, over the same values the
// tile stores (bf16-rounded when the output is bf16, so the sums match the stored feature map).
inline fn statsTile(comptime kind: GemmKind, st: BnStats, acc: *const Acc, shape: ga.TensorShape, block: Block, warp: Warp) void {
    const tc = kind.Tile();
    const half_output = !kind.tf32;
    var sums: [4][2]f32 = @splat(@splat(0));
    var squares: [4][2]f32 = @splat(@splat(0));
    var shift: [4][2]f32 = undefined;
    inline for (0..4) |ni| {
        inline for (0..2) |qp| {
            const col = block.col0 + warp.col * 32 + ni * 8 + (warp.lane % 4) * 2 + qp;
            shift[ni][qp] = if (col < shape.n) st.shift[col] else 0;
        }
    }
    inline for (0..tc.warp_m / 16) |mi| {
        inline for (0..4) |ni| {
            inline for (0..4) |q| {
                const r = block.row0 + warp.row * tc.warp_m + mi * 16 + warp.lane / 4 + (q / 2) * 8;
                const col = block.col0 + warp.col * 32 + ni * 8 + (warp.lane % 4) * 2 + q % 2;
                if (r < shape.m and col < shape.n) {
                    const stored = if (half_output) bf16.decode(bf16.encode(acc[mi][ni][q])) else acc[mi][ni][q];
                    const v = stored - shift[ni][q % 2];
                    sums[ni][q % 2] += v;
                    squares[ni][q % 2] += v * v;
                }
            }
        }
    }
    flushStats(kind, &sums, &squares, shape, block, warp, st.sums, st.partial, st.ordered);
}

// Accumulates the sums of g and g * x_hat per output channel of the tile, where g is the stored output gradient
// masked by the ReLU of the virtual normalized map: the map itself is never written, so the mask and x_hat are
// rebuilt from the raw map x and the affine parameters (mask is gamma * x_hat + beta > 0).
inline fn dxStatsTile(comptime kind: GemmKind, st: DxStats, acc: *const Acc, shape: ga.TensorShape, block: Block, warp: Warp) void {
    const tc = kind.Tile();
    var sums: [4][2]f32 = @splat(@splat(0));
    var squares: [4][2]f32 = @splat(@splat(0));
    var mean: [4][2]f32 = undefined;
    var inv_std: [4][2]f32 = undefined;
    var gamma: [4][2]f32 = undefined;
    var beta: [4][2]f32 = undefined;
    inline for (0..4) |ni| {
        inline for (0..2) |qp| {
            const col = block.col0 + warp.col * 32 + ni * 8 + (warp.lane % 4) * 2 + qp;
            const valid = col < shape.n;
            mean[ni][qp] = if (valid) st.mean[col] else 0;
            inv_std[ni][qp] = if (valid) st.inv_std[col] else 0;
            gamma[ni][qp] = if (valid) st.gamma[col] else 0;
            beta[ni][qp] = if (valid) st.beta[col] else 0;
        }
    }
    inline for (0..tc.warp_m / 16) |mi| {
        inline for (0..4) |ni| {
            inline for (0..2) |qh| {
                const r = block.row0 + warp.row * tc.warp_m + mi * 16 + warp.lane / 4 + qh * 8;
                const col = block.col0 + warp.col * 32 + ni * 8 + (warp.lane % 4) * 2;
                // NOTE: col and shape.n are even (whole 8-channel groups), so the pair load is aligned and in bounds.
                if (r < shape.m and col < shape.n) {
                    const pair: u32 = @as(*align(4) const u32, @ptrCast(@alignCast(st.x + r * shape.n + col))).*;
                    const x0: f32 = @bitCast(pair << 16);
                    const x1: f32 = @bitCast(pair & 0xffff0000);
                    const xh0 = (x0 - mean[ni][0]) * inv_std[ni][0];
                    const xh1 = (x1 - mean[ni][1]) * inv_std[ni][1];
                    const g0 = if (gamma[ni][0] * xh0 + beta[ni][0] > 0) bf16.decode(bf16.encode(acc[mi][ni][2 * qh])) else 0;
                    const g1 = if (gamma[ni][1] * xh1 + beta[ni][1] > 0) bf16.decode(bf16.encode(acc[mi][ni][2 * qh + 1])) else 0;
                    sums[ni][0] += g0;
                    sums[ni][1] += g1;
                    squares[ni][0] += g0 * xh0;
                    squares[ni][1] += g1 * xh1;
                }
            }
        }
    }
    flushStats(kind, &sums, &squares, shape, block, warp, st.sums, st.partial, st.ordered);
}

// Optional operands of a tensor core GEMM beyond the three matrices: epilogue statistics and the on-load affine transform.
const GemmExtras = struct { stats: ?BnStats = null, dx: ?DxStats = null, affine: ?AffineLoad = null };

// One thread block computes one tile of C. Work group x and y pick the tile and work group z picks the range of K for split-K.
inline fn tileGemm(comptime kind: GemmKind, a: [*]const kind.Elem(), b: [*]const kind.Elem(), c: [*]f32, extras: GemmExtras, shape: ga.TensorShape, geo: ga.ConvGeometry) void {
    const tc = kind.Tile();
    const tid: usize = @workItemId(0);
    const warp: Warp = .{ .lane = tid % 32, .row = tid / 32 / (tc.cols / 32), .col = tid / 32 % (tc.cols / 32) };
    const block: Block = .{ .tid = tid, .row0 = @as(usize, @workGroupId(1)) * tc.rows, .col0 = @as(usize, @workGroupId(0)) * tc.cols, .k_begin = @as(usize, @workGroupId(2)) * shape.chunk };
    const tiles = (@min(shape.k_padded, block.k_begin + shape.chunk) -| block.k_begin) / tc.depth;

    var loader = GemmLoader(kind).init(a, b, geo, shape, block);
    var acc: Acc = @splat(@splat(@splat(0)));
    pipeline(kind, &loader, block, tiles, warp, (extras.affine orelse undefined), &acc);
    storeTile(kind, c, &acc, shape, block, warp);
    if (comptime kind.bn_stats) statsTile(kind, extras.stats.?, &acc, shape, block, warp);
    if (comptime kind.dx_stats) dxStatsTile(kind, extras.dx.?, &acc, shape, block, warp);
}

// Applies the affine map and ReLU of a virtual normalized map to a pair of bf16 values held in one register,
// with the scale and shift packed the same way. A NaN marker (an out-of-bounds tap) comes out as exactly zero
// because max treats a NaN operand as missing and keeps the other.
inline fn affinePair(word: u32, scale2: u32, shift2: u32) u32 {
    return asm ("fma.rn.bf16x2 %[out], %[w], %[s], %[h];\n\tmax.bf16x2 %[out], %[out], %[z];"
        : [out] "=r" (-> u32),
        : [w] "r" (word),
          [s] "r" (scale2),
          [h] "r" (shift2),
          [z] "r" (@as(u32, 0)),
    );
}

inline fn bf16Compute(comptime tc: type, comptime kind: GemmKind, warp: Warp, base: usize, kk: usize, col0: usize, aff: AffineLoad, acc: *Acc) void {
    inline for (0..tc.depth / 16) |step| {
        var a_frag: [4][4]u32 = undefined;
        inline for (0..tc.warp_m / 16) |mi| {
            // NOTE: lanes 8j to 8j + 7 address the rows of 8 x 8 block j: block j covers rows + 8 (j % 2) and k + 8 (j / 2),
            // which is the register order of the A fragment. A transposed tile is stored by k, so .trans turns it back.
            const row = warp.row * tc.warp_m + mi * 16 + warp.lane % 8 + 8 * ((warp.lane / 8) % 2);
            if (kind.transposedA()) {
                const k = step * 16 + warp.lane % 8 + 8 * (warp.lane / 16);
                const first_row = warp.row * tc.warp_m + mi * 16 + 8 * ((warp.lane / 8) % 2);
                a_frag[mi] = tc.ldmatrixTransposed(tc.sharedAddress(base + tc.aTransposedSlot(k, first_row / 8)));
            } else {
                a_frag[mi] = tc.ldmatrix(tc.sharedAddress(base + tc.aSlot(row, step * 2 + warp.lane / 16)));
            }
        }
        // NOTE: register j of the fragment holds the elements of k + 8 * (j / 2) + (lane % 4) * 2 and the next one.
        if (comptime kind.affine_a) {
            inline for (0..2) |kh| {
                const k0 = kk + step * 16 + 8 * kh + (warp.lane % 4) * 2;
                const s = aff.scale2[k0 / 2];
                const h = aff.shift2[k0 / 2];
                inline for (0..tc.warp_m / 16) |mi| {
                    a_frag[mi][2 * kh] = affinePair(a_frag[mi][2 * kh], s, h);
                    a_frag[mi][2 * kh + 1] = affinePair(a_frag[mi][2 * kh + 1], s, h);
                }
            }
        }
        inline for (0..2) |pair| {
            const k = step * 16 + warp.lane % 8 + 8 * ((warp.lane / 8) % 2);
            var frag = tc.ldmatrixTransposed(tc.sharedAddress(base + tc.a_elems + tc.bSlot(k, warp.col * 4 + pair * 2 + warp.lane / 16)));
            // NOTE: registers 0 and 1 cover one column of the tile, registers 2 and 3 the next; both elements of a
            // register are the same column (two consecutive positions), so one duplicated scale and shift serve it.
            if (comptime kind.affine_b) {
                inline for (0..2) |half| {
                    const col = col0 + (warp.col * 4 + pair * 2 + half) * 8 + warp.lane / 4;
                    const s = aff.scalec[col];
                    const h = aff.shiftc[col];
                    frag[2 * half] = affinePair(frag[2 * half], s, h);
                    frag[2 * half + 1] = affinePair(frag[2 * half + 1], s, h);
                }
            }
            inline for (0..tc.warp_m / 16) |mi| {
                tc.mma(&acc[mi][pair * 2], a_frag[mi], .{ frag[0], frag[1] });
                tc.mma(&acc[mi][pair * 2 + 1], a_frag[mi], .{ frag[2], frag[3] });
            }
        }
    }
}

// Strict f32 on the CUDA cores, for the elements that the mma fragments of tf32Compute would hold: a thread owns rows r and r + 8 of
// every 16-row group and two neighbouring columns of every group of eight, so storeTile needs no second layout.
inline fn f32Compute(comptime tc: type, warp: Warp, base: usize, acc: *Acc) void {
    inline for (0..tc.depth) |k| {
        var a: [tc.warp_m / 16][2]f32 = undefined;
        inline for (0..tc.warp_m / 16) |mi| {
            const r = warp.row * tc.warp_m + mi * 16 + warp.lane / 4;
            a[mi] = .{ tc.smem[base + r * tc.a_pitch + k], tc.smem[base + (r + 8) * tc.a_pitch + k] };
        }
        inline for (0..4) |ni| {
            const n = warp.col * 32 + ni * 8 + (warp.lane % 4) * 2;
            const b0 = tc.smem[base + tc.a_elems + k * tc.b_pitch + n];
            const b1 = tc.smem[base + tc.a_elems + k * tc.b_pitch + n + 1];
            inline for (0..tc.warp_m / 16) |mi| {
                acc[mi][ni][0] = @mulAdd(f32, a[mi][0], b0, acc[mi][ni][0]);
                acc[mi][ni][1] = @mulAdd(f32, a[mi][0], b1, acc[mi][ni][1]);
                acc[mi][ni][2] = @mulAdd(f32, a[mi][1], b0, acc[mi][ni][2]);
                acc[mi][ni][3] = @mulAdd(f32, a[mi][1], b1, acc[mi][ni][3]);
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
    tileGemm(.{ .mode = .dense }, a, b, c, .{}, shape, zero_geometry);
}

export fn convGemmForward(x: [*]const u16, w: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .gather_a }, x, w, c, .{}, shape, geo);
}

export fn convGemmWeightGrad(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .gather_b }, dy_t, x, c, .{}, shape, geo);
}

export fn convGemmForwardWide(x: [*]const u16, w: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 128, .cols = 128, .mode = .gather_a }, x, w, c, .{}, shape, geo);
}

export fn convGemmForwardStats(x: [*]const u16, w: [*]const u16, c: [*]f32, shift: [*]const f32, sums: [*]f32, partial: [*]f32, ordered: u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .gather_a, .bn_stats = true }, x, w, c, .{ .stats = .{ .shift = shift, .sums = sums, .partial = partial, .ordered = ordered } }, shape, geo);
}

export fn convGemmForwardStatsWide(x: [*]const u16, w: [*]const u16, c: [*]f32, shift: [*]const f32, sums: [*]f32, partial: [*]f32, ordered: u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 128, .cols = 128, .mode = .gather_a, .bn_stats = true }, x, w, c, .{ .stats = .{ .shift = shift, .sums = sums, .partial = partial, .ordered = ordered } }, shape, geo);
}

// Forward over a virtual batch-norm-plus-ReLU input: the gathered left operand is transformed on load from the raw
// map x with the per-position scale and shift, and the epilogue accumulates the next norm's batch statistics.
export fn convGemmForwardAffineStats(x: [*]const u16, w: [*]const u16, c: [*]f32, scale2: [*]const u32, shift2: [*]const u32, stats_shift: [*]const f32, sums: [*]f32, partial: [*]f32, ordered: u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .gather_a, .bn_stats = true, .affine_a = true }, x, w, c, .{ .stats = .{ .shift = stats_shift, .sums = sums, .partial = partial, .ordered = ordered }, .affine = .{ .scale2 = scale2, .shift2 = shift2, .scalec = undefined, .shiftc = undefined } }, shape, geo);
}

export fn convGemmForwardAffineStatsWide(x: [*]const u16, w: [*]const u16, c: [*]f32, scale2: [*]const u32, shift2: [*]const u32, stats_shift: [*]const f32, sums: [*]f32, partial: [*]f32, ordered: u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 128, .cols = 128, .mode = .gather_a, .bn_stats = true, .affine_a = true }, x, w, c, .{ .stats = .{ .shift = stats_shift, .sums = sums, .partial = partial, .ordered = ordered }, .affine = .{ .scale2 = scale2, .shift2 = shift2, .scalec = undefined, .shiftc = undefined } }, shape, geo);
}

// Input-gradient GEMM (reversed geometry) whose epilogue accumulates the masked batch-norm gradient sums of the
// virtual normalized map the gradient belongs to, rebuilding the mask and x_hat from the raw map x_bn.
export fn convGemmForwardDxStats(x: [*]const u16, w: [*]const u16, c: [*]f32, x_bn: [*]const u16, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32, sums: [*]f32, partial: [*]f32, ordered: u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .gather_a, .dx_stats = true }, x, w, c, .{ .dx = .{ .x = x_bn, .mean = mean, .inv_std = inv_std, .gamma = gamma, .beta = beta, .sums = sums, .partial = partial, .ordered = ordered } }, shape, geo);
}

export fn convGemmForwardDxStatsWide(x: [*]const u16, w: [*]const u16, c: [*]f32, x_bn: [*]const u16, mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32, sums: [*]f32, partial: [*]f32, ordered: u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 128, .cols = 128, .mode = .gather_a, .dx_stats = true }, x, w, c, .{ .dx = .{ .x = x_bn, .mean = mean, .inv_std = inv_std, .gamma = gamma, .beta = beta, .sums = sums, .partial = partial, .ordered = ordered } }, shape, geo);
}

// Weight gradient over a virtual batch-norm-plus-ReLU input: the gathered right operand is transformed on load
// with the per-column scale and shift, each a duplicated bf16 pair.
export fn convGemmWeightGradAffine(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, scalec: [*]const u32, shiftc: [*]const u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .mode = .gather_b, .affine_b = true }, dy_t, x, c, .{ .affine = .{ .scale2 = undefined, .shift2 = undefined, .scalec = scalec, .shiftc = shiftc } }, shape, geo);
}

export fn convGemmWeightGradAffineShort(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, scalec: [*]const u32, shiftc: [*]const u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 64, .cols = 128, .mode = .gather_b, .affine_b = true }, dy_t, x, c, .{ .affine = .{ .scale2 = undefined, .shift2 = undefined, .scalec = scalec, .shiftc = shiftc } }, shape, geo);
}

export fn convGemmWeightGradAffineWide(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, scalec: [*]const u32, shiftc: [*]const u32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 128, .cols = 128, .mode = .gather_b, .affine_b = true }, dy_t, x, c, .{ .affine = .{ .scale2 = undefined, .shift2 = undefined, .scalec = scalec, .shiftc = shiftc } }, shape, geo);
}

// Expands the per-channel batch-norm affine map to packed per-(tap, channel) scale and shift words for the
// on-load transform: scale = gamma * inv_std and shift = beta - mean * scale, so relu(scale * x + shift) ==
// relu(norm(x)) up to the bf16 rounding of scale and shift (one extra bf16 step next to the map's own bf16
// storage). Entries past k_real pad the GEMM tiles and are zero. len is even. scalec and shiftc hold len words
// with one value duplicated in both halves; scale2 and shift2 hold len / 2 words of two consecutive positions,
// matching the fragment layouts of the B and A operands respectively.
export fn affineExpand(mean: [*]const f32, inv_std: [*]const f32, gamma: [*]const f32, beta: [*]const f32, scale2: [*]u32, shift2: [*]u32, scalec: [*]u32, shiftc: [*]u32, c: u32, k_real: u32, len: u32) callconv(.nvptx_kernel) void {
    const j = globalIndex();
    if (j >= len) return;
    var s: f32 = 0;
    var h: f32 = 0;
    if (j < k_real) {
        const ch = j % c;
        s = gamma[ch] * inv_std[ch];
        h = beta[ch] - mean[ch] * s;
    }
    scalec[j] = @as(u32, bf16.encode(s)) * 0x10001;
    shiftc[j] = @as(u32, bf16.encode(h)) * 0x10001;
    if (j >= len / 2) return;
    const j0 = 2 * j;
    const j1 = 2 * j + 1;
    var s0: f32 = 0;
    var h0: f32 = 0;
    var s1: f32 = 0;
    var h1: f32 = 0;
    if (j0 < k_real) {
        const ch = j0 % c;
        s0 = gamma[ch] * inv_std[ch];
        h0 = beta[ch] - mean[ch] * s0;
    }
    if (j1 < k_real) {
        const ch = j1 % c;
        s1 = gamma[ch] * inv_std[ch];
        h1 = beta[ch] - mean[ch] * s1;
    }
    scale2[j] = @as(u32, bf16.encode(s1)) << 16 | bf16.encode(s0);
    shift2[j] = @as(u32, bf16.encode(h1)) << 16 | bf16.encode(h0);
}

export fn convGemmWeightGradWide(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 128, .cols = 128, .mode = .gather_b }, dy_t, x, c, .{}, shape, geo);
}

export fn convGemmWeightGradShort(dy_t: [*]const u16, x: [*]const u16, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .rows = 64, .cols = 128, .mode = .gather_b }, dy_t, x, c, .{}, shape, geo);
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
        // NOTE: an f32 tile is twice the bytes of its bf16 twin, so the 128 x 128 tile drops to two stages to stay under the
        // 48 KiB of static shared memory; the smaller tiles keep the three-stage pipeline.
        const stages = if (tile_rows * tile_cols > 128 * 64) 2 else 3;
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
    tileGemm(.{ .tf32 = true, .mode = .gather_a }, x, w, c, .{}, shape, geo);
}

export fn convTf32WeightGrad(dy_t: [*]const f32, x: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .mode = .gather_b }, dy_t, x, c, .{}, shape, geo);
}

export fn convTf32WeightGradShort(dy_t: [*]const f32, x: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .rows = 64, .cols = 128, .mode = .gather_b }, dy_t, x, c, .{}, shape, geo);
}

export fn convExactForward(x: [*]const f32, w: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .exact = true, .mode = .gather_a }, x, w, c, .{}, shape, geo);
}

export fn convExactForwardWide(x: [*]const f32, w: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .exact = true, .rows = 128, .cols = 128, .mode = .gather_a }, x, w, c, .{}, shape, geo);
}

export fn convExactWeightGradWide(dy_t: [*]const f32, x: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .exact = true, .rows = 128, .cols = 128, .mode = .gather_b }, dy_t, x, c, .{}, shape, geo);
}

export fn convExactWeightGrad(dy_t: [*]const f32, x: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .exact = true, .mode = .gather_b }, dy_t, x, c, .{}, shape, geo);
}

export fn convExactWeightGradShort(dy_t: [*]const f32, x: [*]const f32, c: [*]f32, shape: ga.TensorShape, geo: ga.ConvGeometry) callconv(.nvptx_kernel) void {
    tileGemm(.{ .tf32 = true, .exact = true, .rows = 64, .cols = 128, .mode = .gather_b }, dy_t, x, c, .{}, shape, geo);
}

export fn rotateWeightsF32(src: [*]const f32, dst: [*]f32, args: ga.RotateArgs) callconv(.nvptx_kernel) void {
    const t = globalIndex();
    if (t >= @as(usize, args.rows_padded) * args.cols_padded) return;
    const i: u32 = @intCast(t / args.cols_padded);
    const j: u32 = @intCast(t % args.cols_padded);
    const tap = i / args.c_out;
    const co = i % args.c_out;
    const source_tap = (args.k - 1 - tap / args.k) * args.k + (args.k - 1 - tap % args.k);
    const pitch = if (args.src_pitch != 0) args.src_pitch else args.c_out;
    dst[t] = if (j < args.c_in) src[args.src_offset + (source_tap * args.c_in + j) * pitch + co] else 0;
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
