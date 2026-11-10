const std = @import("std");
const Io = std.Io;
const Gpu = @import("gpu/gpu.zig").Gpu;

pub fn naive(comptime T: type, m: usize, k: usize, n: usize, a: []const T, b: []const T, out: []T) void {
    std.debug.assert(a.len >= m * k and b.len >= k * n and out.len >= m * n);
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: T = 0;
            for (0..k) |p| {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

pub fn vec(comptime T: type, m: usize, k: usize, n: usize, a: []const T, b: []const T, out: []T) void {
    std.debug.assert(a.len >= m * k and b.len >= k * n and out.len >= m * n);
    const lanes = comptime std.simd.suggestVectorLength(T) orelse 1;
    const Vec = @Vector(lanes, T);

    @memset(out[0 .. m * n], 0);
    for (0..m) |i| {
        const out_row = out[i * n ..][0..n];
        for (0..k) |p| {
            const aip = a[i * k + p];
            const aip_v: Vec = @splat(aip);
            const b_row = b[p * n ..][0..n];

            var j: usize = 0;
            while (j + lanes <= n) : (j += lanes) {
                const bv: Vec = b_row[j..][0..lanes].*;
                const ov: Vec = out_row[j..][0..lanes].*;
                out_row[j..][0..lanes].* = @mulAdd(Vec, aip_v, bv, ov);
            }
            while (j < n) : (j += 1) {
                out_row[j] += aip * b_row[j];
            }
        }
    }
}

pub const TileConfig = struct {
    mr: usize = 6,
    nv: usize = 2,
    mc: usize = 96,
    kc: usize = 256,
    nc: usize = 1024,
    trans_a: bool = false,
    trans_b: bool = false,
    accumulate: bool = false,
};

pub const Parallel = struct { io: ?Io = null, threads: usize = 1, gpu: ?*Gpu = null };

const min_flops_per_part = 2_000_000;
const max_workers = 15;
const spin_limit = 4000;

const Mailbox = struct {
    seq: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    parked: std.atomic.Value(u32) = .init(0),
    run: *const fn (*const anyopaque, usize) void = undefined,
    ctx: *const anyopaque = undefined,
    part: usize = 0,
    io: Io = undefined,
};

var mailboxes: [max_workers]Mailbox = @splat(.{});
var spawned: usize = 0;

pub fn flushDenormals() void {
    if (comptime @import("builtin").cpu.arch != .x86_64) return;
    var csr: u32 = 0;
    asm volatile ("stmxcsr (%[p])"
        :
        : [p] "r" (&csr),
        : .{ .memory = true });
    csr |= 0x8040;
    asm volatile ("ldmxcsr (%[p])"
        :
        : [p] "r" (&csr),
        : .{ .memory = true });
}

fn workerMain(box: *Mailbox) void {
    flushDenormals();
    var seen: u32 = 0;
    while (true) {
        var spins: u32 = 0;
        while (box.seq.load(.acquire) == seen) {
            if (spins < spin_limit) {
                spins += 1;
                std.atomic.spinLoopHint();
                continue;
            }
            box.parked.store(1, .seq_cst);
            if (box.seq.load(.seq_cst) == seen) box.io.futexWait(u32, &box.seq.raw, seen) catch {};
            box.parked.store(0, .monotonic);
            spins = 0;
        }
        seen = box.seq.load(.acquire);
        box.run(box.ctx, box.part);
        box.done.store(seen, .release);
    }
}

fn parallelRun(par: Parallel, parts: usize, ctx: *const anyopaque, run: *const fn (*const anyopaque, usize) void) void {
    const workers = parts - 1;
    while (spawned < workers) : (spawned += 1) {
        mailboxes[spawned].io = par.io.?;
        const thread = std.Thread.spawn(.{}, workerMain, .{&mailboxes[spawned]}) catch {
            for (0..parts) |p| run(ctx, p);
            return;
        };
        thread.detach();
    }
    for (mailboxes[0..workers], 1..) |*box, part| {
        box.run = run;
        box.ctx = ctx;
        box.part = part;
        box.seq.store(box.seq.load(.monotonic) +% 1, .seq_cst);
        if (box.parked.load(.seq_cst) == 1) box.io.futexWake(u32, &box.seq.raw, 1);
    }
    run(ctx, 0);
    for (mailboxes[0..workers]) |*box| {
        const want = box.seq.load(.monotonic);
        while (box.done.load(.acquire) != want) std.atomic.spinLoopHint();
    }
}

fn laneCount(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(T) orelse 1;
}

fn roundUp(x: usize, to: usize) usize {
    return (x + to - 1) / to * to;
}

const TileDims = struct { kc: usize, nc: usize };

fn tiledDims(comptime T: type, comptime cfg: TileConfig, k: usize, n: usize) TileDims {
    return .{
        .kc = @min(cfg.kc, k),
        .nc = @min(cfg.nc, roundUp(n, cfg.nv * laneCount(T))),
    };
}

const Split = struct { by_rows: bool, parts: usize, chunk: usize };

pub fn parallelFor(par: Parallel, len: usize, min_chunk: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize, usize) void) void {
    const parts = if (par.io == null) 1 else @max(1, @min(par.threads, max_workers + 1, len / min_chunk));
    if (parts == 1) return body(ctx, 0, len);

    const Job = struct { ctx: @TypeOf(ctx), len: usize, chunk: usize };
    const Task = struct {
        fn run(raw: *const anyopaque, p: usize) void {
            const j: *const Job = @ptrCast(@alignCast(raw));
            const lo = @min(p * j.chunk, j.len);
            body(j.ctx, lo, @min(lo + j.chunk, j.len));
        }
    };
    const job: Job = .{ .ctx = ctx, .len = len, .chunk = (len + parts - 1) / parts };
    parallelRun(par, parts, &job, Task.run);
}

fn splitFor(comptime T: type, comptime cfg: TileConfig, par: Parallel, m: usize, k: usize, n: usize) Split {
    const by_rows = m >= n;
    const unit = if (by_rows) cfg.mr else cfg.nv * laneCount(T);
    const tiles = ((if (by_rows) m else n) + unit - 1) / unit;
    const by_work = 2 * m * k * n / min_flops_per_part;
    const want = if (par.io == null) 1 else @max(1, @min(par.threads, max_workers + 1, by_work));
    const per = (tiles + want - 1) / want;
    return .{ .by_rows = by_rows, .chunk = per * unit, .parts = (tiles + per - 1) / per };
}

fn partWorkLen(comptime T: type, comptime cfg: TileConfig, s: Split, m: usize, k: usize, n: usize) usize {
    _ = m;
    const d = tiledDims(T, cfg, k, if (s.by_rows) n else @min(n, s.chunk));
    return d.kc * d.nc;
}

pub fn tiledWorkLenFor(comptime T: type, comptime cfg: TileConfig, par: Parallel, m: usize, k: usize, n: usize) usize {
    const s = splitFor(T, cfg, par, m, k, n);
    return s.parts * partWorkLen(T, cfg, s, m, k, n);
}

fn packB(
    comptime T: type,
    comptime NR: usize,
    comptime trans: bool,
    b: []const T,
    k: usize,
    n: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    dst: []T,
) void {
    var jr: usize = 0;
    while (jr < nc) : (jr += NR) {
        const strip = dst[(jr / NR) * kc * NR ..];
        const cols = @min(NR, nc - jr);
        if (trans) {
            for (0..NR) |c| {
                if (c < cols) {
                    const src = b[(jc + jr + c) * k + pc ..][0..kc];
                    for (src, 0..) |x, p| strip[p * NR + c] = x;
                } else {
                    for (0..kc) |p| strip[p * NR + c] = 0;
                }
            }
        } else {
            for (0..kc) |p| {
                const src = b[(pc + p) * n + jc + jr ..];
                const row = strip[p * NR ..][0..NR];
                if (cols == NR) {
                    row.* = src[0..NR].*;
                } else {
                    for (0..cols) |c| row[c] = src[c];
                    for (cols..NR) |c| row[c] = 0;
                }
            }
        }
    }
}

inline fn microTile(
    comptime T: type,
    comptime MR: usize,
    comptime NV: usize,
    kc: usize,
    a: []const T,
    a_off: [MR]usize,
    a_step: usize,
    bp: []const T,
) [MR][NV]@Vector(laneCount(T), T) {
    const lanes = comptime laneCount(T);
    const Vec = @Vector(lanes, T);
    const NR = NV * lanes;

    var acc: [MR][NV]Vec = @splat(@splat(@splat(0)));
    for (0..kc) |p| {
        var bv: [NV]Vec = undefined;
        inline for (0..NV) |v| {
            bv[v] = bp[p * NR + v * lanes ..][0..lanes].*;
        }
        inline for (0..MR) |r| {
            const av: Vec = @splat(a[a_off[r] + p * a_step]);
            inline for (0..NV) |v| {
                acc[r][v] = @mulAdd(Vec, av, bv[v], acc[r][v]);
            }
        }
    }
    return acc;
}

pub fn tiledWith(
    comptime T: type,
    comptime cfg: TileConfig,
    par: Parallel,
    m: usize,
    k: usize,
    n: usize,
    a: []const T,
    b: []const T,
    out: []T,
    work: []T,
) void {
    const s = splitFor(T, cfg, par, m, k, n);
    const part_len = partWorkLen(T, cfg, s, m, k, n);
    std.debug.assert(work.len >= s.parts * part_len);
    if (s.parts == 1) return tiledRange(T, cfg, m, k, n, a, b, out, 0, m, 0, n, work);

    const Ctx = struct { s: Split, part_len: usize, m: usize, k: usize, n: usize, a: []const T, b: []const T, out: []T, work: []T };
    const Task = struct {
        fn run(raw: *const anyopaque, p: usize) void {
            const c: *const Ctx = @ptrCast(@alignCast(raw));
            const lo = p * c.s.chunk;
            const w = c.work[p * c.part_len ..][0..c.part_len];
            if (c.s.by_rows) {
                tiledRange(T, cfg, c.m, c.k, c.n, c.a, c.b, c.out, lo, @min(lo + c.s.chunk, c.m), 0, c.n, w);
            } else {
                tiledRange(T, cfg, c.m, c.k, c.n, c.a, c.b, c.out, 0, c.m, lo, @min(lo + c.s.chunk, c.n), w);
            }
        }
    };
    const ctx: Ctx = .{ .s = s, .part_len = part_len, .m = m, .k = k, .n = n, .a = a, .b = b, .out = out, .work = work };
    parallelRun(par, s.parts, &ctx, Task.run);
}

fn tiledRange(
    comptime T: type,
    comptime cfg: TileConfig,
    m: usize,
    k: usize,
    n: usize,
    a: []const T,
    b: []const T,
    out: []T,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    work: []T,
) void {
    const MR = cfg.mr;
    const NV = cfg.nv;
    const MC = cfg.mc;
    const KC = cfg.kc;
    const NC = cfg.nc;
    const lanes = comptime laneCount(T);
    const Vec = @Vector(lanes, T);
    const NR = NV * lanes;
    comptime {
        if (MC % MR != 0) @compileError("tiled: mc must be a multiple of mr");
        if (NC % NR != 0) @compileError("tiled: nc must be a multiple of nv * lanes");
    }
    std.debug.assert(a.len >= m * k and b.len >= k * n and out.len >= m * n);
    const d = tiledDims(T, cfg, k, c1 - c0);
    std.debug.assert(work.len >= d.kc * d.nc);
    const b_pack = work[0 .. d.kc * d.nc];

    if (!cfg.accumulate) {
        for (r0..r1) |r| @memset(out[r * n + c0 .. r * n + c1], 0);
    }

    var jc: usize = c0;
    while (jc < c1) : (jc += NC) {
        const nc = @min(NC, c1 - jc);
        var pc: usize = 0;
        while (pc < k) : (pc += KC) {
            const kc = @min(KC, k - pc);
            packB(T, NR, cfg.trans_b, b, k, n, pc, jc, kc, nc, b_pack);

            var ic: usize = r0;
            while (ic < r1) : (ic += MC) {
                const mc = @min(MC, r1 - ic);

                var jr: usize = 0;
                while (jr < nc) : (jr += NR) {
                    const nr_eff = @min(NR, nc - jr);
                    const bp = b_pack[(jr / NR) * kc * NR ..];

                    var ir: usize = 0;
                    while (ir < mc) : (ir += MR) {
                        const mr_eff = @min(MR, mc - ir);
                        var a_off: [MR]usize = undefined;
                        inline for (0..MR) |r| {
                            const row = ic + ir + @min(r, mr_eff - 1);
                            a_off[r] = if (cfg.trans_a) pc * m + row else row * k + pc;
                        }
                        const acc = microTile(T, MR, NV, kc, a, a_off, if (cfg.trans_a) m else 1, bp);
                        const c = out[(ic + ir) * n + jc + jr ..];

                        if (mr_eff == MR and nr_eff == NR) {
                            inline for (0..MR) |r| {
                                inline for (0..NV) |v| {
                                    const dst = c[r * n + v * lanes ..][0..lanes];
                                    const old: Vec = dst.*;
                                    dst.* = old + acc[r][v];
                                }
                            }
                        } else {
                            var tile: [MR][NR]T = undefined;
                            inline for (0..MR) |r| {
                                inline for (0..NV) |v| {
                                    tile[r][v * lanes ..][0..lanes].* = acc[r][v];
                                }
                            }
                            for (0..mr_eff) |r| {
                                for (0..nr_eff) |cc| c[r * n + cc] += tile[r][cc];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn expectTiledCombos(
    comptime T: type,
    comptime base: TileConfig,
    par: Parallel,
    m: usize,
    k: usize,
    n: usize,
    a: []const T,
    b: []const T,
    at: []const T,
    bt: []const T,
    want: []const T,
    got: []T,
    work: []T,
) !void {
    inline for (.{ .{ false, false }, .{ true, false }, .{ false, true }, .{ true, true } }) |flags| {
        const cfg = comptime blk: {
            var c = base;
            c.trans_a = flags[0];
            c.trans_b = flags[1];
            break :blk c;
        };
        @memset(got, 99);
        tiledWith(T, cfg, par, m, k, n, if (flags[0]) at else a, if (flags[1]) bt else b, got, work);
        try std.testing.expectEqualSlices(T, want, got);
    }
}

fn expectSameResult(comptime T: type, par: Parallel, m: usize, k: usize, n: usize) !void {
    const gpa = std.testing.allocator;
    const a = try gpa.alloc(T, m * k);
    defer gpa.free(a);
    const b = try gpa.alloc(T, k * n);
    defer gpa.free(b);
    const want = try gpa.alloc(T, m * n);
    defer gpa.free(want);
    const got = try gpa.alloc(T, m * n);
    defer gpa.free(got);

    for (a, 0..) |*x, idx| x.* = @as(T, @floatFromInt(idx % 7)) - 3;
    for (b, 0..) |*x, idx| x.* = @as(T, @floatFromInt(idx % 5)) - 2;

    naive(T, m, k, n, a, b, want);

    @memset(got, 99);
    vec(T, m, k, n, a, b, got);
    try std.testing.expectEqualSlices(T, want, got);

    const at = try gpa.alloc(T, m * k);
    defer gpa.free(at);
    const bt = try gpa.alloc(T, k * n);
    defer gpa.free(bt);
    for (0..m) |i| {
        for (0..k) |p| at[p * m + i] = a[i * k + p];
    }
    for (0..k) |p| {
        for (0..n) |j| bt[j * k + p] = b[p * n + j];
    }

    const small: TileConfig = .{ .mr = 4, .nv = 2, .mc = 8, .kc = 5, .nc = 16 };
    const work_small = try gpa.alloc(T, tiledWorkLenFor(T, small, par, m, k, n));
    defer gpa.free(work_small);
    try expectTiledCombos(T, small, par, m, k, n, a, b, at, bt, want, got, work_small);

    const dflt: TileConfig = .{};
    const work_dflt = try gpa.alloc(T, tiledWorkLenFor(T, dflt, par, m, k, n));
    defer gpa.free(work_dflt);
    try expectTiledCombos(T, dflt, par, m, k, n, a, b, at, bt, want, got, work_dflt);
}

test "parallel tiled matmul matches in both split directions" {
    const par: Parallel = .{ .io = std.testing.io, .threads = 4 };
    const by_rows = splitFor(f32, .{}, par, 200, 193, 197);
    const by_cols = splitFor(f32, .{}, par, 150, 200, 260);
    try std.testing.expect(by_rows.by_rows and by_rows.parts > 1);
    try std.testing.expect(!by_cols.by_rows and by_cols.parts > 1);

    try expectSameResult(f32, par, 200, 193, 197);
    try expectSameResult(f32, par, 150, 200, 260);
    try expectSameResult(f64, par, 120, 150, 140);
}

test "flushDenormals makes subnormal results zero on the calling thread only" {
    if (@import("builtin").cpu.arch != .x86_64) return error.SkipZigTest;
    const Probe = struct {
        fn product(flush: bool, out: *f32) void {
            if (flush) flushDenormals();
            var a: f32 = 1e-30;
            var b: f32 = 1e-10;
            const pa: *volatile f32 = &a;
            const pb: *volatile f32 = &b;
            out.* = pa.* * pb.*;
        }
    };
    var plain: f32 = undefined;
    var flushed: f32 = undefined;
    const t1 = try std.Thread.spawn(.{}, Probe.product, .{ false, &plain });
    t1.join();
    const t2 = try std.Thread.spawn(.{}, Probe.product, .{ true, &flushed });
    t2.join();
    try std.testing.expect(plain != 0);
    try std.testing.expectEqual(@as(f32, 0), flushed);
}

test "worker pool survives many park and wake cycles" {
    const par: Parallel = .{ .io = std.testing.io, .threads = 6 };
    for (0..60) |i| try expectSameResult(f32, par, 110 + i % 7, 130, 120 + i % 5);
}

test "kernels agree, sizes not a multiple of the vector width" {
    const shapes = [_][3]usize{
        .{ 5, 7, 11 },   .{ 3, 1, 9 },   .{ 1, 4, 1 },  .{ 16, 16, 16 }, .{ 17, 33, 65 }, .{ 6, 5, 16 },
        .{ 12, 12, 32 }, .{ 13, 9, 40 }, .{ 7, 3, 17 }, .{ 24, 10, 48 }, .{ 50, 70, 90 },
    };
    for (shapes) |s| try expectSameResult(f32, .{}, s[0], s[1], s[2]);
    try expectSameResult(f64, .{}, 9, 5, 13);
    try expectSameResult(f64, .{}, 33, 21, 29);
}
