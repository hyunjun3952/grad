const std = @import("std");

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

pub fn blockedWith(
    comptime T: type,
    comptime MR: usize,
    comptime NV: usize,
    m: usize,
    k: usize,
    n: usize,
    a: []const T,
    b: []const T,
    out: []T,
) void {
    std.debug.assert(a.len >= m * k and b.len >= k * n and out.len >= m * n);
    const lanes = comptime std.simd.suggestVectorLength(T) orelse 1;
    const Vec = @Vector(lanes, T);
    const NR = NV * lanes;

    var i: usize = 0;
    while (i + MR <= m) : (i += MR) {
        var j: usize = 0;
        while (j + NR <= n) : (j += NR) {
            var acc: [MR][NV]Vec = @splat(@splat(@splat(0)));
            for (0..k) |p| {
                var bv: [NV]Vec = undefined;
                inline for (0..NV) |v| {
                    bv[v] = b[p * n + j + v * lanes ..][0..lanes].*;
                }
                inline for (0..MR) |r| {
                    const av: Vec = @splat(a[(i + r) * k + p]);
                    inline for (0..NV) |v| {
                        acc[r][v] = @mulAdd(Vec, av, bv[v], acc[r][v]);
                    }
                }
            }
            inline for (0..MR) |r| {
                inline for (0..NV) |v| {
                    out[(i + r) * n + j + v * lanes ..][0..lanes].* = acc[r][v];
                }
            }
        }
        for (i..i + MR) |row| {
            for (j..n) |col| {
                var s: T = 0;
                for (0..k) |p| s += a[row * k + p] * b[p * n + col];
                out[row * n + col] = s;
            }
        }
    }
    if (i < m) vec(T, m - i, k, n, a[i * k ..], b, out[i * n ..]);
}

pub const TileConfig = struct {
    mr: usize = 6,
    nv: usize = 2,
    mc: usize = 96,
    kc: usize = 256,
    nc: usize = 1024,
    trans_a: bool = false,
    trans_b: bool = false,
};

fn laneCount(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(T) orelse 1;
}

fn roundUp(x: usize, to: usize) usize {
    return (x + to - 1) / to * to;
}

const TileDims = struct { mc: usize, kc: usize, nc: usize };

fn tiledDims(comptime T: type, comptime cfg: TileConfig, m: usize, k: usize, n: usize) TileDims {
    const NR = cfg.nv * laneCount(T);
    return .{
        .mc = @min(cfg.mc, roundUp(m, cfg.mr)),
        .kc = @min(cfg.kc, k),
        .nc = @min(cfg.nc, roundUp(n, NR)),
    };
}

pub fn tiledWorkLenFor(comptime T: type, comptime cfg: TileConfig, m: usize, k: usize, n: usize) usize {
    const d = tiledDims(T, cfg, m, k, n);
    return d.mc * d.kc + d.kc * d.nc;
}

fn packA(
    comptime T: type,
    comptime MR: usize,
    comptime trans: bool,
    a: []const T,
    m: usize,
    k: usize,
    ic: usize,
    pc: usize,
    mc: usize,
    kc: usize,
    dst: []T,
) void {
    var ir: usize = 0;
    while (ir < mc) : (ir += MR) {
        const strip = dst[(ir / MR) * kc * MR ..];
        const rows = @min(MR, mc - ir);
        if (trans) {
            for (0..kc) |p| {
                const src = a[(pc + p) * m + ic + ir ..];
                for (0..MR) |r| strip[p * MR + r] = if (r < rows) src[r] else 0;
            }
        } else {
            for (0..MR) |r| {
                if (r < rows) {
                    const src = a[(ic + ir + r) * k + pc ..][0..kc];
                    for (src, 0..) |x, p| strip[p * MR + r] = x;
                } else {
                    for (0..kc) |p| strip[p * MR + r] = 0;
                }
            }
        }
    }
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
    ap: []const T,
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
            const av: Vec = @splat(ap[p * MR + r]);
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
    m: usize,
    k: usize,
    n: usize,
    a: []const T,
    b: []const T,
    out: []T,
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
    const d = tiledDims(T, cfg, m, k, n);
    std.debug.assert(work.len >= d.mc * d.kc + d.kc * d.nc);

    const a_pack = work[0 .. d.mc * d.kc];
    const b_pack = work[d.mc * d.kc ..][0 .. d.kc * d.nc];

    @memset(out[0 .. m * n], 0);

    var jc: usize = 0;
    while (jc < n) : (jc += NC) {
        const nc = @min(NC, n - jc);
        var pc: usize = 0;
        while (pc < k) : (pc += KC) {
            const kc = @min(KC, k - pc);
            packB(T, NR, cfg.trans_b, b, k, n, pc, jc, kc, nc, b_pack);

            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc = @min(MC, m - ic);
                packA(T, MR, cfg.trans_a, a, m, k, ic, pc, mc, kc, a_pack);

                var jr: usize = 0;
                while (jr < nc) : (jr += NR) {
                    const nr_eff = @min(NR, nc - jr);
                    const bp = b_pack[(jr / NR) * kc * NR ..];

                    var ir: usize = 0;
                    while (ir < mc) : (ir += MR) {
                        const mr_eff = @min(MR, mc - ir);
                        const ap = a_pack[(ir / MR) * kc * MR ..];
                        const acc = microTile(T, MR, NV, kc, ap, bp);
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
        tiledWith(T, cfg, m, k, n, if (flags[0]) at else a, if (flags[1]) bt else b, got, work);
        try std.testing.expectEqualSlices(T, want, got);
    }
}

fn expectSameResult(comptime T: type, m: usize, k: usize, n: usize) !void {
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

    @memset(got, 99);
    blockedWith(T, 4, 2, m, k, n, a, b, got);
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
    const work_small = try gpa.alloc(T, tiledWorkLenFor(T, small, m, k, n));
    defer gpa.free(work_small);
    try expectTiledCombos(T, small, m, k, n, a, b, at, bt, want, got, work_small);

    const dflt: TileConfig = .{};
    const work_dflt = try gpa.alloc(T, tiledWorkLenFor(T, dflt, m, k, n));
    defer gpa.free(work_dflt);
    try expectTiledCombos(T, dflt, m, k, n, a, b, at, bt, want, got, work_dflt);
}

test "kernels agree, sizes not a multiple of the vector width" {
    try expectSameResult(f32, 5, 7, 11);
    try expectSameResult(f32, 3, 1, 9);
    try expectSameResult(f32, 1, 4, 1);
    try expectSameResult(f32, 16, 16, 16);
    try expectSameResult(f32, 17, 33, 65);
    try expectSameResult(f64, 9, 5, 13);
    try expectSameResult(f32, 6, 5, 16);
    try expectSameResult(f32, 12, 12, 32);
    try expectSameResult(f32, 13, 9, 40);
    try expectSameResult(f32, 7, 3, 17);
    try expectSameResult(f32, 24, 10, 48);
    try expectSameResult(f32, 50, 70, 90);
    try expectSameResult(f64, 33, 21, 29);
}
