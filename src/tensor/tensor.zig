const std = @import("std");
const Allocator = std.mem.Allocator;
const kernels = @import("../cpu/kernels.zig");

pub fn Tensor(comptime T: type, comptime dims: []const usize) type {
    comptime {
        for (dims) |d| {
            if (d == 0) @compileError("zero-sized dimensions are not supported");
        }
    }

    return struct {
        const Self = @This();

        pub const Elem = T;
        pub const shape = dims;
        pub const rank = dims.len;
        pub const len = blk: {
            var n: usize = 1;
            for (dims) |d| n *= d;
            break :blk n;
        };

        data: []T,

        pub fn empty(gpa: Allocator) Allocator.Error!Self {
            return .{ .data = try gpa.alloc(T, len) };
        }

        pub fn zeros(gpa: Allocator) Allocator.Error!Self {
            const t = try empty(gpa);
            @memset(t.data, 0);
            return t;
        }

        pub fn full(gpa: Allocator, value: T) Allocator.Error!Self {
            const t = try empty(gpa);
            @memset(t.data, value);
            return t;
        }

        pub fn fromSlice(gpa: Allocator, values: []const T) Allocator.Error!Self {
            std.debug.assert(values.len == len);
            const t = try empty(gpa);
            @memcpy(t.data, values);
            return t;
        }

        pub fn clone(self: Self, gpa: Allocator) Allocator.Error!Self {
            return fromSlice(gpa, self.data);
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            gpa.free(self.data);
        }
    };
}

pub const MatmulOpts = @import("../gpu/gpu.zig").MatmulOpts;

pub fn MatmulResult(comptime A: type, comptime B: type, comptime opts: MatmulOpts) type {
    if (A.rank != 2 or B.rank != 2) {
        @compileError("matmul requires rank-2 tensors");
    }
    if (A.Elem != B.Elem and !(A.Elem == u16 and B.Elem == f32)) {
        @compileError("matmul element types differ; only bf16 activations times f32 weights may be mixed");
    }
    const m = if (opts.trans_a) A.shape[1] else A.shape[0];
    const ka = if (opts.trans_a) A.shape[0] else A.shape[1];
    const kb = if (opts.trans_b) B.shape[1] else B.shape[0];
    const n = if (opts.trans_b) B.shape[0] else B.shape[1];
    if (ka != kb) {
        @compileError(std.fmt.comptimePrint(
            "matmul shape mismatch: ({d}x{d}) * ({d}x{d})",
            .{ m, ka, kb, n },
        ));
    }
    return Tensor(A.Elem, &.{ m, n });
}

pub fn matmulInto(
    gpa: Allocator,
    par: kernels.Parallel,
    a: anytype,
    b: anytype,
    out: anytype,
    comptime opts: MatmulOpts,
) !void {
    const A = @TypeOf(a);
    comptime {
        if (!std.mem.eql(usize, @TypeOf(out).shape, MatmulResult(A, @TypeOf(b), opts).shape)) {
            @compileError("matmulInto: out does not have the product's shape");
        }
    }
    const M = @TypeOf(out).shape[0];
    const N = @TypeOf(out).shape[1];
    const K = if (opts.trans_a) A.shape[0] else A.shape[1];

    if (par.gpu) |g| return g.matmul(a.data, b.data, out.data, .{ .m = M, .n = N, .k = K }, opts);
    if (comptime A.Elem == u16 or @TypeOf(b).Elem == u16 or @TypeOf(out).Elem == u16) return error.NoDevice;

    const cfg: kernels.TileConfig = .{ .trans_a = opts.trans_a, .trans_b = opts.trans_b, .accumulate = opts.accumulate };
    const work = try gpa.alloc(A.Elem, kernels.matmulTiledWorkLen(A.Elem, cfg, par, M, K, N));
    defer gpa.free(work);
    kernels.matmulTiled(A.Elem, cfg, par, .{ .m = M, .k = K, .n = N, .a = a.data, .b = b.data, .out = out.data }, work);
}

pub fn matmul(
    gpa: Allocator,
    par: kernels.Parallel,
    a: anytype,
    b: anytype,
    comptime opts: MatmulOpts,
) !MatmulResult(@TypeOf(a), @TypeOf(b), opts) {
    comptime {
        if (opts.accumulate) @compileError("matmul returns a fresh tensor; use matmulInto to accumulate");
    }
    std.debug.assert(par.gpu == null);
    const out = try MatmulResult(@TypeOf(a), @TypeOf(b), opts).empty(gpa);
    errdefer out.deinit(gpa);
    try matmulInto(gpa, par, a, b, out, opts);
    return out;
}
