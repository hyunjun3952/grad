const std = @import("std");
const kernels = @import("kernels.zig");

pub fn zeroGrad(par: kernels.Parallel, params: anytype) void {
    inline for (params) |p| {
        const G = @TypeOf(p.grad.data);
        const body = struct {
            fn run(g: G, lo: usize, hi: usize) void {
                @memset(g[lo..hi], 0);
            }
        }.run;
        kernels.parallelFor(par, p.grad.data.len, 1 << 15, p.grad.data, body);
    }
}

pub fn sgd(par: kernels.Parallel, lr: f32, params: anytype) void {
    inline for (params) |p| {
        const E = @TypeOf(p.value).Elem;
        const Ctx = struct { w: []E, g: []const E, lr: E };
        const body = struct {
            fn run(c: Ctx, lo: usize, hi: usize) void {
                for (c.w[lo..hi], c.g[lo..hi]) |*w, g| w.* -= c.lr * g;
            }
        }.run;
        kernels.parallelFor(par, p.value.data.len, 1 << 14, Ctx{ .w = p.value.data, .g = p.grad.data, .lr = lr }, body);
    }
}

pub const Adam = struct {
    m: []f32,
    v: []f32,
    t: u32 = 0,
    lr: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,

    pub fn init(gpa: std.mem.Allocator, lr: f32, params: anytype) std.mem.Allocator.Error!Adam {
        var total: usize = 0;
        inline for (params) |p| total += p.value.data.len;
        const m = try gpa.alloc(f32, total);
        errdefer gpa.free(m);
        const v = try gpa.alloc(f32, total);
        @memset(m, 0);
        @memset(v, 0);
        return .{ .m = m, .v = v, .lr = lr };
    }

    pub fn deinit(self: Adam, gpa: std.mem.Allocator) void {
        gpa.free(self.m);
        gpa.free(self.v);
    }

    pub fn step(self: *Adam, par: kernels.Parallel, params: anytype) void {
        self.t += 1;
        const t: f64 = @floatFromInt(self.t);
        const c1 = 1 - std.math.pow(f64, self.beta1, t);
        const c2 = 1 - std.math.pow(f64, self.beta2, t);

        var off: usize = 0;
        inline for (params) |p| {
            const len = p.value.data.len;
            const Ctx = struct { w: []f32, g: []const f32, m: []f32, v: []f32, b1: f32, b2: f32, step: f32, inv_sqrt_c2: f32, eps: f32 };
            const body = struct {
                fn run(c: Ctx, lo: usize, hi: usize) void {
                    const lanes = std.simd.suggestVectorLength(f32) orelse 1;
                    var i = lo;
                    while (i + lanes <= hi) : (i += lanes) update(lanes, c, i);
                    while (i < hi) : (i += 1) update(1, c, i);
                }

                inline fn update(comptime n: usize, c: Ctx, i: usize) void {
                    const V = @Vector(n, f32);
                    const g: V = c.g[i..][0..n].*;
                    var m: V = c.m[i..][0..n].*;
                    var v: V = c.v[i..][0..n].*;
                    const b1: V = @splat(c.b1);
                    const b2: V = @splat(c.b2);
                    const one: V = @splat(1);
                    m = b1 * m + (one - b1) * g;
                    v = b2 * v + (one - b2) * g * g;
                    const scale: V = @splat(c.step);
                    const inv_sqrt_c2: V = @splat(c.inv_sqrt_c2);
                    const eps: V = @splat(c.eps);
                    const w: V = c.w[i..][0..n].*;
                    c.w[i..][0..n].* = w - scale * m / (@sqrt(v) * inv_sqrt_c2 + eps);
                    c.m[i..][0..n].* = m;
                    c.v[i..][0..n].* = v;
                }
            }.run;
            const ctx: Ctx = .{
                .w = p.value.data,
                .g = p.grad.data,
                .m = self.m[off..][0..len],
                .v = self.v[off..][0..len],
                .b1 = self.beta1,
                .b2 = self.beta2,
                .step = @floatCast(self.lr / c1),
                .inv_sqrt_c2 = @floatCast(1 / @sqrt(c2)),
                .eps = self.eps,
            };
            kernels.parallelFor(par, len, 1 << 14, ctx, body);
            off += len;
        }
    }
};

const testing = std.testing;
const Tensor = @import("tensor.zig").Tensor;
const Var = @import("autograd.zig").Var;

test "adam: first step moves by lr, converges on a quadratic" {
    const gpa = testing.allocator;
    const One = Tensor(f32, &.{1});
    const p: Var(One) = .{ .value = try One.fromSlice(gpa, &.{0}), .grad = try One.fromSlice(gpa, &.{1}), .needs_grad = true };
    defer p.value.deinit(gpa);
    defer p.grad.deinit(gpa);

    var first = try Adam.init(gpa, 0.1, .{p});
    defer first.deinit(gpa);
    first.step(.{}, .{p});
    try testing.expectApproxEqAbs(@as(f32, -0.1), p.value.data[0], 1e-6);

    p.value.data[0] = 0;
    var adam = try Adam.init(gpa, 0.05, .{p});
    defer adam.deinit(gpa);
    for (0..600) |_| {
        p.grad.data[0] = 2 * (p.value.data[0] - 3);
        adam.step(.{}, .{p});
    }
    try testing.expectApproxEqAbs(@as(f32, 3), p.value.data[0], 0.05);
}

test "adam gives identical weights with one thread and with worker threads" {
    const gpa = testing.allocator;
    const Big = Tensor(f32, &.{100003});
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();

    var single: Var(Big) = .{ .value = try Big.empty(gpa), .grad = try Big.empty(gpa), .needs_grad = true };
    defer single.value.deinit(gpa);
    defer single.grad.deinit(gpa);
    var multi: Var(Big) = .{ .value = try Big.empty(gpa), .grad = try Big.empty(gpa), .needs_grad = true };
    defer multi.value.deinit(gpa);
    defer multi.grad.deinit(gpa);
    for (single.value.data, multi.value.data) |*a, *b| {
        a.* = rng.float(f32);
        b.* = a.*;
    }

    var adam_single = try Adam.init(gpa, 0.01, .{single});
    defer adam_single.deinit(gpa);
    var adam_multi = try Adam.init(gpa, 0.01, .{multi});
    defer adam_multi.deinit(gpa);
    const par: kernels.Parallel = .{ .io = testing.io, .threads = 4 };

    for (0..5) |_| {
        for (single.grad.data, multi.grad.data) |*a, *b| {
            a.* = rng.float(f32) - 0.5;
            b.* = a.*;
        }
        adam_single.step(.{}, .{single});
        adam_multi.step(par, .{multi});
    }
    try testing.expectEqualSlices(f32, single.value.data, multi.value.data);
}

test "sgd moves weights against the gradient, also across parallel chunks, and zeroGrad clears" {
    const gpa = testing.allocator;
    const Small = Tensor(f32, &.{3});
    const Big = Tensor(f32, &.{100003});

    var small: Var(Small) = .{ .value = try Small.fromSlice(gpa, &.{ 1, 2, 3 }), .grad = try Small.fromSlice(gpa, &.{ 10, 20, 30 }), .needs_grad = true };
    defer small.value.deinit(gpa);
    defer small.grad.deinit(gpa);
    var big: Var(Big) = .{ .value = try Big.full(gpa, 1), .grad = try Big.empty(gpa), .needs_grad = true };
    defer big.value.deinit(gpa);
    defer big.grad.deinit(gpa);
    for (big.grad.data, 0..) |*g, i| g.* = @floatFromInt(i % 11);

    sgd(.{ .io = testing.io, .threads = 4 }, 0.1, .{ small, big });

    try testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, small.value.data);
    for (big.value.data, 0..) |w, i| {
        try testing.expectApproxEqAbs(1.0 - 0.1 * @as(f32, @floatFromInt(i % 11)), w, 1e-6);
    }

    zeroGrad(.{ .io = testing.io, .threads = 4 }, .{ small, big });
    try testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, small.grad.data);
    for (big.grad.data) |g| try testing.expectEqual(@as(f32, 0), g);
}
