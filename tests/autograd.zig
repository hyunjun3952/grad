const std = @import("std");
const lib = @import("grad");
const PoolOut = lib.autograd.PoolOut;
const Allocator = std.mem.Allocator;
const ConvOut = lib.autograd.ConvOut;
const bf16_math = lib.math.bf16;
const drop = lib.math.dropout;
const cpu = lib.cpu_ops;
const kernels = lib.kernels;
const Tape = lib.autograd.Tape;
const Var = lib.autograd.Var;
const tensor = lib.tensor;
const Tensor = tensor.Tensor;
const reshape = lib.autograd.reshape;
const ops = lib.ops;
const norm = lib.math.norm;
const ConvOpts = lib.autograd.ConvOpts;
const PoolOpts = lib.autograd.PoolOpts;
const gpu_mod = lib.gpu;

const testing = std.testing;

test "crossEntropy of equal logits, and gradient fan-out accumulates" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator() };

    const x = try t.leaf(try Tensor(f32, &.{ 1, 2 }).zeros(t.gpa));
    const logits = try t.add(x, x);
    const loss = try t.crossEntropy(logits, &.{0});
    try t.backward(loss);

    try testing.expectApproxEqAbs(@as(f32, 0.6931472), loss.value.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.5), logits.grad.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), logits.grad.data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1), x.grad.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), x.grad.data[1], 1e-6);
}

fn bigGradients(gpa: Allocator, threads: usize, x: Tensor(f32, &.{ 64, 784 }), w1: Tensor(f32, &.{ 784, 128 }), w2: Tensor(f32, &.{ 128, 10 }), labels: []const u8) !struct { loss: f32, g1: []f32, g2: []f32 } {
    var t: Tape = .{ .gpa = gpa, .par = if (threads > 1) .{ .io = testing.io, .threads = threads } else .{} };
    const vx = try t.constant(x);
    const vw1 = try t.leaf(w1);
    const vw2 = try t.leaf(w2);
    const loss = try t.crossEntropy(try t.matmul(try t.relu(try t.matmul(vx, vw1)), vw2), labels);
    try t.backward(loss);
    return .{ .loss = loss.value.data[0], .g1 = vw1.grad.data, .g2 = vw2.grad.data };
}

test "gradients are bit-identical with one thread and with worker threads" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const x = try Tensor(f32, &.{ 64, 784 }).empty(gpa);
    const w1 = try Tensor(f32, &.{ 784, 128 }).empty(gpa);
    const w2 = try Tensor(f32, &.{ 128, 10 }).empty(gpa);
    for (x.data) |*v| v.* = rng.float(f32);
    for (w1.data) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    for (w2.data) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    var labels: [64]u8 = undefined;
    for (&labels) |*l| l.* = rng.uintLessThan(u8, 10);

    const single = try bigGradients(gpa, 1, x, w1, w2, &labels);
    const multi = try bigGradients(gpa, 4, x, w1, w2, &labels);
    try testing.expectEqual(single.loss, multi.loss);
    try testing.expectEqualSlices(f32, single.g1, multi.g1);
    try testing.expectEqualSlices(f32, single.g2, multi.g2);
}

test "a constant input is skipped and parameter gradients stay identical" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    var p: SmallNet.Inputs = undefined;
    inline for (.{ "x", "w1", "b1", "s", "w2" }) |name| {
        @field(p, name) = try @FieldType(SmallNet.Inputs, name).empty(gpa);
        for (@field(p, name).data) |*v| v.* = rng.float(f32) * 2 - 1;
    }

    var with_x: Tape = .{ .gpa = gpa };
    const a = try SmallNet.forward(&with_x, p, true, &SmallNet.labels);
    try with_x.backward(a.loss);
    var without_x: Tape = .{ .gpa = gpa };
    const b = try SmallNet.forward(&without_x, p, false, &SmallNet.labels);
    try without_x.backward(b.loss);

    try testing.expect(!b.x.needs_grad);
    try testing.expect(without_x.nodes.items.len == with_x.nodes.items.len);
    inline for (.{ "w1", "b1", "s", "w2" }) |name| {
        try testing.expectEqualSlices(f32, @field(a, name).grad.data, @field(b, name).grad.data);
    }
}

test "dropout is the identity at eval, and backward reuses the forward mask" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const Row = Tensor(f32, &.{ 1, 1000 });
    var t: Tape = .{ .gpa = arena.allocator() };
    const x = try t.leaf(try Row.full(t.gpa, 1));
    try testing.expectEqual(x.value.data.ptr, (try t.dropout(x, 0.8)).value.data.ptr);

    t.training = true;
    const y = try t.dropout(x, 0.8);
    @memset(y.grad.data, 1);
    try t.nodes.items[0].run(&t, t.nodes.items[0].ctx);
    var kept: usize = 0;
    for (y.value.data, x.grad.data) |v, g| {
        try testing.expectEqual(v, g);
        kept += @intFromBool(v != 0);
    }
    try testing.expect(kept > 700 and kept < 900);
}

test "gpu dropout draws the same mask as the cpu, with the seed split between the tape and the device step seed" {
    const gpu = try gpu_mod.Gpu.forTest(1 << 20);
    defer gpu.deinit();
    // NOTE: a recorded training step keeps part of the seed on the device; tape seed 1 plus step seed 2 must equal seed 3.
    try gpu.setStepSeed(2);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const Row = Tensor(f32, &.{ 1, 1000 });

    var cpu_tape: Tape = .{ .gpa = arena.allocator(), .training = true, .seed = 3 };
    const cpu_x = try cpu_tape.leaf(try Row.full(cpu_tape.gpa, 1));
    const want = try cpu_tape.dropout(cpu_x, 0.7);
    @memset(want.grad.data, 1);
    try cpu_tape.nodes.items[0].run(&cpu_tape, cpu_tape.nodes.items[0].ctx);

    const ones = try gpu.alloc(f32, Row.len);
    try gpu.fill(ones, 1);
    var gpu_tape: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true, .seed = 1 };
    const grad = try gpu.alloc(f32, Row.len);
    try gpu.fill(grad, 0);
    const gpu_x: Var(Row) = .{ .value = .{ .data = ones }, .grad = .{ .data = grad }, .needs_grad = true };
    const got = try gpu_tape.dropout(gpu_x, 0.7);
    try gpu.fill(got.grad.data, 1);
    try gpu_tape.nodes.items[0].run(&gpu_tape, gpu_tape.nodes.items[0].ctx);

    const back = try arena.allocator().alloc(f32, Row.len);
    try gpu.download(back, got.value.data);
    try testing.expectEqualSlices(f32, want.value.data, back);
    try gpu.download(back, grad);
    try testing.expectEqualSlices(f32, cpu_x.grad.data, back);
}

test "batch norm at eval time uses the running statistics" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const p = try gradcheck.randomInputs(BnCase, gpa, 42);
    const running_mean = try BnCase.Channels.fromSlice(gpa, &.{ 0.5, -0.5, 0 });
    const running_var = try BnCase.Channels.fromSlice(gpa, &.{ 4, 1, 0.25 });

    var t: Tape = .{ .gpa = gpa };
    const x = try t.constant(p.x);
    const y = try t.batchNorm(x, try t.constant(p.gamma), try t.constant(p.beta), .{ .mean = running_mean, .variance = running_var });
    for (0..6) |r| {
        for (0..3) |ch| {
            const want = p.gamma.data[ch] * (p.x.data[r * 3 + ch] - running_mean.data[ch]) / @sqrt(running_var.data[ch] + 1e-5) + p.beta.data[ch];
            try testing.expectApproxEqAbs(want, y.value.data[r * 3 + ch], 1e-5);
        }
    }
    try testing.expectEqualSlices(f32, &.{ 0.5, -0.5, 0 }, running_mean.data);
}

const gradcheck = @import("gradcheck.zig");

const SmallNet = struct {
    const X = Tensor(f32, &.{ 4, 3 });
    const W1 = Tensor(f32, &.{ 3, 5 });
    const B1 = Tensor(f32, &.{5});
    const S = Tensor(f32, &.{ 1, 5 });
    const W2 = Tensor(f32, &.{ 5, 3 });

    pub const Inputs = struct { x: X, w1: W1, b1: B1, s: S, w2: W2 };
    pub const names = .{ "x", "w1", "b1", "s", "w2" };
    pub const labels = [_]u8{ 0, 2, 1, 2 };
    const Result = struct {
        loss: Var(Tensor(f32, &.{})),
        x: Var(X),
        w1: Var(W1),
        b1: Var(B1),
        s: Var(S),
        w2: Var(W2),
    };

    fn forward(t: *Tape, p: Inputs, x_grad: bool, label_slice: []const u8) !Result {
        const x = if (x_grad) try t.leaf(p.x) else try t.constant(p.x);
        const w1 = try t.leaf(p.w1);
        const b1 = try t.leaf(p.b1);
        const s = try t.leaf(p.s);
        const w2 = try t.leaf(p.w2);

        const h = try t.relu(try t.add(try t.matmul(x, w1), b1));
        const logits = try t.matmul(try t.mul(h, s), w2);
        const loss = try t.crossEntropy(logits, label_slice);
        return .{ .loss = loss, .x = x, .w1 = w1, .b1 = b1, .s = s, .w2 = w2 };
    }

    pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !Result {
        return forward(t, p, true, label_slice);
    }

    pub fn acceptable(_: Allocator, p: Inputs) !bool {
        var smallest: f32 = std.math.floatMax(f32);
        for (0..4) |r| {
            for (0..5) |c| {
                var pre = p.b1.data[c];
                for (0..3) |k| pre += p.x.data[r * 3 + k] * p.w1.data[k * 5 + c];
                smallest = @min(smallest, @abs(pre));
            }
        }
        return smallest > 0.05;
    }
};

const MseNet = struct {
    const X = Tensor(f32, &.{ 4, 3 });
    const W = Tensor(f32, &.{ 3, 2 });
    const Y = Tensor(f32, &.{ 4, 2 });

    pub const Inputs = struct { x: X, w: W, y: Y };
    pub const names = .{ "x", "w", "y" };
    pub const labels = [_]u8{ 0, 0, 0, 0 };
    const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), w: Var(W), y: Var(Y) };

    pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
        const x = try t.leaf(p.x);
        const w = try t.leaf(p.w);
        const y = try t.leaf(p.y);
        return .{ .loss = try t.mse(try t.matmul(x, w), y), .x = x, .w = w, .y = y };
    }

    pub fn acceptable(_: Allocator, _: Inputs) !bool {
        return true;
    }
};

const ActivationNet = struct {
    const X = Tensor(f32, &.{ 4, 3 });
    const W = Tensor(f32, &.{ 3, 5 });
    const Y = Tensor(f32, &.{ 4, 5 });

    pub const Inputs = struct { x: X, w: W, y: Y };
    pub const names = .{ "x", "w", "y" };
    pub const labels = [_]u8{ 0, 0, 0, 0 };
    const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), w: Var(W), y: Var(Y) };

    pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
        const x = try t.leaf(p.x);
        const w = try t.leaf(p.w);
        const y = try t.leaf(p.y);
        const h = try t.activate(try t.activate(try t.activate(try t.matmul(x, w), .gelu), .tanh), .sigmoid);
        return .{ .loss = try t.mse(h, y), .x = x, .w = w, .y = y };
    }

    pub fn acceptable(_: Allocator, _: Inputs) !bool {
        return true;
    }
};

fn PoolCase(comptime opts: PoolOpts) type {
    return struct {
        const X = Tensor(f32, &.{ 2, 7, 6, 3 });
        const Pooled = PoolOut(X, opts);
        const Y = Tensor(f32, &.{ 2, Pooled.shape[1] * Pooled.shape[2] * 3 });

        pub const Inputs = struct { x: X, y: Y };
        pub const names = .{ "x", "y" };
        pub const labels = [_]u8{ 0, 0 };
        const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), y: Var(Y) };

        pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
            const x = try t.leaf(p.x);
            const y = try t.leaf(p.y);
            const flat = reshape(try t.add(try t.maxPool(x, opts), try t.avgPool(x, opts)), &.{ 2, Y.shape[1] });
            return .{ .loss = try t.mse(flat, y), .x = x, .y = y };
        }

        pub fn acceptable(_: Allocator, _: Inputs) !bool {
            return true;
        }
    };
}

const HighRankNet = struct {
    const A = Tensor(f32, &.{ 2, 1, 3, 1, 2, 1 });
    const B = Tensor(f32, &.{ 1, 2, 1, 2, 1, 3 });
    const Y = Tensor(f32, &.{ 12, 12 });

    pub const Inputs = struct { a: A, b: B, y: Y };
    pub const names = .{ "a", "b", "y" };
    pub const labels: [12]u8 = @splat(0);
    const Result = struct { loss: Var(Tensor(f32, &.{})), a: Var(A), b: Var(B), y: Var(Y) };

    pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
        const a = try t.leaf(p.a);
        const b = try t.leaf(p.b);
        const y = try t.leaf(p.y);
        const sum = try t.add(a, b);
        const product = try t.mul(a, b);
        const mixed = try t.add(sum, product);
        return .{ .loss = try t.mse(reshape(mixed, &.{ 12, 12 }), y), .a = a, .b = b, .y = y };
    }

    pub fn acceptable(_: Allocator, _: Inputs) !bool {
        return true;
    }
};

const WideConvNet = struct {
    const X = Tensor(f32, &.{ 64, 16, 16, 64 });
    const W = Tensor(f32, &.{ 576, 128 });
    const Y = Tensor(f32, &.{ 64, 16 * 16 * 128 });

    pub const Inputs = struct { x: X, w: W, y: Y };
    pub const names = .{ "x", "w", "y" };
    pub const labels: [64]u8 = @splat(0);
    const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), w: Var(W), y: Var(Y) };

    pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
        const x = try t.leaf(p.x);
        const w = try t.leaf(p.w);
        const y = try t.leaf(p.y);
        return .{ .loss = try t.mse(reshape(try t.conv2d(x, w, .{ .pad = 1 }), &.{ 64, 16 * 16 * 128 }), y), .x = x, .w = w, .y = y };
    }

    pub fn acceptable(_: Allocator, _: Inputs) !bool {
        return true;
    }
};

const DilatedConvNet = struct {
    const X = Tensor(f32, &.{ 4, 16, 16, 64 });
    const W = Tensor(f32, &.{ 576, 32 });
    const Y = Tensor(f32, &.{ 4, 16 * 16 * 32 });

    pub const Inputs = struct { x: X, w: W, y: Y };
    pub const names = .{ "x", "w", "y" };
    pub const labels: [4]u8 = @splat(0);
    const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), w: Var(W), y: Var(Y) };

    pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
        const x = try t.leaf(p.x);
        const w = try t.leaf(p.w);
        const y = try t.leaf(p.y);
        return .{ .loss = try t.mse(reshape(try t.conv2d(x, w, .{ .pad = 2, .dilation = 2 }), &.{ 4, 16 * 16 * 32 }), y), .x = x, .w = w, .y = y };
    }

    pub fn acceptable(_: Allocator, _: Inputs) !bool {
        return true;
    }
};

const GroupedConvNet = struct {
    const X = Tensor(f32, &.{ 8, 16, 16, 64 });
    const W = Tensor(f32, &.{ 144, 64 });
    const Y = Tensor(f32, &.{ 8, 16 * 16 * 64 });

    pub const Inputs = struct { x: X, w: W, y: Y };
    pub const names = .{ "x", "w", "y" };
    pub const labels: [8]u8 = @splat(0);
    const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), w: Var(W), y: Var(Y) };

    pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
        const x = try t.leaf(p.x);
        const w = try t.leaf(p.w);
        const y = try t.leaf(p.y);
        return .{ .loss = try t.mse(reshape(try t.conv2d(x, w, .{ .pad = 1, .groups = 4 }), &.{ 8, 16 * 16 * 64 }), y), .x = x, .w = w, .y = y };
    }

    pub fn acceptable(_: Allocator, _: Inputs) !bool {
        return true;
    }
};

fn LayerNormNet(comptime rows: usize, comptime cols: usize) type {
    return struct {
        const X = Tensor(f32, &.{ rows, cols });
        const C = Tensor(f32, &.{cols});

        pub const Inputs = struct { x: X, gamma: C, beta: C, y: X };
        pub const names = .{ "x", "gamma", "beta", "y" };
        pub const labels: [rows]u8 = @splat(0);
        const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), gamma: Var(C), beta: Var(C), y: Var(X) };

        pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
            const x = try t.leaf(p.x);
            const gamma = try t.leaf(p.gamma);
            const beta = try t.leaf(p.beta);
            const y = try t.leaf(p.y);
            return .{ .loss = try t.mse(try t.layerNorm(x, gamma, beta), y), .x = x, .gamma = gamma, .beta = beta, .y = y };
        }

        pub fn acceptable(_: Allocator, _: Inputs) !bool {
            return true;
        }
    };
}

// A fully connected layer in one op, wide enough that the GPU product takes its larger tiles too.
fn LinearNet(comptime rows: usize, comptime n_in: usize, comptime n_out: usize) type {
    return struct {
        const X = Tensor(f32, &.{ rows, n_in });
        const W = Tensor(f32, &.{ n_in, n_out });
        const B = Tensor(f32, &.{n_out});
        const Y = Tensor(f32, &.{ rows, n_out });

        pub const Inputs = struct { x: X, w: W, b: B, y: Y };
        pub const names = .{ "x", "w", "b", "y" };
        pub const labels: [rows]u8 = @splat(0);
        const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), w: Var(W), b: Var(B), y: Var(Y) };

        pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
            const x = try t.leaf(p.x);
            const w = try t.leaf(p.w);
            const b = try t.leaf(p.b);
            const y = try t.leaf(p.y);
            return .{ .loss = try t.mse(try t.linear(x, w, b), y), .x = x, .w = w, .b = b, .y = y };
        }

        pub fn acceptable(_: Allocator, _: Inputs) !bool {
            return true;
        }
    };
}
// Both halves of a GRU step chained: every input of gruGates and gruNext receives a gradient, and the update gate feeds both.
const GruOpsNet = struct {
    const Wide = Tensor(f32, &.{ 3, 15 });
    const Recurrent = Tensor(f32, &.{ 3, 10 });
    const State = Tensor(f32, &.{ 3, 5 });

    pub const Inputs = struct { gx: Wide, gh: Recurrent, gc: State, h: State, y: State };
    pub const names = .{ "gx", "gh", "gc", "h", "y" };
    pub const labels: [3]u8 = @splat(0);
    const Result = struct { loss: Var(Tensor(f32, &.{})), gx: Var(Wide), gh: Var(Recurrent), gc: Var(State), h: Var(State), y: Var(State) };

    pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
        const gx = try t.leaf(p.gx);
        const gh = try t.leaf(p.gh);
        const gc = try t.leaf(p.gc);
        const h = try t.leaf(p.h);
        const y = try t.leaf(p.y);
        const update, const reset_h = try t.gruGates(gx, gh, h);
        // The reset state and the candidate projection are independent leaves here, so their sum stands in for the candidate matmul.
        const next = try t.gruNext(gx, try t.add(gc, reset_h), update, h);
        // NOTE: the loss is scaled up so that an error of a few percent in one gate stays above the tolerance.
        const scale = try t.constant(try t.filledTensor(Tensor(f32, &.{}), 100));
        return .{ .loss = try t.mul(try t.mse(next, y), scale), .gx = gx, .gh = gh, .gc = gc, .h = h, .y = y };
    }

    pub fn acceptable(_: Allocator, _: Inputs) !bool {
        return true;
    }
};

fn BatchedNet(comptime trans_b: bool) type {
    return struct {
        const A = Tensor(f32, &.{ 3, 4, 5 });
        const B = Tensor(f32, if (trans_b) &.{ 3, 6, 5 } else &.{ 3, 5, 6 });
        const Y = Tensor(f32, &.{ 12, 6 });
        pub const Inputs = struct { a: A, b: B, y: Y };
        pub const names = .{ "a", "b", "y" };
        pub const labels: [12]u8 = @splat(0);
        const Result = struct { loss: Var(Tensor(f32, &.{})), a: Var(A), b: Var(B), y: Var(Y) };

        pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
            const a = try t.leaf(p.a);
            const b = try t.leaf(p.b);
            const y = try t.leaf(p.y);
            return .{ .loss = try t.mse(reshape(try t.batchedMatmul(a, b, trans_b), &.{ 12, 6 }), y), .a = a, .b = b, .y = y };
        }

        pub fn acceptable(_: Allocator, _: Inputs) !bool {
            return true;
        }
    };
}

fn SoftmaxNet(comptime period: usize, comptime rows: usize, comptime cols: usize, comptime scale: f32) type {
    return struct {
        const X = Tensor(f32, &.{ rows, cols });
        pub const Inputs = struct { x: X, y: X };
        pub const names = .{ "x", "y" };
        pub const labels: [rows]u8 = @splat(0);
        const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), y: Var(X) };

        pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
            const x = try t.leaf(p.x);
            const y = try t.leaf(p.y);
            return .{ .loss = try t.mse(try t.softmax(x, .{ .causal_period = period, .scale = scale }), y), .x = x, .y = y };
        }

        pub fn acceptable(_: Allocator, _: Inputs) !bool {
            return true;
        }
    };
}

// Rows wider than a warp and not a multiple of its width, and more rows than a block has threads, with label smoothing.
const WideCrossEntropyNet = struct {
    const X = Tensor(f32, &.{ 300, 200 });
    pub const Inputs = struct { x: X };
    pub const names = .{"x"};
    pub const labels: [300]u8 = blk: {
        var values: [300]u8 = undefined;
        for (&values, 0..) |*v, i| v.* = @intCast(i * 7 % 200);
        break :blk values;
    };
    pub const smoothing: f32 = 0.1;
    const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X) };

    pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !Result {
        const x = try t.leaf(p.x);
        return .{ .loss = try t.crossEntropy(x, label_slice), .x = x };
    }

    pub fn acceptable(_: Allocator, _: Inputs) !bool {
        return true;
    }
};

const SmoothedNet = struct {
    pub const Inputs = SmallNet.Inputs;
    pub const names = SmallNet.names;
    pub const labels = SmallNet.labels;
    pub const smoothing: f32 = 0.1;

    pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !SmallNet.Result {
        return SmallNet.build(t, p, label_slice);
    }

    pub fn acceptable(gpa: Allocator, p: Inputs) !bool {
        return SmallNet.acceptable(gpa, p);
    }
};

const ConvPoolNet = struct {
    const X = Tensor(f32, &.{ 2, 6, 6, 2 });
    const W = Tensor(f32, &.{ 18, 3 });
    const B = Tensor(f32, &.{3});
    const V = Tensor(f32, &.{ 12, 3 });
    const Act = Tensor(f32, &.{ 2, 4, 4, 3 });

    pub const Inputs = struct { x: X, w: W, b: B, v: V };
    pub const names = .{ "x", "w", "b", "v" };
    pub const labels = [_]u8{ 2, 0 };
    const Result = struct { loss: Var(Tensor(f32, &.{})), act: Var(Act), x: Var(X), w: Var(W), b: Var(B), v: Var(V) };

    pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !Result {
        const x = try t.leaf(p.x);
        const w = try t.leaf(p.w);
        const b = try t.leaf(p.b);
        const v = try t.leaf(p.v);
        const act = try t.add(try t.conv2d(x, w, .{}), b);
        const flat = reshape(try t.maxPool(act, .{}), &.{ 2, 12 });
        const loss = try t.crossEntropy(try t.matmul(flat, v), label_slice);
        return .{ .loss = loss, .act = act, .x = x, .w = w, .b = b, .v = v };
    }

    pub fn acceptable(gpa: Allocator, p: Inputs) !bool {
        var t: Tape = .{ .gpa = gpa };
        const act = (try build(&t, p, &labels)).act.value;
        var smallest: f32 = std.math.floatMax(f32);
        for (0..2) |n| {
            for (0..2) |oy| {
                for (0..2) |ox| {
                    for (0..3) |c| {
                        var top: f32 = -std.math.floatMax(f32);
                        var second = top;
                        for (0..4) |q| {
                            const v = act.data[((n * 4 + 2 * oy + q / 2) * 4 + 2 * ox + q % 2) * 3 + c];
                            if (v > top) {
                                second = top;
                                top = v;
                            } else if (v > second) second = v;
                        }
                        smallest = @min(smallest, top - second);
                    }
                }
            }
        }
        return smallest > 0.1;
    }
};

fn ConvCase(comptime opts: ConvOpts, comptime k: usize, comptime c_in: usize, comptime c_out: usize) type {
    return struct {
        const In = Tensor(f32, &.{ 2, 5, 5, c_in });
        const Wt = Tensor(f32, &.{ k * k * c_in / opts.groups, c_out });
        const Out = ConvOut(In, Wt, opts);

        pub const Inputs = struct { x: In, w: Wt };
        pub const names = .{ "x", "w" };
        pub const labels = [_]u8{ 1, 4 };
        const Result = struct { loss: Var(Tensor(f32, &.{})), y: Var(Out), x: Var(In), w: Var(Wt) };

        pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !Result {
            const x = try t.leaf(p.x);
            const w = try t.leaf(p.w);
            const y = try t.conv2d(x, w, opts);
            const loss = try t.crossEntropy(reshape(y, &.{ 2, Out.len / 2 }), label_slice);
            return .{ .loss = loss, .y = y, .x = x, .w = w };
        }

        fn expectDirectConvolution() !void {
            var arena = std.heap.ArenaAllocator.init(testing.allocator);
            defer arena.deinit();
            const p = try gradcheck.randomInputs(@This(), arena.allocator(), 31);
            var t: Tape = .{ .gpa = arena.allocator() };
            const run = try build(&t, p, &labels);

            const oh = Out.shape[1];
            const ow = Out.shape[2];
            for (0..2) |n| {
                for (0..oh) |oy| {
                    for (0..ow) |ox| {
                        for (0..c_out) |oc| {
                            var want: f32 = 0;
                            const per_group = c_in / opts.groups;
                            const group = oc / (c_out / opts.groups);
                            for (0..k) |ky| {
                                for (0..k) |kx| {
                                    const py = oy * opts.stride + ky * opts.dilation;
                                    const px = ox * opts.stride + kx * opts.dilation;
                                    if (py < opts.pad or px < opts.pad or py - opts.pad >= 5 or px - opts.pad >= 5) continue;
                                    for (0..per_group) |c| {
                                        want += p.x.data[((n * 5 + py - opts.pad) * 5 + px - opts.pad) * c_in + group * per_group + c] * p.w.data[((ky * k + kx) * per_group + c) * c_out + oc];
                                    }
                                }
                            }
                            try testing.expectApproxEqAbs(want, run.y.value.data[((n * oh + oy) * ow + ox) * c_out + oc], 1e-6 * @as(f32, k * k * c_in));
                        }
                    }
                }
            }
        }
    };
}

const BnCase = struct {
    const Channels = Tensor(f32, &.{3});
    const X = Tensor(f32, &.{ 6, 3 });
    const V = Tensor(f32, &.{ 3, 4 });

    pub const Inputs = struct { x: X, gamma: Channels, beta: Channels, v: V };
    pub const names = .{ "x", "gamma", "beta", "v" };
    pub const value_names = .{ "running_mean", "running_var" };
    pub const labels = [_]u8{ 0, 3, 1, 2, 3, 0 };
    const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), gamma: Var(Channels), beta: Var(Channels), v: Var(V), running_mean: Channels, running_var: Channels };

    pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !Result {
        const x = try t.leaf(p.x);
        const gamma = try t.leaf(p.gamma);
        const beta = try t.leaf(p.beta);
        const v = try t.leaf(p.v);
        const running_mean = try t.zeroedTensor(Channels);
        const running_var = try t.filledTensor(Channels, 1);
        const normalized = try t.batchNorm(x, gamma, beta, .{ .mean = running_mean, .variance = running_var });
        const loss = try t.crossEntropy(try t.matmul(normalized, v), label_slice);
        return .{ .loss = loss, .x = x, .gamma = gamma, .beta = beta, .v = v, .running_mean = running_mean, .running_var = running_var };
    }
};

const BnReluCase = struct {
    pub const Inputs = BnCase.Inputs;
    pub const names = BnCase.names;
    pub const value_names = BnCase.value_names;
    pub const labels = BnCase.labels;

    fn network(t: *Tape, p: Inputs, label_slice: []const u8, comptime fused: bool) !BnCase.Result {
        const x = try t.leaf(p.x);
        const gamma = try t.leaf(p.gamma);
        const beta = try t.leaf(p.beta);
        const v = try t.leaf(p.v);
        const running_mean = try t.zeroedTensor(BnCase.Channels);
        const running_var = try t.filledTensor(BnCase.Channels, 1);
        const stats = .{ .mean = running_mean, .variance = running_var };
        const activated = if (fused) try t.batchNormRelu(x, gamma, beta, stats) else try t.relu(try t.batchNorm(x, gamma, beta, stats));
        const loss = try t.crossEntropy(try t.matmul(activated, v), label_slice);
        return .{ .loss = loss, .x = x, .gamma = gamma, .beta = beta, .v = v, .running_mean = running_mean, .running_var = running_var };
    }

    pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !BnCase.Result {
        return network(t, p, label_slice, true);
    }

    pub fn acceptable(gpa: Allocator, p: Inputs) !bool {
        var t: Tape = .{ .gpa = gpa, .training = true };
        const running_mean = try t.zeroedTensor(BnCase.Channels);
        const running_var = try t.filledTensor(BnCase.Channels, 1);
        const y = try t.batchNorm(try t.constant(p.x), try t.constant(p.gamma), try t.constant(p.beta), .{ .mean = running_mean, .variance = running_var });
        var smallest: f32 = std.math.floatMax(f32);
        for (y.value.data) |v| smallest = @min(smallest, @abs(v));
        return smallest > 0.05;
    }
};

test "fused batch norm and relu gives exactly the same values and gradients as the two ops" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const p = try gradcheck.randomInputs(BnReluCase, gpa, 44);

    var separate_tape: Tape = .{ .gpa = gpa, .training = true };
    const separate = try BnReluCase.network(&separate_tape, p, &BnReluCase.labels, false);
    try separate_tape.backward(separate.loss);
    var fused_tape: Tape = .{ .gpa = gpa, .training = true };
    const fused = try BnReluCase.network(&fused_tape, p, &BnReluCase.labels, true);
    try fused_tape.backward(fused.loss);

    try testing.expectEqual(separate.loss.value.data[0], fused.loss.value.data[0]);
    inline for (BnReluCase.names) |name| {
        try testing.expectEqualSlices(f32, @field(separate, name).grad.data, @field(fused, name).grad.data);
    }
    try testing.expect(fused_tape.nodes.items.len < separate_tape.nodes.items.len);
}

test "batch norm with a fused relu matches finite differences, the gpu and the tensor-core path" {
    try gradcheck.expectFiniteDifferences(BnReluCase, 45, 1e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(BnReluCase, 46, 1e-5);
    try gradcheck.expectTensorCoresClose(BnReluCase, 46, 3e-2);
}

const BnAddReluCase = struct {
    pub const Inputs = struct { x: BnCase.X, r: BnCase.X, gamma: BnCase.Channels, beta: BnCase.Channels, v: BnCase.V };
    pub const names = .{ "x", "r", "gamma", "beta", "v" };
    pub const value_names = BnCase.value_names;
    pub const labels = BnCase.labels;
    const Result = struct {
        loss: Var(Tensor(f32, &.{})),
        x: Var(BnCase.X),
        r: Var(BnCase.X),
        gamma: Var(BnCase.Channels),
        beta: Var(BnCase.Channels),
        v: Var(BnCase.V),
        running_mean: BnCase.Channels,
        running_var: BnCase.Channels,
    };

    fn network(t: *Tape, p: Inputs, label_slice: []const u8, comptime fused: bool) !Result {
        const x = try t.leaf(p.x);
        const r = try t.leaf(p.r);
        const gamma = try t.leaf(p.gamma);
        const beta = try t.leaf(p.beta);
        const v = try t.leaf(p.v);
        const running_mean = try t.zeroedTensor(BnCase.Channels);
        const running_var = try t.filledTensor(BnCase.Channels, 1);
        const stats = .{ .mean = running_mean, .variance = running_var };
        const activated = if (fused) try t.batchNormAddRelu(x, r, gamma, beta, stats) else try t.addRelu(try t.batchNorm(x, gamma, beta, stats), r);
        const loss = try t.crossEntropy(try t.matmul(activated, v), label_slice);
        return .{ .loss = loss, .x = x, .r = r, .gamma = gamma, .beta = beta, .v = v, .running_mean = running_mean, .running_var = running_var };
    }

    pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !Result {
        return network(t, p, label_slice, true);
    }

    pub fn acceptable(gpa: Allocator, p: Inputs) !bool {
        var t: Tape = .{ .gpa = gpa, .training = true };
        const running_mean = try t.zeroedTensor(BnCase.Channels);
        const running_var = try t.filledTensor(BnCase.Channels, 1);
        const y = try t.add(try t.batchNorm(try t.constant(p.x), try t.constant(p.gamma), try t.constant(p.beta), .{ .mean = running_mean, .variance = running_var }), try t.constant(p.r));
        var smallest: f32 = std.math.floatMax(f32);
        for (y.value.data) |v| smallest = @min(smallest, @abs(v));
        return smallest > 0.05;
    }
};

test "batch norm with a fused residual and relu equals the separate ops and matches finite differences and the gpu" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const p = try gradcheck.randomInputs(BnAddReluCase, gpa, 50);

    var separate_tape: Tape = .{ .gpa = gpa, .training = true };
    const separate = try BnAddReluCase.network(&separate_tape, p, &BnAddReluCase.labels, false);
    try separate_tape.backward(separate.loss);
    var fused_tape: Tape = .{ .gpa = gpa, .training = true };
    const fused = try BnAddReluCase.network(&fused_tape, p, &BnAddReluCase.labels, true);
    try fused_tape.backward(fused.loss);

    try testing.expectEqual(separate.loss.value.data[0], fused.loss.value.data[0]);
    inline for (BnAddReluCase.names) |name| {
        try testing.expectEqualSlices(f32, @field(separate, name).grad.data, @field(fused, name).grad.data);
    }
    try testing.expect(fused_tape.nodes.items.len < separate_tape.nodes.items.len);

    try gradcheck.expectFiniteDifferences(BnAddReluCase, 51, 1e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(BnAddReluCase, 52, 1e-5);
}

const AddReluNet = struct {
    const X = Tensor(f32, &.{ 4, 3 });
    const W = Tensor(f32, &.{ 3, 5 });
    const V = Tensor(f32, &.{ 5, 3 });

    pub const Inputs = struct { x: X, w1: W, w2: W, v: V };
    pub const names = .{ "x", "w1", "w2", "v" };
    pub const labels = [_]u8{ 0, 2, 1, 2 };
    const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), w1: Var(W), w2: Var(W), v: Var(V) };

    fn network(t: *Tape, p: Inputs, label_slice: []const u8, comptime fused: bool) !Result {
        const x = try t.leaf(p.x);
        const w1 = try t.leaf(p.w1);
        const w2 = try t.leaf(p.w2);
        const v = try t.leaf(p.v);
        const left = try t.matmul(x, w1);
        const right = try t.matmul(x, w2);
        const activated = if (fused) try t.addRelu(left, right) else try t.relu(try t.add(left, right));
        const loss = try t.crossEntropy(try t.matmul(activated, v), label_slice);
        return .{ .loss = loss, .x = x, .w1 = w1, .w2 = w2, .v = v };
    }

    pub fn build(t: *Tape, p: Inputs, label_slice: []const u8) !Result {
        return network(t, p, label_slice, true);
    }

    pub fn acceptable(gpa: Allocator, p: Inputs) !bool {
        var t: Tape = .{ .gpa = gpa };
        const sum = try t.add(try t.matmul(try t.constant(p.x), try t.constant(p.w1)), try t.matmul(try t.constant(p.x), try t.constant(p.w2)));
        var smallest: f32 = std.math.floatMax(f32);
        for (sum.value.data) |v| smallest = @min(smallest, @abs(v));
        return smallest > 0.05;
    }
};

test "fused add and relu equals the two ops exactly and matches finite differences, the gpu and tensor cores" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const p = try gradcheck.randomInputs(AddReluNet, gpa, 47);

    var separate_tape: Tape = .{ .gpa = gpa };
    const separate = try AddReluNet.network(&separate_tape, p, &AddReluNet.labels, false);
    try separate_tape.backward(separate.loss);
    var fused_tape: Tape = .{ .gpa = gpa };
    const fused = try AddReluNet.network(&fused_tape, p, &AddReluNet.labels, true);
    try fused_tape.backward(fused.loss);
    try testing.expectEqual(separate.loss.value.data[0], fused.loss.value.data[0]);
    inline for (AddReluNet.names) |name| {
        try testing.expectEqualSlices(f32, @field(separate, name).grad.data, @field(fused, name).grad.data);
    }

    try gradcheck.expectFiniteDifferences(AddReluNet, 48, 1e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(AddReluNet, 49, 1e-5);
    try gradcheck.expectTensorCoresClose(AddReluNet, 49, 3e-2);
}

test "addRelu leaves a constant operand without a gradient, on the cpu and on the gpu" {
    const gpu = try gpu_mod.Gpu.forTest(1 << 20);
    defer gpu.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const Row = Tensor(f32, &.{ 1, 6 });
    const left = [_]f32{ 1, -2, 3, -4, 0.5, -0.5 };
    const right = [_]f32{ -3, 1, -1, 5, 0.25, 0.75 };

    var cpu_tape: Tape = .{ .gpa = arena.allocator() };
    const cpu_a = try cpu_tape.leaf(try Row.fromSlice(cpu_tape.gpa, &left));
    const cpu_out = try cpu_tape.addRelu(cpu_a, try cpu_tape.constant(try Row.fromSlice(cpu_tape.gpa, &right)));
    @memset(cpu_out.grad.data, 1);
    try cpu_tape.nodes.items[0].run(&cpu_tape, cpu_tape.nodes.items[0].ctx);
    try testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1, 1, 1 }, cpu_a.grad.data);

    const device_left = try gpu.alloc(f32, 6);
    const device_right = try gpu.alloc(f32, 6);
    const device_grad = try gpu.alloc(f32, 6);
    try gpu.upload(device_left, &left);
    try gpu.upload(device_right, &right);
    try gpu.fill(device_grad, 0);
    var gpu_tape: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const gpu_a: Var(Row) = .{ .value = .{ .data = device_left }, .grad = .{ .data = device_grad }, .needs_grad = true };
    const gpu_out = try gpu_tape.addRelu(gpu_a, try gpu_tape.constant(Row{ .data = device_right }));
    try gpu.fill(gpu_out.grad.data, 1);
    try gpu_tape.nodes.items[0].run(&gpu_tape, gpu_tape.nodes.items[0].ctx);
    var got: [6]f32 = undefined;
    try gpu.download(&got, device_grad);
    try testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1, 1, 1 }, &got);
}

test "gpu and cpu agree on a small network" {
    try gradcheck.expectGpuMatchesCpu(SmallNet, 21, 1e-5);
    try gradcheck.expectTensorCoresClose(SmallNet, 21, 3e-2);
}

test "mse is the mean squared difference and its gradients point towards the target" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator() };
    const prediction = try t.leaf(try Tensor(f32, &.{ 2, 2 }).fromSlice(t.gpa, &.{ 1, 2, 3, 4 }));
    const target = try t.leaf(try Tensor(f32, &.{ 2, 2 }).fromSlice(t.gpa, &.{ 0, 0, 0, 0 }));
    const loss = try t.mse(prediction, target);
    try t.backward(loss);

    try testing.expectEqual(@as(f32, 7.5), loss.value.data[0]);
    try testing.expectEqualSlices(f32, &.{ 0.5, 1, 1.5, 2 }, prediction.grad.data);
    try testing.expectEqualSlices(f32, &.{ -0.5, -1, -1.5, -2 }, target.grad.data);
}

test "gradients of the mse loss match finite differences and the gpu" {
    try gradcheck.expectFiniteDifferences(MseNet, 7, 5e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(MseNet, 21, 1e-5);
}

test "gradients through gelu, tanh and sigmoid match finite differences and the gpu" {
    try gradcheck.expectFiniteDifferences(ActivationNet, 7, 5e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(ActivationNet, 21, 2e-5);
}

fn expectPoolingMatchesReference(comptime opts: PoolOpts) !void {
    const X = Tensor(f32, &.{ 2, 7, 6, 3 });
    const P = PoolOut(X, opts);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x_values = try a.alloc(f32, X.len);
    for (x_values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 37 % X.len)) * 0.05;
    const targets = try a.alloc(f32, P.len);
    for (targets, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.3;

    var t: Tape = .{ .gpa = a };
    const x = try t.leaf(try X.fromSlice(a, x_values));
    const pooled = try t.add(try t.maxPool(x, opts), try t.avgPool(x, opts));
    const loss = try t.mse(reshape(pooled, &.{ 2, P.len / 2 }), try t.constant(try Tensor(f32, &.{ 2, P.len / 2 }).fromSlice(a, targets)));
    try t.backward(loss);

    const reference = try a.alloc(f32, P.len);
    const upstream = try a.alloc(f32, P.len);
    const expected_grad = try a.alloc(f32, X.len);
    @memset(expected_grad, 0);
    var oy: usize = 0;
    const side_y = X.shape[1];
    const side_x = X.shape[2];
    const out_h = P.shape[1];
    const out_w = P.shape[2];
    for (0..2) |n| {
        oy = 0;
        while (oy < out_h) : (oy += 1) {
            for (0..out_w) |ox| {
                for (0..3) |ch| {
                    var best: ?usize = null;
                    var total: f32 = 0;
                    var count: f32 = 0;
                    for (0..opts.kernel) |ky| {
                        for (0..opts.kernel) |kx| {
                            const yy = oy * opts.stride + ky;
                            const xx = ox * opts.stride + kx;
                            if (yy < opts.pad or xx < opts.pad or yy - opts.pad >= side_y or xx - opts.pad >= side_x) continue;
                            const at = ((n * side_y + yy - opts.pad) * side_x + xx - opts.pad) * 3 + ch;
                            if (best == null or x_values[at] > x_values[best.?]) best = at;
                            total += x_values[at];
                            count += 1;
                        }
                    }
                    const j = ((n * out_h + oy) * out_w + ox) * 3 + ch;
                    reference[j] = x_values[best.?] + total / count;
                    upstream[j] = 2 * (reference[j] - targets[j]) / @as(f32, @floatFromInt(P.len));
                    expected_grad[best.?] += upstream[j];
                    for (0..opts.kernel) |ky| {
                        for (0..opts.kernel) |kx| {
                            const yy = oy * opts.stride + ky;
                            const xx = ox * opts.stride + kx;
                            if (yy < opts.pad or xx < opts.pad or yy - opts.pad >= side_y or xx - opts.pad >= side_x) continue;
                            expected_grad[((n * side_y + yy - opts.pad) * side_x + xx - opts.pad) * 3 + ch] += upstream[j] / count;
                        }
                    }
                }
            }
        }
    }
    for (reference, pooled.value.data) |want, got| try testing.expectApproxEqAbs(want, got, 1e-5);
    for (expected_grad, x.grad.data) |want, got| try testing.expectApproxEqAbs(want, got, 1e-5);
}

test "pooling with other windows, strides and padding matches a brute-force reference, also on the gpu" {
    inline for (.{ PoolOpts{}, PoolOpts{ .kernel = 3, .stride = 2 }, PoolOpts{ .kernel = 3, .stride = 2, .pad = 1 }, PoolOpts{ .kernel = 3, .stride = 1, .pad = 1 }, PoolOpts{ .kernel = 2, .stride = 1 } }) |opts| {
        try expectPoolingMatchesReference(opts);
        try gradcheck.expectGpuMatchesCpu(PoolCase(opts), 6, 1e-5);
    }
}
test "broadcasting and its gradient reductions work for tensors of rank six on the cpu and the gpu" {
    try gradcheck.expectFiniteDifferences(HighRankNet, 3, 1e-2, 3e-3);
    try gradcheck.expectGpuMatchesCpu(HighRankNet, 4, 1e-5);
}

test "casting to bf16 and back rounds the values and passes the gradient through" {
    const gpu = try gpu_mod.Gpu.forTest(1 << 20);
    defer gpu.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const X = Tensor(f32, &.{ 2, 4 });
    const values = [_]f32{ 1, 1.00195312, -2.5, 3.14159265, 0.1, 1000.5, -0.0078125, 65 };
    const device_x = try gpu.alloc(f32, X.len);
    const device_grad = try gpu.alloc(f32, X.len);
    try gpu.upload(device_x, &values);
    try gpu.fill(device_grad, 0);

    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const x: Var(X) = .{ .value = .{ .data = device_x }, .grad = .{ .data = device_grad }, .needs_grad = true };
    const half = try t.cast(x, u16);
    const back = try t.cast(half, f32);
    var host: [8]f32 = undefined;
    try gpu.download(&host, back.value.data);
    for (values, host) |want, got| try testing.expectEqual(bf16_math.decode(bf16_math.encode(want)), got);

    try gpu.fill(back.grad.data, 2);
    for (0..2) |i| {
        const node = t.nodes.items[1 - i];
        try node.run(&t, node.ctx);
    }
    try gpu.download(&host, device_grad);
    for (host) |g| try testing.expectEqual(@as(f32, 2), g);
}

// Runs batchNormRelu (or batchNormAddRelu with residual r when r_values is not empty) on the device in f32 or bf16 and downloads
// the output, the input gradient, the residual gradient into dr_values and the running statistics. The channel count selects
// the reduction kernel: an even count lets a bf16 lane read two channels at once, an odd count keeps the one-channel path.
fn runBatchNorm(comptime half: bool, comptime channels: usize, gpu: *gpu_mod.Gpu, x_values: []const f32, r_values: []const f32, dy_values: []const f32, out_values: []f32, dx_values: []f32, dr_values: []f32, running: []f32) !void {
    const X = Tensor(f32, &.{ 600, channels });
    const Channels = Tensor(f32, &.{channels});
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true };
    const x = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    try gpu.upload(x.value.data, x_values);
    try gpu.fill(x.grad.data, 0);
    const r = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    try gpu.fill(r.value.data, 0);
    if (r_values.len != 0) try gpu.upload(r.value.data, r_values);
    try gpu.fill(r.grad.data, 0);
    const gamma = try t.leaf(Channels{ .data = try gpu.alloc(f32, channels) });
    const beta = try t.leaf(Channels{ .data = try gpu.alloc(f32, channels) });
    try gpu.fill(gamma.value.data, 1.5);
    try gpu.fill(beta.value.data, 0.25);
    try gpu.fill(gamma.grad.data, 0);
    try gpu.fill(beta.grad.data, 0);
    const mean = Channels{ .data = try gpu.alloc(f32, channels) };
    const variance = Channels{ .data = try gpu.alloc(f32, channels) };
    try gpu.fill(mean.data, 0);
    try gpu.fill(variance.data, 1);

    const input = if (half) try t.cast(x, u16) else x;
    const running_stats = .{ .mean = mean, .variance = variance };
    const out = if (r_values.len == 0)
        try t.batchNormRelu(input, gamma, beta, running_stats)
    else
        try t.batchNormAddRelu(input, if (half) try t.cast(r, u16) else r, gamma, beta, running_stats);
    const back = try t.cast(out, f32);
    try gpu.download(out_values, back.value.data);
    try gpu.upload(back.grad.data, dy_values);
    var at = t.nodes.items.len;
    while (at > 0) {
        at -= 1;
        try t.nodes.items[at].run(&t, t.nodes.items[at].ctx);
    }
    try gpu.download(dx_values, x.grad.data);
    try gpu.download(dr_values, r.grad.data);
    try gpu.download(running[0..channels], mean.data);
    try gpu.download(running[channels..], variance.data);
}

test "batch norm on bf16 activations matches the f32 path within bf16 rounding, forward, backward and running statistics" {
    const gpu = try gpu_mod.Gpu.forTest(8 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(31);
    const rng = prng.random();
    var x: [600 * 16]f32 = undefined;
    var dy: [600 * 16]f32 = undefined;
    for (&x) |*v| v.* = rng.floatNorm(f32) * 2 + 0.5;
    for (&dy) |*v| v.* = rng.floatNorm(f32);
    var wide_out: [600 * 16]f32 = undefined;
    var half_out: [600 * 16]f32 = undefined;
    var wide_dx: [600 * 16]f32 = undefined;
    var half_dx: [600 * 16]f32 = undefined;
    var wide_running: [32]f32 = undefined;
    var half_running: [32]f32 = undefined;
    var unused: [600 * 16]f32 = undefined;
    try runBatchNorm(false, 16, gpu, &x, &.{}, &dy, &wide_out, &wide_dx, &unused, &wide_running);
    try runBatchNorm(true, 16, gpu, &x, &.{}, &dy, &half_out, &half_dx, &unused, &half_running);
    for (wide_out, half_out) |want, got| try testing.expectApproxEqAbs(want, got, 0.04 + 0.01 * @abs(want));
    var flipped: usize = 0;
    for (wide_dx, half_dx) |want, got| {
        if (@abs(want - got) > 0.03 + 0.02 * @abs(want)) flipped += 1;
    }
    try testing.expect(flipped <= 3);
    for (wide_running, half_running) |want, got| try testing.expectApproxEqAbs(want, got, 0.02 + 0.01 * @abs(want));
}

test "batch norm on bf16 activations with an odd channel count matches the f32 path within bf16 rounding" {
    const gpu = try gpu_mod.Gpu.forTest(8 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(34);
    const rng = prng.random();
    const c = 15;
    var x: [600 * c]f32 = undefined;
    var dy: [600 * c]f32 = undefined;
    for (&x) |*v| v.* = rng.floatNorm(f32) * 2 + 0.5;
    for (&dy) |*v| v.* = rng.floatNorm(f32);
    var wide_out: [600 * c]f32 = undefined;
    var half_out: [600 * c]f32 = undefined;
    var wide_dx: [600 * c]f32 = undefined;
    var half_dx: [600 * c]f32 = undefined;
    var wide_running: [2 * c]f32 = undefined;
    var half_running: [2 * c]f32 = undefined;
    var unused: [600 * c]f32 = undefined;
    // NOTE: 15 channels keep a bf16 lane on one channel per read; the even-count tests take the paired path instead.
    try runBatchNorm(false, c, gpu, &x, &.{}, &dy, &wide_out, &wide_dx, &unused, &wide_running);
    try runBatchNorm(true, c, gpu, &x, &.{}, &dy, &half_out, &half_dx, &unused, &half_running);
    for (wide_out, half_out) |want, got| try testing.expectApproxEqAbs(want, got, 0.04 + 0.01 * @abs(want));
    var flipped: usize = 0;
    for (wide_dx, half_dx) |want, got| {
        if (@abs(want - got) > 0.03 + 0.02 * @abs(want)) flipped += 1;
    }
    try testing.expect(flipped <= 3);
    for (wide_running, half_running) |want, got| try testing.expectApproxEqAbs(want, got, 0.02 + 0.01 * @abs(want));
}

test "batch norm with a fused residual and relu on bf16 activations matches the f32 path within bf16 rounding" {
    const gpu = try gpu_mod.Gpu.forTest(8 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(32);
    const rng = prng.random();
    var x: [600 * 16]f32 = undefined;
    var r: [600 * 16]f32 = undefined;
    var dy: [600 * 16]f32 = undefined;
    for (&x, &r, &dy) |*a, *b, *c| {
        a.* = rng.floatNorm(f32) * 2 + 0.5;
        b.* = rng.floatNorm(f32);
        c.* = rng.floatNorm(f32);
    }
    var wide_out: [x.len]f32 = undefined;
    var half_out: [x.len]f32 = undefined;
    var wide_dx: [x.len]f32 = undefined;
    var half_dx: [x.len]f32 = undefined;
    var wide_dr: [x.len]f32 = undefined;
    var half_dr: [x.len]f32 = undefined;
    var wide_running: [32]f32 = undefined;
    var half_running: [32]f32 = undefined;
    try runBatchNorm(false, 16, gpu, &x, &r, &dy, &wide_out, &wide_dx, &wide_dr, &wide_running);
    try runBatchNorm(true, 16, gpu, &x, &r, &dy, &half_out, &half_dx, &half_dr, &half_running);
    for (wide_out, half_out) |want, got| try testing.expectApproxEqAbs(want, got, 0.04 + 0.01 * @abs(want));
    var flipped: usize = 0;
    var active: usize = 0;
    for (wide_dx, half_dx, wide_dr, half_dr, wide_out) |want, got, want_r, got_r, o| {
        if (@abs(want - got) > 0.03 + 0.02 * @abs(want)) flipped += 1;
        if (@abs(want_r - got_r) > 0.01 + 0.01 * @abs(want_r)) flipped += 1;
        if (o > 0) active += 1;
    }
    try testing.expect(flipped <= 6);
    // NOTE: the residual gradient is the masked output gradient, so it must be zero exactly where the output is.
    for (wide_dr, wide_out, dy) |g, o, d| try testing.expectEqual(if (o > 0) d else 0, g);
    try testing.expect(active > x.len / 4 and active < x.len * 3 / 4);
}

fn runHalfChain(comptime half: bool, gpu: *gpu_mod.Gpu, a_values: []const f32, b_values: []const f32, g_values: []const f32, out: []f32, grad_a: []f32, grad_b: []f32) !void {
    const X = Tensor(f32, &.{ 2, 8, 8, 4 });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true };
    const a = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    const b = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    try gpu.upload(a.value.data, a_values);
    try gpu.upload(b.value.data, b_values);
    try gpu.fill(a.grad.data, 0);
    try gpu.fill(b.grad.data, 0);
    const lhs = if (half) try t.cast(a, u16) else a;
    const rhs = if (half) try t.cast(b, u16) else b;
    const merged = try t.addRelu(lhs, rhs);
    const pooled = try t.avgPool(try t.maxPool(merged, .{}), .{});
    const result = try t.cast(pooled, f32);
    try gpu.download(out, result.value.data);
    try gpu.upload(result.grad.data, g_values);
    var at = t.nodes.items.len;
    while (at > 0) {
        at -= 1;
        try t.nodes.items[at].run(&t, t.nodes.items[at].ctx);
    }
    try gpu.download(grad_a, a.grad.data);
    try gpu.download(grad_b, b.grad.data);
}

test "addRelu, max pooling and average pooling on bf16 activations match the f32 path within bf16 rounding" {
    const gpu = try gpu_mod.Gpu.forTest(8 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(33);
    const rng = prng.random();
    var a: [2 * 8 * 8 * 4]f32 = undefined;
    var b: [2 * 8 * 8 * 4]f32 = undefined;
    var g: [2 * 2 * 2 * 4]f32 = undefined;
    for (&a) |*v| v.* = rng.floatNorm(f32);
    for (&b) |*v| v.* = rng.floatNorm(f32);
    for (&g) |*v| v.* = rng.floatNorm(f32);
    var wide: [g.len]f32 = undefined;
    var half: [g.len]f32 = undefined;
    var wide_a: [a.len]f32 = undefined;
    var wide_b: [a.len]f32 = undefined;
    var half_a: [a.len]f32 = undefined;
    var half_b: [a.len]f32 = undefined;
    try runHalfChain(false, gpu, &a, &b, &g, &wide, &wide_a, &wide_b);
    try runHalfChain(true, gpu, &a, &b, &g, &half, &half_a, &half_b);
    for (wide, half) |want, got| try testing.expectApproxEqAbs(want, got, 0.03 + 0.01 * @abs(want));
    var flipped: usize = 0;
    for (wide_a, half_a, wide_b, half_b) |wa, ha, wb, hb| {
        if (@abs(wa - ha) > 0.01 + 0.02 * @abs(wa)) flipped += 1;
        if (@abs(wb - hb) > 0.01 + 0.02 * @abs(wb)) flipped += 1;
    }
    try testing.expect(flipped <= 8);
}

fn runConvChain(comptime half: bool, gpu: *gpu_mod.Gpu, x_values: []const f32, w_values: []const f32, g_values: []const f32, out: []f32, grad_x: []f32, grad_w: []f32) !void {
    const X = Tensor(f32, &.{ 2, 8, 8, 64 });
    const W = Tensor(f32, &.{ 576, 32 });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true };
    const x = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    const w = try t.leaf(W{ .data = try gpu.alloc(f32, W.len) });
    try gpu.upload(x.value.data, x_values);
    try gpu.upload(w.value.data, w_values);
    try gpu.fill(x.grad.data, 0);
    try gpu.fill(w.grad.data, 0);
    const input = if (half) try t.cast(x, u16) else x;
    const y = try t.cast(try t.conv2d(input, w, .{ .pad = 1 }), f32);
    try gpu.download(out, y.value.data);
    try gpu.upload(y.grad.data, g_values);
    var at = t.nodes.items.len;
    while (at > 0) {
        at -= 1;
        try t.nodes.items[at].run(&t, t.nodes.items[at].ctx);
    }
    try gpu.download(grad_x, x.grad.data);
    try gpu.download(grad_w, w.grad.data);
}

test "a convolution on bf16 activations agrees with the f32 convolution within bf16 rounding, outputs and both gradients" {
    const gpu = try gpu_mod.Gpu.forTest(32 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(37);
    const rng = prng.random();
    const x = try testing.allocator.alloc(f32, 2 * 8 * 8 * 64);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, 576 * 32);
    defer testing.allocator.free(w);
    const g = try testing.allocator.alloc(f32, 2 * 8 * 8 * 32);
    defer testing.allocator.free(g);
    for (x) |*v| v.* = rng.floatNorm(f32);
    for (w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    for (g) |*v| v.* = rng.floatNorm(f32);
    const outputs = try testing.allocator.alloc(f32, 4 * g.len + 2 * x.len + 2 * w.len);
    defer testing.allocator.free(outputs);
    const wide_y, const half_y = .{ outputs[0..g.len], outputs[g.len..][0..g.len] };
    const wide_dx, const half_dx = .{ outputs[2 * g.len ..][0..x.len], outputs[2 * g.len + x.len ..][0..x.len] };
    const wide_dw, const half_dw = .{ outputs[2 * g.len + 2 * x.len ..][0..w.len], outputs[2 * g.len + 2 * x.len + w.len ..][0..w.len] };
    try runConvChain(false, gpu, x, w, g, wide_y, wide_dx, wide_dw);
    try runConvChain(true, gpu, x, w, g, half_y, half_dx, half_dw);
    for (wide_y, half_y) |want, got| try testing.expectApproxEqAbs(want, got, 0.08 + 0.02 * @abs(want));
    for (wide_dx, half_dx) |want, got| try testing.expectApproxEqAbs(want, got, 0.25 + 0.03 * @abs(want));
    for (wide_dw, half_dw) |want, got| try testing.expectApproxEqAbs(want, got, 0.6 + 0.03 * @abs(want));
}

// Runs convBatchNormRelu (batch statistics from the conv epilogue) or conv2d followed by batchNormRelu (a separate
// reduction) on bf16 activations and downloads the output, the input, weight, gamma, beta and residual gradients
// and the running statistics. r_values empty selects the plain ReLU variant.
fn runConvBn(comptime fused: bool, comptime n: usize, comptime side: usize, comptime c_in: usize, comptime c_out: usize, gpu: *gpu_mod.Gpu, x_values: []const f32, w_values: []const f32, r_values: []const f32, dy_values: []const f32, out: []f32, dx: []f32, dw: []f32, dgamma: []f32, dbeta: []f32, dr: []f32, running: []f32) !void {
    const X = Tensor(f32, &.{ n, side, side, c_in });
    const W = Tensor(f32, &.{ 9 * c_in, c_out });
    const Y = Tensor(f32, &.{ n, side, side, c_out });
    const Channels = Tensor(f32, &.{c_out});
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true };
    const x = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    const w = try t.leaf(W{ .data = try gpu.alloc(f32, W.len) });
    try gpu.upload(x.value.data, x_values);
    try gpu.upload(w.value.data, w_values);
    try gpu.fill(x.grad.data, 0);
    try gpu.fill(w.grad.data, 0);
    const r = try t.leaf(Y{ .data = try gpu.alloc(f32, Y.len) });
    try gpu.fill(r.value.data, 0);
    if (r_values.len != 0) try gpu.upload(r.value.data, r_values);
    try gpu.fill(r.grad.data, 0);
    const gamma = try t.leaf(Channels{ .data = try gpu.alloc(f32, c_out) });
    const beta = try t.leaf(Channels{ .data = try gpu.alloc(f32, c_out) });
    try gpu.fill(gamma.value.data, 1.5);
    try gpu.fill(beta.value.data, 0.25);
    try gpu.fill(gamma.grad.data, 0);
    try gpu.fill(beta.grad.data, 0);
    const mean = Channels{ .data = try gpu.alloc(f32, c_out) };
    const variance = Channels{ .data = try gpu.alloc(f32, c_out) };
    try gpu.fill(mean.data, 0);
    try gpu.fill(variance.data, 1);

    const input = try t.cast(x, u16);
    const stats = .{ .mean = mean, .variance = variance };
    const y = if (r_values.len == 0)
        (if (fused)
            try t.convBatchNormRelu(input, w, gamma, beta, stats, .{ .pad = 1 })
        else
            try t.batchNormRelu(try t.conv2d(input, w, .{ .pad = 1 }), gamma, beta, stats))
    else blk: {
        const residual = try t.cast(r, u16);
        break :blk if (fused)
            try t.convBatchNormAddRelu(input, w, residual, gamma, beta, stats, .{ .pad = 1 })
        else
            try t.batchNormAddRelu(try t.conv2d(input, w, .{ .pad = 1 }), residual, gamma, beta, stats);
    };
    const back = try t.cast(y, f32);
    try gpu.download(out, back.value.data);
    try gpu.upload(back.grad.data, dy_values);
    var at = t.nodes.items.len;
    while (at > 0) {
        at -= 1;
        try t.nodes.items[at].run(&t, t.nodes.items[at].ctx);
    }
    if (dx.len != 0) try gpu.download(dx, x.grad.data);
    if (dw.len != 0) try gpu.download(dw, w.grad.data);
    if (dgamma.len != 0) try gpu.download(dgamma, gamma.grad.data);
    if (dbeta.len != 0) try gpu.download(dbeta, beta.grad.data);
    if (r_values.len != 0) try gpu.download(dr, r.grad.data);
    try gpu.download(running[0..c_out], mean.data);
    try gpu.download(running[c_out..], variance.data);
}

fn expectConvBnClose(fused: []const f32, unfused: []const f32, abs: f32, rel: f32, max_flipped: usize) !void {
    var flipped: usize = 0;
    for (fused, unfused) |got, want| {
        if (!std.math.isFinite(got) or @abs(want - got) > abs + rel * @abs(want)) flipped += 1;
    }
    try testing.expect(flipped <= max_flipped);
}

// Compares buffers whose differences are dominated by ReLU mask flips: about 0.1% of the positions of a chain's
// output sit close enough to the boundary that the one-ulp difference of the virtual map (rebuilt with
// bf16-rounded scale and shift) flips the mask, and each flip redirects a whole gradient value. Per-element
// tolerances cannot hold there; the root-mean-square distance and the largest error relative to the largest
// wanted value are what separate that rounding cascade from a broken kernel.
fn expectConvBnNoise(fused: []const f32, unfused: []const f32, max_rms: f64, max_worst: f32) !void {
    var sq_diff: f64 = 0;
    var sq_want: f64 = 0;
    var worst: f32 = 0;
    var max_want: f32 = 0;
    for (fused, unfused) |got, want| {
        try testing.expect(std.math.isFinite(got));
        const err = @abs(want - got);
        if (err > worst) worst = err;
        sq_diff += @as(f64, err) * err;
        sq_want += @as(f64, want) * want;
        if (@abs(want) > max_want) max_want = @abs(want);
    }
    try testing.expect(@sqrt(sq_diff / sq_want) <= max_rms);
    try testing.expect(worst <= max_worst * max_want);
}

const ChainBufs = struct { out: []f32, dx: []f32, dw1: []f32, dw2: []f32, dg1: []f32, db1: []f32, dg2: []f32, db2: []f32, dr: []f32, run1: []f32, run2: []f32 };

fn runConvBnChain(comptime fused: bool, comptime n: usize, comptime side: usize, comptime c_in: usize, comptime c1: usize, comptime c2: usize, gpu: *gpu_mod.Gpu, x_values: []const f32, w1_values: []const f32, w2_values: []const f32, r_values: []const f32, dy_values: []const f32, bufs: ChainBufs) !void {
    const X = Tensor(f32, &.{ n, side, side, c_in });
    const W1 = Tensor(f32, &.{ 9 * c_in, c1 });
    const W2 = Tensor(f32, &.{ 9 * c1, c2 });
    const Y = Tensor(f32, &.{ n, side, side, c2 });
    const Ch1 = Tensor(f32, &.{c1});
    const Ch2 = Tensor(f32, &.{c2});
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true };
    const x = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    const w1 = try t.leaf(W1{ .data = try gpu.alloc(f32, W1.len) });
    const w2 = try t.leaf(W2{ .data = try gpu.alloc(f32, W2.len) });
    const r = try t.leaf(Y{ .data = try gpu.alloc(f32, Y.len) });
    try gpu.upload(x.value.data, x_values);
    try gpu.upload(w1.value.data, w1_values);
    try gpu.upload(w2.value.data, w2_values);
    try gpu.upload(r.value.data, r_values);
    try gpu.fill(x.grad.data, 0);
    try gpu.fill(w1.grad.data, 0);
    try gpu.fill(w2.grad.data, 0);
    try gpu.fill(r.grad.data, 0);
    const gamma1 = try t.leaf(Ch1{ .data = try gpu.alloc(f32, c1) });
    const beta1 = try t.leaf(Ch1{ .data = try gpu.alloc(f32, c1) });
    const gamma2 = try t.leaf(Ch2{ .data = try gpu.alloc(f32, c2) });
    const beta2 = try t.leaf(Ch2{ .data = try gpu.alloc(f32, c2) });
    try gpu.fill(gamma1.value.data, 1.5);
    try gpu.fill(beta1.value.data, 0.25);
    try gpu.fill(gamma2.value.data, 1.5);
    try gpu.fill(beta2.value.data, 0.25);
    try gpu.fill(gamma1.grad.data, 0);
    try gpu.fill(beta1.grad.data, 0);
    try gpu.fill(gamma2.grad.data, 0);
    try gpu.fill(beta2.grad.data, 0);
    const mean1 = Ch1{ .data = try gpu.alloc(f32, c1) };
    const variance1 = Ch1{ .data = try gpu.alloc(f32, c1) };
    const mean2 = Ch2{ .data = try gpu.alloc(f32, c2) };
    const variance2 = Ch2{ .data = try gpu.alloc(f32, c2) };
    try gpu.fill(mean1.data, 0);
    try gpu.fill(variance1.data, 1);
    try gpu.fill(mean2.data, 0);
    try gpu.fill(variance2.data, 1);

    const input = try t.cast(x, u16);
    const residual = try t.cast(r, u16);
    const stats1 = .{ .mean = mean1, .variance = variance1 };
    const stats2 = .{ .mean = mean2, .variance = variance2 };
    const y = if (fused)
        try t.convBnChain(input, w1, gamma1, beta1, stats1, w2, residual, gamma2, beta2, stats2, .{ .pad = 1 })
    else
        try t.convBatchNormAddRelu(try t.convBatchNormRelu(input, w1, gamma1, beta1, stats1, .{ .pad = 1 }), w2, residual, gamma2, beta2, stats2, .{ .pad = 1 });
    const back = try t.cast(y, f32);
    try gpu.download(bufs.out, back.value.data);
    try gpu.upload(back.grad.data, dy_values);
    var at = t.nodes.items.len;
    while (at > 0) {
        at -= 1;
        try t.nodes.items[at].run(&t, t.nodes.items[at].ctx);
    }
    try gpu.download(bufs.dx, x.grad.data);
    try gpu.download(bufs.dw1, w1.grad.data);
    try gpu.download(bufs.dw2, w2.grad.data);
    try gpu.download(bufs.dg1, gamma1.grad.data);
    try gpu.download(bufs.db1, beta1.grad.data);
    try gpu.download(bufs.dg2, gamma2.grad.data);
    try gpu.download(bufs.db2, beta2.grad.data);
    try gpu.download(bufs.dr, r.grad.data);
    try gpu.download(bufs.run1[0..c1], mean1.data);
    try gpu.download(bufs.run1[c1..], variance1.data);
    try gpu.download(bufs.run2[0..c2], mean2.data);
    try gpu.download(bufs.run2[c2..], variance2.data);
}

fn expectChainMatch(comptime n: usize, comptime side: usize, comptime c_in: usize, comptime c1: usize, comptime c2: usize, seed: u64, gpu_bytes: usize) !void {
    const gpu = try gpu_mod.Gpu.forTest(gpu_bytes);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const xl = n * side * side * c_in;
    const yl = n * side * side * c2;
    const w1l = 9 * c_in * c1;
    const w2l = 9 * c1 * c2;
    const x = try testing.allocator.alloc(f32, xl);
    defer testing.allocator.free(x);
    const w1 = try testing.allocator.alloc(f32, w1l);
    defer testing.allocator.free(w1);
    const w2 = try testing.allocator.alloc(f32, w2l);
    defer testing.allocator.free(w2);
    const r = try testing.allocator.alloc(f32, yl);
    defer testing.allocator.free(r);
    const dy = try testing.allocator.alloc(f32, yl);
    defer testing.allocator.free(dy);
    for (x) |*v| v.* = rng.floatNorm(f32) * 2 + 0.5;
    for (w1) |*v| v.* = rng.floatNorm(f32) * 0.05;
    for (w2) |*v| v.* = rng.floatNorm(f32) * 0.05;
    for (r) |*v| v.* = rng.floatNorm(f32);
    for (dy) |*v| v.* = rng.floatNorm(f32);
    const run_len = yl + xl + w1l + w2l + 4 * c1 + 4 * c2 + yl;
    const fused = try testing.allocator.alloc(f32, run_len);
    defer testing.allocator.free(fused);
    const plain = try testing.allocator.alloc(f32, run_len);
    defer testing.allocator.free(plain);
    const carve = struct {
        fn of(buf: []f32) ChainBufs {
            var at: usize = 0;
            var out: ChainBufs = undefined;
            inline for (.{ "out", "dx", "dw1", "dw2", "dg1", "db1", "dg2", "db2", "dr", "run1", "run2" }) |name| {
                const len = comptime blk: {
                    if (std.mem.eql(u8, name, "out") or std.mem.eql(u8, name, "dr")) break :blk yl;
                    if (std.mem.eql(u8, name, "dx")) break :blk xl;
                    if (std.mem.eql(u8, name, "dw1")) break :blk w1l;
                    if (std.mem.eql(u8, name, "dw2")) break :blk w2l;
                    if (std.mem.eql(u8, name, "run1")) break :blk 2 * c1;
                    if (std.mem.eql(u8, name, "run2")) break :blk 2 * c2;
                    if (name[name.len - 1] == '1') break :blk c1;
                    break :blk c2;
                };
                @field(out, name) = buf[at..][0..len];
                at += len;
            }
            return out;
        }
    }.of;
    const fused_bufs = carve(fused);
    const plain_bufs = carve(plain);
    try runConvBnChain(true, n, side, c_in, c1, c2, gpu, x, w1, w2, r, dy, fused_bufs);
    try runConvBnChain(false, n, side, c_in, c1, c2, gpu, x, w1, w2, r, dy, plain_bufs);
    // NOTE: the virtual map is rebuilt on load with bf16-rounded scale and shift (one extra bf16 step next to
    // the map's own bf16 storage), so the forward output lands within a rounding of the materialized path, and
    // the gradients pick up the mask-flip cascade described at expectConvBnNoise.
    try expectConvBnClose(fused_bufs.out, plain_bufs.out, 0.06, 0.01, yl / 1000);
    try expectConvBnNoise(fused_bufs.dx, plain_bufs.dx, 0.06, 0.3);
    try expectConvBnNoise(fused_bufs.dw1, plain_bufs.dw1, 0.06, 0.3);
    try expectConvBnNoise(fused_bufs.dw2, plain_bufs.dw2, 0.06, 0.3);
    try expectConvBnNoise(fused_bufs.dg1, plain_bufs.dg1, 0.06, 0.3);
    try expectConvBnNoise(fused_bufs.db1, plain_bufs.db1, 0.06, 0.3);
    try expectConvBnNoise(fused_bufs.dg2, plain_bufs.dg2, 0.06, 0.3);
    try expectConvBnNoise(fused_bufs.db2, plain_bufs.db2, 0.06, 0.3);
    try expectConvBnClose(fused_bufs.dr, plain_bufs.dr, 0.01, 0.01, yl / 500);
    try expectConvBnClose(fused_bufs.run1, plain_bufs.run1, 0.02, 0.02, 0);
    try expectConvBnClose(fused_bufs.run2, plain_bufs.run2, 0.02, 0.02, 0);

    // A deterministic device combines the per-block sums in order; both orders are valid roundings of the same sums.
    gpu.deterministic = true;
    defer gpu.deterministic = false;
    try runConvBnChain(true, n, side, c_in, c1, c2, gpu, x, w1, w2, r, dy, plain_bufs);
    try expectConvBnClose(fused_bufs.out, plain_bufs.out, 0.06, 0.01, yl / 1000);
    try expectConvBnClose(fused_bufs.dx, plain_bufs.dx, 0.05, 0.03, xl / 1000);
    try expectConvBnClose(fused_bufs.dw1, plain_bufs.dw1, 0.05, 0.03, w1l / 1000);
    try expectConvBnClose(fused_bufs.dw2, plain_bufs.dw2, 0.05, 0.03, w2l / 1000);
    try expectConvBnClose(fused_bufs.dg1, plain_bufs.dg1, 0.02, 0.02, 0);
    try expectConvBnClose(fused_bufs.db1, plain_bufs.db1, 0.02, 0.02, 0);
}

test "conv batch-norm chain around a virtual map matches the unfused pair, forward and backward" {
    try expectChainMatch(2, 16, 64, 64, 64, 51, 64 << 20);
}

test "conv batch-norm chain with a non-tile-multiple output width matches the unfused pair" {
    try expectChainMatch(2, 16, 64, 64, 96, 52, 64 << 20);
}

test "conv batch-norm chain on the wide tiles matches the unfused pair" {
    try expectChainMatch(12, 32, 64, 128, 128, 53, 256 << 20);
}

test "conv + batch norm with statistics from the conv epilogue matches the unfused path, forward and backward" {
    const gpu = try gpu_mod.Gpu.forTest(64 << 20);
    defer gpu.deinit();
    const n = 2;
    const side = 16;
    const c_in = 64;
    // NOTE: 96 output channels give the 64-wide tile a partial second tile, exercising the edge guards of the epilogue.
    const c_out = 96;
    var prng = std.Random.DefaultPrng.init(41);
    const rng = prng.random();
    const xl = n * side * side * c_in;
    const wl = 9 * c_in * c_out;
    const yl = n * side * side * c_out;
    const x = try testing.allocator.alloc(f32, xl);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, wl);
    defer testing.allocator.free(w);
    const dy = try testing.allocator.alloc(f32, yl);
    defer testing.allocator.free(dy);
    for (x) |*v| v.* = rng.floatNorm(f32) * 2 + 0.5;
    for (w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    for (dy) |*v| v.* = rng.floatNorm(f32);
    const bufs = try testing.allocator.alloc(f32, 2 * (yl + xl + wl + 4 * c_out));
    defer testing.allocator.free(bufs);
    const fused_out = bufs[0..yl];
    const fused_dx = bufs[yl..][0..xl];
    const fused_dw = bufs[yl + xl ..][0..wl];
    const fused_dg = bufs[yl + xl + wl ..][0..c_out];
    const fused_db = bufs[yl + xl + wl + c_out ..][0..c_out];
    const fused_running = bufs[yl + xl + wl + 2 * c_out ..][0 .. 2 * c_out];
    const plain_out = bufs[yl + xl + wl + 4 * c_out ..][0..yl];
    const plain_dx = bufs[2 * yl + xl + wl + 4 * c_out ..][0..xl];
    const plain_dw = bufs[2 * yl + 2 * xl + wl + 4 * c_out ..][0..wl];
    const plain_dg = bufs[2 * yl + 2 * xl + 2 * wl + 4 * c_out ..][0..c_out];
    const plain_db = bufs[2 * yl + 2 * xl + 2 * wl + 5 * c_out ..][0..c_out];
    const plain_running = bufs[2 * yl + 2 * xl + 2 * wl + 6 * c_out ..][0 .. 2 * c_out];
    try runConvBn(true, n, side, c_in, c_out, gpu, x, w, &.{}, dy, fused_out, fused_dx, fused_dw, fused_dg, fused_db, &.{}, fused_running);
    try runConvBn(false, n, side, c_in, c_out, gpu, x, w, &.{}, dy, plain_out, plain_dx, plain_dw, plain_dg, plain_db, &.{}, plain_running);
    // NOTE: the two reductions sum in a different order, so an occasional output lands one bf16 ulp apart.
    try expectConvBnClose(fused_out, plain_out, 0.06, 0.01, yl / 1000);
    try expectConvBnClose(fused_dx, plain_dx, 0.05, 0.03, xl / 1000);
    try expectConvBnClose(fused_dw, plain_dw, 0.05, 0.03, wl / 1000);
    try expectConvBnClose(fused_dg, plain_dg, 0.02, 0.02, 0);
    try expectConvBnClose(fused_db, plain_db, 0.02, 0.02, 0);
    try expectConvBnClose(fused_running, plain_running, 0.02, 0.02, 0);

    // A deterministic device combines the per-block sums in order; both orders are valid roundings of the same sums.
    gpu.deterministic = true;
    defer gpu.deterministic = false;
    try runConvBn(true, n, side, c_in, c_out, gpu, x, w, &.{}, dy, plain_out, plain_dx, plain_dw, plain_dg, plain_db, &.{}, plain_running);
    try expectConvBnClose(fused_out, plain_out, 0.06, 0.01, yl / 1000);
    try expectConvBnClose(fused_dx, plain_dx, 0.05, 0.03, xl / 1000);
    try expectConvBnClose(fused_dw, plain_dw, 0.05, 0.03, wl / 1000);
}

test "conv + batch norm + residual with statistics from the conv epilogue matches the unfused path" {
    const gpu = try gpu_mod.Gpu.forTest(64 << 20);
    defer gpu.deinit();
    const n = 2;
    const side = 16;
    const c = 64;
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const xl = n * side * side * c;
    const wl = 9 * c * c;
    const x = try testing.allocator.alloc(f32, xl);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, wl);
    defer testing.allocator.free(w);
    const r = try testing.allocator.alloc(f32, xl);
    defer testing.allocator.free(r);
    const dy = try testing.allocator.alloc(f32, xl);
    defer testing.allocator.free(dy);
    for (x) |*v| v.* = rng.floatNorm(f32) * 2 + 0.5;
    for (w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    for (r) |*v| v.* = rng.floatNorm(f32);
    for (dy) |*v| v.* = rng.floatNorm(f32);
    const fused = try testing.allocator.alloc(f32, 3 * xl + wl + 4 * c);
    defer testing.allocator.free(fused);
    const plain = try testing.allocator.alloc(f32, fused.len);
    defer testing.allocator.free(plain);
    const fused_out, const fused_dx, const fused_dw, const fused_dr, const fused_g, const fused_b, const fused_run = .{ fused[0..xl], fused[xl..][0..xl], fused[2 * xl ..][0..wl], fused[2 * xl + wl ..][0..xl], fused[2 * xl + wl + xl ..][0..c], fused[2 * xl + wl + xl + c ..][0..c], fused[2 * xl + wl + xl + 2 * c ..][0 .. 2 * c] };
    const plain_out, const plain_dx, const plain_dw, const plain_dr, const plain_g, const plain_b, const plain_run = .{ plain[0..xl], plain[xl..][0..xl], plain[2 * xl ..][0..wl], plain[2 * xl + wl ..][0..xl], plain[2 * xl + wl + xl ..][0..c], plain[2 * xl + wl + xl + c ..][0..c], plain[2 * xl + wl + xl + 2 * c ..][0 .. 2 * c] };
    try runConvBn(true, n, side, c, c, gpu, x, w, r, dy, fused_out, fused_dx, fused_dw, fused_g, fused_b, fused_dr, fused_run);
    try runConvBn(false, n, side, c, c, gpu, x, w, r, dy, plain_out, plain_dx, plain_dw, plain_g, plain_b, plain_dr, plain_run);
    try expectConvBnClose(fused_out, plain_out, 0.06, 0.01, xl / 1000);
    try expectConvBnClose(fused_dx, plain_dx, 0.05, 0.03, xl / 1000);
    try expectConvBnClose(fused_dw, plain_dw, 0.05, 0.03, wl / 1000);
    try expectConvBnClose(fused_dr, plain_dr, 0.01, 0.01, 0);
    try expectConvBnClose(fused_g, plain_g, 0.02, 0.02, 0);
    try expectConvBnClose(fused_b, plain_b, 0.02, 0.02, 0);
    try expectConvBnClose(fused_run, plain_run, 0.02, 0.02, 0);
}

test "conv + batch norm epilogue statistics on the wide tile match the unfused path" {
    const gpu = try gpu_mod.Gpu.forTest(256 << 20);
    defer gpu.deinit();
    const n = 12;
    const side = 32;
    const c_in = 64;
    const c_out = 128;
    var prng = std.Random.DefaultPrng.init(43);
    const rng = prng.random();
    const xl = n * side * side * c_in;
    const wl = 9 * c_in * c_out;
    const yl = n * side * side * c_out;
    const x = try testing.allocator.alloc(f32, xl);
    defer testing.allocator.free(x);
    const w = try testing.allocator.alloc(f32, wl);
    defer testing.allocator.free(w);
    const dy = try testing.allocator.alloc(f32, yl);
    defer testing.allocator.free(dy);
    for (x) |*v| v.* = rng.floatNorm(f32);
    for (w) |*v| v.* = rng.floatNorm(f32) * 0.05;
    for (dy) |*v| v.* = rng.floatNorm(f32);
    const fused = try testing.allocator.alloc(f32, yl + 2 * c_out);
    defer testing.allocator.free(fused);
    const plain = try testing.allocator.alloc(f32, fused.len);
    defer testing.allocator.free(plain);
    try runConvBn(true, n, side, c_in, c_out, gpu, x, w, &.{}, dy, fused[0..yl], &.{}, &.{}, &.{}, &.{}, &.{}, fused[yl..]);
    try runConvBn(false, n, side, c_in, c_out, gpu, x, w, &.{}, dy, plain[0..yl], &.{}, &.{}, &.{}, &.{}, &.{}, plain[yl..]);
    try expectConvBnClose(fused[0..yl], plain[0..yl], 0.06, 0.01, yl / 1000);
    try expectConvBnClose(fused[yl..], plain[yl..], 0.02, 0.02, 0);
}

test "softmax rows, with and without a causal mask, have correct values and gradients on the cpu and the gpu" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator() };
    const x = try t.leaf(try Tensor(f32, &.{ 3, 3 }).fromSlice(t.gpa, &.{ 0, 0, 0, 0, 0, 0, 1, 2, 3 }));
    const p = try t.softmax(x, .{ .causal_period = 3 });
    try testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0.5, 0.5, 0, 0.09003057, 0.24472848, 0.66524094 }, p.value.data);
    // A scale of 0.5 on the last row (1, 2, 3) is a softmax of (0.5, 1, 1.5), computed here with the standard exp.
    const halved = try t.softmax(x, .{ .scale = 0.5 });
    const total = @exp(@as(f64, 0.5)) + @exp(@as(f64, 1)) + @exp(@as(f64, 1.5));
    for (halved.value.data[6..], [_]f64{ 0.5, 1, 1.5 }) |got, v| try testing.expectApproxEqAbs(@as(f32, @floatCast(@exp(v) / total)), got, 1e-6);
    inline for (.{ 0, 3 }) |period| {
        try gradcheck.expectFiniteDifferences(SoftmaxNet(period, 6, 3, 1), 7, 1e-2, 3e-3);
        try gradcheck.expectGpuMatchesCpu(SoftmaxNet(period, 6, 3, 1), 8, 1e-5);
    }
    // Rows wider than a warp, one of them causal with a period shorter than the row (the rest of the row is masked).
    try gradcheck.expectGpuMatchesCpu(SoftmaxNet(0, 37, 200, 1), 15, 1e-5);
    try gradcheck.expectGpuMatchesCpu(SoftmaxNet(40, 40, 200, 1), 16, 1e-5);
    // A scaled softmax, as attention uses it: the scale must reach both the values and the input gradient.
    try gradcheck.expectFiniteDifferences(SoftmaxNet(3, 6, 3, 0.35), 9, 1e-2, 3e-3);
    try gradcheck.expectGpuMatchesCpu(SoftmaxNet(40, 40, 200, 0.35), 10, 1e-5);
    try gradcheck.expectGpuMatchesCpu(WideCrossEntropyNet, 17, 1e-4);
}

test "linear has correct gradients for the input, the weights and the bias, on the cpu and the gpu" {
    try gradcheck.expectFiniteDifferences(LinearNet(4, 3, 5), 18, 1e-2, 3e-3);
    try gradcheck.expectGpuMatchesCpu(LinearNet(4, 3, 5), 19, 1e-5);
    try gradcheck.expectGpuMatchesCpu(LinearNet(130, 70, 150), 20, 1e-4);
}
// Runs add (per-row vector), relu and mul (equal shapes) forward and backward on tensors big enough to be split across threads.
fn elementwiseOnPool(par: lib.kernels.Parallel, out_x: []f32, out_b: []f32) !f32 {
    const rows = 512;
    const cols = 256;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true };
    var prng = std.Random.DefaultPrng.init(31);
    const x = try t.leaf(try Tensor(f32, &.{ rows, cols }).empty(t.gpa));
    const b = try t.leaf(try Tensor(f32, &.{cols}).empty(t.gpa));
    const y = try t.constant(try Tensor(f32, &.{ rows, cols }).empty(t.gpa));
    for (x.value.data) |*v| v.* = prng.random().floatNorm(f32);
    for (b.value.data) |*v| v.* = prng.random().floatNorm(f32);
    for (y.value.data) |*v| v.* = prng.random().floatNorm(f32);
    @memset(x.grad.data, 0);
    @memset(b.grad.data, 0);
    const activated = try t.relu(try t.add(x, b));
    const loss = try t.mse(try t.mul(activated, x), y);
    try t.backward(loss);
    @memcpy(out_x, x.grad.data);
    @memcpy(out_b, b.grad.data);
    return loss.value.data[0];
}

test "cpu element-wise ops give identical results on the worker pool and on one thread" {
    const gpa = testing.allocator;
    const one_x = try gpa.alloc(f32, 512 * 256);
    defer gpa.free(one_x);
    const many_x = try gpa.alloc(f32, 512 * 256);
    defer gpa.free(many_x);
    var one_b: [256]f32 = undefined;
    var many_b: [256]f32 = undefined;
    const one_loss = try elementwiseOnPool(.{}, one_x, &one_b);
    const many_loss = try elementwiseOnPool(.{ .io = testing.io, .threads = 4 }, many_x, &many_b);
    try testing.expect(one_loss > 0);
    // The single-thread run takes the small-tensor path of the same kernels, so also check one element by hand:
    // loss = mean((relu(x + b) * x - y)^2), so d loss / d x[i] = 2 * (relu(x + b) * x - y) / n * (relu(x + b) + x * (x + b > 0)).
    var prng = std.Random.DefaultPrng.init(31);
    const rng = prng.random();
    const xs = try gpa.alloc(f32, 512 * 256);
    defer gpa.free(xs);
    var bs: [256]f32 = undefined;
    const ys = try gpa.alloc(f32, 512 * 256);
    defer gpa.free(ys);
    for (xs) |*v| v.* = rng.floatNorm(f32);
    for (&bs) |*v| v.* = rng.floatNorm(f32);
    for (ys) |*v| v.* = rng.floatNorm(f32);
    var want_loss: f64 = 0;
    for (xs, ys, 0..) |x, y, i| {
        const pre = x + bs[i % 256];
        const r = @max(pre, 0);
        want_loss += (r * x - y) * (r * x - y);
    }
    try testing.expectApproxEqAbs(@as(f32, @floatCast(want_loss / @as(f64, @floatFromInt(xs.len)))), one_loss, 1e-4);
    for ([_]usize{ 0, 1, 255, 256, 70000, 131071 }) |i| {
        const pre = xs[i] + bs[i % 256];
        const r = @max(pre, 0);
        const slope = r + xs[i] * (if (pre > 0) @as(f32, 1) else 0);
        const want = 2 * (r * xs[i] - ys[i]) / @as(f32, @floatFromInt(xs.len)) * slope;
        try testing.expectApproxEqAbs(want, one_x[i], 1e-6);
    }
    try testing.expectEqual(one_loss, many_loss);
    try testing.expectEqualSlices(f32, one_x, many_x);
    try testing.expectEqualSlices(f32, &one_b, &many_b);
}
fn chunkedConvolution(par: lib.kernels.Parallel, out_y: []f32, out_dw: []f32, out_dx: []f32) !void {
    const n = 30;
    const size = 28;
    const c_out = 4;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true };
    var prng = std.Random.DefaultPrng.init(37);
    const rng = prng.random();
    const x = try t.leaf(try Tensor(f32, &.{ n, size, size, 1 }).empty(t.gpa));
    const w = try t.leaf(try Tensor(f32, &.{ 25, c_out }).empty(t.gpa));
    const target = try t.constant(try Tensor(f32, &.{ n, size * size * c_out }).empty(t.gpa));
    for (x.value.data) |*v| v.* = rng.floatNorm(f32);
    for (w.value.data) |*v| v.* = rng.floatNorm(f32);
    for (target.value.data) |*v| v.* = rng.floatNorm(f32);
    @memset(w.grad.data, 0);
    @memset(x.grad.data, 0);
    const y = try t.conv2d(x, w, .{ .pad = 2 });
    const loss = try t.mse(reshape(y, &.{ n, size * size * c_out }), target);
    try t.backward(loss);
    @memcpy(out_y, y.value.data);
    @memcpy(out_dw, w.grad.data);
    @memcpy(out_dx, x.grad.data);

    // Direct reference in double precision: y = sum x * w over the 5 x 5 window, dw = sum x * dy with dy = 2 (y - target) / count.
    const count: f64 = @floatFromInt(y.value.data.len);
    var want_dw: [25 * c_out]f64 = @splat(0);
    var want_dx: [n * size * size]f64 = @splat(0);
    for (0..n) |b| {
        for (0..size) |oy| {
            for (0..size) |ox| {
                for (0..c_out) |oc| {
                    var sum: f64 = 0;
                    for (0..25) |j| {
                        const py = oy + j / 5;
                        const px = ox + j % 5;
                        if (py < 2 or px < 2 or py - 2 >= size or px - 2 >= size) continue;
                        sum += @as(f64, x.value.data[(b * size + py - 2) * size + px - 2]) * w.value.data[j * c_out + oc];
                    }
                    const at = ((b * size + oy) * size + ox) * c_out + oc;
                    try testing.expectApproxEqAbs(@as(f32, @floatCast(sum)), y.value.data[at], 1e-5);
                    const dy = 2 * (sum - target.value.data[at]) / count;
                    for (0..25) |j| {
                        const py = oy + j / 5;
                        const px = ox + j % 5;
                        if (py < 2 or px < 2 or py - 2 >= size or px - 2 >= size) continue;
                        want_dw[j * c_out + oc] += @as(f64, x.value.data[(b * size + py - 2) * size + px - 2]) * dy;
                        want_dx[(b * size + py - 2) * size + px - 2] += @as(f64, w.value.data[j * c_out + oc]) * dy;
                    }
                }
            }
        }
    }
    for (want_dw, w.grad.data) |want, got| try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got, 1e-5);
    for (want_dx, x.grad.data) |want, got| try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got, 1e-5);
}

test "a cpu convolution built in chunks of rows matches a direct reference, outputs and both gradients, on one thread and on the pool" {
    // 30 x 28 x 28 = 23520 positions of a 5 x 5 window: more than one chunk of the im2col matrix, with a shorter last chunk.
    var one_y: [30 * 28 * 28 * 4]f32 = undefined;
    var many_y: [30 * 28 * 28 * 4]f32 = undefined;
    var one_dw: [100]f32 = undefined;
    var many_dw: [100]f32 = undefined;
    var one_dx: [30 * 28 * 28]f32 = undefined;
    var many_dx: [30 * 28 * 28]f32 = undefined;
    try chunkedConvolution(.{}, &one_y, &one_dw, &one_dx);
    try chunkedConvolution(.{ .io = testing.io, .threads = 4 }, &many_y, &many_dw, &many_dx);
    try testing.expectEqualSlices(f32, &one_dx, &many_dx);
    try testing.expectEqualSlices(f32, &one_y, &many_y);
    for (one_dw, many_dw) |a, b| try testing.expectApproxEqAbs(a, b, 1e-5);
}

test "the fused gru ops have correct gradients for every input, on the cpu and the gpu" {
    try gradcheck.expectFiniteDifferences(GruOpsNet, 11, 1e-2, 5e-3);
    try gradcheck.expectGpuMatchesCpu(GruOpsNet, 12, 1e-5);
}

test "layer norm has correct gradients for the input, the scale and the shift, on the cpu and the gpu" {
    try gradcheck.expectFiniteDifferences(LayerNormNet(5, 6), 9, 1e-2, 5e-3);
    try gradcheck.expectGpuMatchesCpu(LayerNormNet(5, 6), 10, 1e-4);
    // Rows wider than a warp, and not a multiple of its width, make each lane loop more than once.
    try gradcheck.expectGpuMatchesCpu(LayerNormNet(37, 200), 13, 1e-4);
    try gradcheck.expectGpuMatchesCpu(LayerNormNet(9, 32), 14, 1e-4);
}

test "embedding gathers rows by id and scatters the gradient back, repeated ids accumulate, on the cpu and the gpu" {
    const gpu = try gpu_mod.Gpu.forTest(1 << 20);
    defer gpu.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const T = Tensor(f32, &.{ 5, 3 });
    const table_values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const host_ids = [_]u32{ 4, 1, 1, 0 };
    const expected = [_]f32{ 13, 14, 15, 4, 5, 6, 4, 5, 6, 1, 2, 3 };
    const expected_grad = [_]f32{ 10, 11, 12, 11, 13, 15, 0, 0, 0, 0, 0, 0, 1, 2, 3 };
    var upstream: [12]f32 = undefined;
    for (&upstream, 0..) |*g, i| g.* = @floatFromInt(i + 1);

    var on_cpu: Tape = .{ .gpa = arena.allocator() };
    const table = try on_cpu.leaf(try T.fromSlice(on_cpu.gpa, &table_values));
    const rows = try on_cpu.embedding(table, 4, &host_ids, f32);
    try testing.expectEqualSlices(f32, &expected, rows.value.data);
    @memcpy(rows.grad.data, &upstream);
    try on_cpu.nodes.items[0].run(&on_cpu, on_cpu.nodes.items[0].ctx);
    try testing.expectEqualSlices(f32, &expected_grad, table.grad.data);

    const device_table = try gpu.alloc(f32, T.len);
    const device_grad = try gpu.alloc(f32, T.len);
    const device_ids = try gpu.alloc(u32, host_ids.len);
    try gpu.upload(device_table, &table_values);
    try gpu.upload(device_ids, &host_ids);
    try gpu.fill(device_grad, 0);
    var on_gpu: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const gpu_table: Var(T) = .{ .value = .{ .data = device_table }, .grad = .{ .data = device_grad }, .needs_grad = true };
    const gathered = try on_gpu.embedding(gpu_table, 4, device_ids, f32);
    var host: [12]f32 = undefined;
    try gpu.download(&host, gathered.value.data);
    try testing.expectEqualSlices(f32, &expected, &host);
    try gpu.upload(gathered.grad.data, &upstream);
    try on_gpu.nodes.items[0].run(&on_gpu, on_gpu.nodes.items[0].ctx);
    var grads: [15]f32 = undefined;
    try gpu.download(&grads, device_grad);
    try testing.expectEqualSlices(f32, &expected_grad, &grads);
}
test "batched matrix products, plain and with a transposed right operand, have correct gradients on the cpu and the gpu" {
    inline for (.{ false, true }) |trans_b| {
        try gradcheck.expectFiniteDifferences(BatchedNet(trans_b), 11, 1e-2, 5e-3);
        try gradcheck.expectGpuMatchesCpu(BatchedNet(trans_b), 12, 1e-4);
    }
}

test "label smoothing blends the target and its gradient with the uniform distribution" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator(), .smoothing = 0.2 };
    const x = try t.leaf(try Tensor(f32, &.{ 1, 2 }).fromSlice(t.gpa, &.{ 1, 0 }));
    const loss = try t.crossEntropy(x, &.{0});
    try t.backward(loss);

    try testing.expectApproxEqAbs(@as(f32, 0.4133), loss.value.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -0.1689), x.grad.data[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.1689), x.grad.data[1], 1e-4);
}

test "gradients with label smoothing match finite differences and the gpu" {
    try gradcheck.expectFiniteDifferences(SmoothedNet, 7, 5e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(SmoothedNet, 21, 1e-5);
}

test "gradients of a small network match finite differences" {
    try gradcheck.expectFiniteDifferences(SmallNet, 7, 5e-3, 3e-3);
}

test "gradients of a convolution, pooling and linear network match finite differences and the gpu" {
    try gradcheck.expectFiniteDifferences(ConvPoolNet, 5, 1e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(ConvPoolNet, 8, 1e-5);
    try gradcheck.expectTensorCoresClose(ConvPoolNet, 8, 3e-2);
}

test "conv2d with padding and stride matches a direct convolution, finite differences and the gpu" {
    inline for (.{ ConvOpts{}, ConvOpts{ .stride = 1, .pad = 1 }, ConvOpts{ .stride = 2, .pad = 1 }, ConvOpts{ .stride = 2, .pad = 0 } }) |opts| {
        inline for (.{ .{ 2, 3 }, .{ 4, 8 }, .{ 8, 4 } }) |channels| {
            const Case = ConvCase(opts, 3, channels[0], channels[1]);
            try Case.expectDirectConvolution();
            try gradcheck.expectFiniteDifferences(Case, 31, 1e-3, 3e-3);
            try gradcheck.expectGpuMatchesCpu(Case, 32, 1e-5);
            try gradcheck.expectTensorCoresClose(Case, 32, 3e-2);
            try gradcheck.expectCudnnClose(Case, 32, 3e-2);
        }
    }
}

test "grouped, depthwise and dilated convolutions match a direct convolution, finite differences and the gpu" {
    inline for (.{
        .{ ConvOpts{ .pad = 1, .groups = 2 }, 3, 4, 6 },
        .{ ConvOpts{ .pad = 1, .groups = 4 }, 3, 4, 4 },
        .{ ConvOpts{ .pad = 2, .dilation = 2 }, 3, 3, 4 },
        .{ ConvOpts{ .stride = 2, .pad = 2, .dilation = 2, .groups = 2 }, 3, 4, 2 },
        .{ ConvOpts{ .dilation = 2, .groups = 2 }, 2, 2, 4 },
        // Padding beyond the reverse geometry (dilation * (k - 1) < pad) used to fail to compile.
        .{ ConvOpts{ .pad = 1 }, 1, 3, 4 },
        .{ ConvOpts{ .pad = 1, .groups = 2 }, 1, 4, 6 },
    }) |config| {
        const Case = ConvCase(config[0], config[1], config[2], config[3]);
        try Case.expectDirectConvolution();
        try gradcheck.expectFiniteDifferences(Case, 34, 1e-3, 3e-3);
        try gradcheck.expectGpuMatchesCpu(Case, 35, 1e-5);
    }
}

test "the wide 128x128 tensor-core kernels agree with the cpu on a grid large enough to select them" {
    try gradcheck.expectTensorCoresClose(WideConvNet, 12, 0.2);
}

test "dilated convolutions on the implicit GEMM match the cpu" {
    try gradcheck.expectGpuMatchesCpu(DilatedConvNet, 41, 1e-4);
    try gradcheck.expectTensorCoresClose(DilatedConvNet, 42, 0.2);
}

test "grouped convolutions on the implicit GEMM match the cpu" {
    try gradcheck.expectGpuMatchesCpu(GroupedConvNet, 43, 1e-4);
    try gradcheck.expectTensorCoresClose(GroupedConvNet, 44, 3e-2);
}

test "implicit-GEMM convolutions on tensor cores match a direct convolution and the cpu" {
    inline for (.{ .{ ConvOpts{ .pad = 1 }, 3, 64, 32 }, .{ ConvOpts{}, 3, 64, 64 }, .{ ConvOpts{ .pad = 1 }, 3, 128, 64 }, .{ ConvOpts{}, 1, 64, 128 } }) |config| {
        const Case = ConvCase(config[0], config[1], config[2], config[3]);
        try Case.expectDirectConvolution();
        try gradcheck.expectGpuMatchesCpu(Case, 33, 1e-4);
        try gradcheck.expectTensorCoresClose(Case, 33, 0.15);
        try gradcheck.expectCudnnClose(Case, 33, 0.15);
        try gradcheck.expectTf32Close(Case, 33, 0.1);
    }
}
test "batch norm gradients match finite differences, and the gpu matches the cpu including running statistics" {
    try gradcheck.expectFiniteDifferences(BnCase, 41, 1e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(BnCase, 43, 1e-5);
}
test "average pooling takes window means and spreads gradients evenly, also on the gpu" {
    const gpu = try gpu_mod.Gpu.forTest(1 << 20);
    defer gpu.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const Img = Tensor(f32, &.{ 1, 2, 4, 2 });
    const values = [_]f32{ 1, 10, 2, 20, 5, 0, 6, 0, 3, 30, 4, 40, 7, 0, 8, 0 };

    var t: Tape = .{ .gpa = arena.allocator() };
    const x = try t.leaf(try Img.fromSlice(t.gpa, &values));
    const y = try t.avgPool(x, .{});
    try testing.expectEqualSlices(f32, &.{ 2.5, 25, 6.5, 0 }, y.value.data);
    @memset(y.grad.data, 4);
    try t.nodes.items[0].run(&t, t.nodes.items[0].ctx);
    for (x.grad.data) |g| try testing.expectEqual(@as(f32, 1), g);

    const dx = try gpu.alloc(f32, Img.len);
    const dev_x = try gpu.alloc(f32, Img.len);
    try gpu.upload(dev_x, &values);
    try gpu.fill(dx, 0);
    var gpu_tape: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const gy = try gpu_tape.avgPool(Var(Img){ .value = .{ .data = dev_x }, .grad = .{ .data = dx }, .needs_grad = true }, .{});
    var back: [4]f32 = undefined;
    try gpu.download(&back, gy.value.data);
    try testing.expectEqualSlices(f32, y.value.data, &back);
    try gpu.fill(gy.grad.data, 4);
    try gpu_tape.nodes.items[0].run(&gpu_tape, gpu_tape.nodes.items[0].ctx);
    var grads: [16]f32 = undefined;
    try gpu.download(&grads, dx);
    for (grads) |g| try testing.expectEqual(@as(f32, 1), g);
}

const OpsResult = struct { out: [64 * 16]f32, grad_x: [64 * 16]f32, grad_bias: [16]f32, grad_gamma: [16]f32, grad_beta: [16]f32 };

fn runElementwiseChain(comptime half: bool, gpu: *gpu_mod.Gpu, x_values: []const f32, scale_values: []const f32, g_values: []const f32, result: *OpsResult) !void {
    const X = Tensor(f32, &.{ 64, 16 });
    const Row = Tensor(f32, &.{16});
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true };
    const x = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    const bias = try t.leaf(Row{ .data = try gpu.alloc(f32, 16) });
    const gamma = try t.leaf(Row{ .data = try gpu.alloc(f32, 16) });
    const beta = try t.leaf(Row{ .data = try gpu.alloc(f32, 16) });
    try gpu.upload(x.value.data, x_values);
    try gpu.upload(bias.value.data, scale_values[0..16]);
    try gpu.upload(gamma.value.data, scale_values[16..32]);
    try gpu.upload(beta.value.data, scale_values[32..48]);
    inline for (.{ x, bias, gamma, beta }) |v| try gpu.fill(v.grad.data, 0);

    const input = if (half) try t.cast(x, u16) else x;
    const shift = if (half) try t.cast(bias, u16) else bias;
    const normed = try t.layerNorm(input, gamma, beta);
    const summed = try t.add(normed, shift);
    const product = try t.mul(summed, normed);
    const smooth = try t.activate(product, .gelu);
    const rectified = try t.relu(smooth);
    const thinned = try t.dropout(rectified, 0.8);
    // NOTE: the gain makes the softmax inputs large enough that its output depends on every op before it; near zero it is almost uniform.
    const gain = try t.constant(try t.filledTensor(Tensor(f32, &.{}), 6));
    const sharpened = try t.mul(thinned, if (half) try t.cast(gain, u16) else gain);
    const probabilities = try t.cast(try t.softmax(sharpened, .{ .causal_period = 4 }), f32);
    try gpu.download(&result.out, probabilities.value.data);
    try gpu.upload(probabilities.grad.data, g_values);
    var at = t.nodes.items.len;
    while (at > 0) {
        at -= 1;
        try t.nodes.items[at].run(&t, t.nodes.items[at].ctx);
    }
    try gpu.download(&result.grad_x, x.grad.data);
    try gpu.download(&result.grad_bias, bias.grad.data);
    try gpu.download(&result.grad_gamma, gamma.grad.data);
    try gpu.download(&result.grad_beta, beta.grad.data);
}

// Counts the elements that differ by more than a small share of the largest reference value.
fn countOutliers(want: []const f32, got: []const f32) usize {
    var largest: f32 = 0;
    for (want) |v| largest = @max(largest, @abs(v));
    var outliers: usize = 0;
    for (want, got) |w, g| {
        if (@abs(w - g) > 0.06 * largest + 0.03 * @abs(w)) outliers += 1;
    }
    return outliers;
}

// An independent double precision version of the chain that runChain builds: layer norm, bias, product, gelu, relu, dropout,
// gain and a causal softmax. It shares nothing with the kernels except the dropout hash, which has its own test.
fn chainReference(x: []const f64, bias: []const f64, gamma: []const f64, beta: []const f64, out: []f64) void {
    for (0..64) |r| {
        var mean: f64 = 0;
        for (x[r * 16 ..][0..16]) |v| mean += v / 16;
        var variance: f64 = 0;
        for (x[r * 16 ..][0..16]) |v| variance += (v - mean) * (v - mean) / 16;
        const inv_std = 1 / @sqrt(variance + 1e-5);
        var z: [16]f64 = undefined;
        for (0..16) |j| {
            const normed = gamma[j] * (x[r * 16 + j] - mean) * inv_std + beta[j];
            const product = (normed + bias[j]) * normed;
            const inner = 0.7978845608 * (product + 0.044715 * product * product * product);
            const smooth = 0.5 * product * (1 + std.math.tanh(inner));
            const kept = @as(f32, @floatFromInt(drop.hash(0, r * 16 + j) >> 40)) * (1.0 / 16777216.0) < 0.8;
            z[j] = 6 * (if (kept) @max(smooth, 0) / 0.8 else 0);
        }
        const limit = r % 4 + 1;
        var largest: f64 = z[0];
        for (z[1..limit]) |v| largest = @max(largest, v);
        var total: f64 = 0;
        for (z[0..limit]) |v| total += @exp(v - largest);
        for (0..16) |j| out[r * 16 + j] = if (j < limit) @exp(z[j] - largest) / total else 0;
    }
}

fn chainLoss(x: []const f64, bias: []const f64, gamma: []const f64, beta: []const f64, g: []const f32) f64 {
    var out: [64 * 16]f64 = undefined;
    chainReference(x, bias, gamma, beta, &out);
    var total: f64 = 0;
    for (out, g) |p, weight| total += p * weight;
    return total;
}

// Central differences of the reference loss with respect to one input vector.
fn chainGradient(comptime which: usize, inputs: [4][]f64, g: []const f32, grad: []f64) void {
    const h = 1e-6;
    for (inputs[which], 0..) |*v, i| {
        const kept = v.*;
        v.* = kept + h;
        const up = chainLoss(inputs[0], inputs[1], inputs[2], inputs[3], g);
        v.* = kept - h;
        const down = chainLoss(inputs[0], inputs[1], inputs[2], inputs[3], g);
        v.* = kept;
        grad[i] = (up - down) / (2 * h);
    }
}

fn largestOf(values: []const f64) f64 {
    var largest: f64 = 0;
    for (values) |v| largest = @max(largest, @abs(v));
    return largest;
}

test "layer norm, add, mul, gelu, relu, dropout and softmax on bf16 activations match the f32 path within bf16 rounding" {
    const gpu = try gpu_mod.Gpu.forTest(8 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(41);
    const rng = prng.random();
    var x: [64 * 16]f32 = undefined;
    var g: [64 * 16]f32 = undefined;
    var scales: [48]f32 = undefined;
    for (&x) |*v| v.* = rng.floatNorm(f32) * 2 + 0.5;
    for (&g) |*v| v.* = rng.floatNorm(f32);
    for (&scales) |*v| v.* = rng.floatNorm(f32) * 0.5 + 1;
    var wide: OpsResult = undefined;
    var half: OpsResult = undefined;
    try runElementwiseChain(false, gpu, &x, &scales, &g, &wide);
    try runElementwiseChain(true, gpu, &x, &scales, &g, &half);
    for (wide.out, half.out) |want, got| try testing.expectApproxEqAbs(want, got, 0.02 + 0.02 * @abs(want));
    try testing.expect(countOutliers(&wide.grad_x, &half.grad_x) <= 10);
    try testing.expectEqual(@as(usize, 0), countOutliers(&wide.grad_bias, &half.grad_bias));
    try testing.expectEqual(@as(usize, 0), countOutliers(&wide.grad_gamma, &half.grad_gamma));
    try testing.expectEqual(@as(usize, 0), countOutliers(&wide.grad_beta, &half.grad_beta));

    // The f32 run must also agree with the independent reference, so that a mistake in code shared by both paths shows up.
    var xs: [64 * 16]f64 = undefined;
    var shift: [16]f64 = undefined;
    var scale: [16]f64 = undefined;
    var offset: [16]f64 = undefined;
    for (x, &xs) |v, *d| d.* = v;
    for (scales[0..16], &shift) |v, *d| d.* = v;
    for (scales[16..32], &scale) |v, *d| d.* = v;
    for (scales[32..48], &offset) |v, *d| d.* = v;
    var want_out: [64 * 16]f64 = undefined;
    chainReference(&xs, &shift, &scale, &offset, &want_out);
    try expectCloseToReference(&want_out, &wide.out, 0.01, 2e-3);
    const inputs = [4][]f64{ &xs, &shift, &scale, &offset };
    var want_x: [64 * 16]f64 = undefined;
    var want_bias: [16]f64 = undefined;
    var want_gamma: [16]f64 = undefined;
    var want_beta: [16]f64 = undefined;
    chainGradient(0, inputs, &g, &want_x);
    chainGradient(1, inputs, &g, &want_bias);
    chainGradient(2, inputs, &g, &want_gamma);
    chainGradient(3, inputs, &g, &want_beta);
    try expectCloseToReference(&want_x, &wide.grad_x, 0.01, 0.01 * largestOf(&want_x));
    try expectCloseToReference(&want_bias, &wide.grad_bias, 0.01, 0.01 * largestOf(&want_bias));
    try expectCloseToReference(&want_gamma, &wide.grad_gamma, 0.01, 0.01 * largestOf(&want_gamma));
    try expectCloseToReference(&want_beta, &wide.grad_beta, 0.01, 0.01 * largestOf(&want_beta));
}

fn rounded(v: f32) f32 {
    return bf16_math.decode(bf16_math.encode(v));
}

fn runBackwardOf(t: *Tape) !void {
    var at = t.nodes.items.len;
    while (at > 0) {
        at -= 1;
        try t.nodes.items[at].run(t, t.nodes.items[at].ctx);
    }
}

fn expectCloseToReference(want: []const f64, got: []const f32, relative: f64, absolute: f64) !void {
    for (want, got) |w, g| try testing.expectApproxEqAbs(@as(f32, @floatCast(w)), g, @as(f32, @floatCast(absolute + relative * @abs(w))));
}

fn runMatmulChain(comptime half: bool, gpu: *gpu_mod.Gpu, x_values: []const f32, w_values: []const f32, g_values: []const f32, out: []f32, grad_x: []f32, grad_w: []f32) !void {
    const X = Tensor(f32, &.{ 70, 50 });
    const W = Tensor(f32, &.{ 50, 45 });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const x = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    const w = try t.leaf(W{ .data = try gpu.alloc(f32, W.len) });
    try gpu.upload(x.value.data, x_values);
    try gpu.upload(w.value.data, w_values);
    try gpu.fill(x.grad.data, 0);
    try gpu.fill(w.grad.data, 0);
    const input = if (half) try t.cast(x, u16) else x;
    const y = try t.cast(try t.matmul(input, w), f32);
    try gpu.download(out, y.value.data);
    try gpu.upload(y.grad.data, g_values);
    try runBackwardOf(&t);
    try gpu.download(grad_x, x.grad.data);
    try gpu.download(grad_w, w.grad.data);
}

test "a product of bf16 activations and f32 weights matches a double precision reference, outputs and both gradients, on odd sizes" {
    const gpu = try gpu_mod.Gpu.forTest(8 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(51);
    const rng = prng.random();
    var x: [70 * 50]f32 = undefined;
    var w: [50 * 45]f32 = undefined;
    var g: [70 * 45]f32 = undefined;
    for (&x) |*v| v.* = rng.floatNorm(f32);
    for (&w) |*v| v.* = rng.floatNorm(f32) * 0.3;
    for (&g) |*v| v.* = rng.floatNorm(f32);

    // The reference multiplies the values the kernels actually see: x and the gradient rounded to bf16, w rounded when it is packed.
    var want_out: [70 * 45]f64 = @splat(0);
    var want_x: [70 * 50]f64 = @splat(0);
    var want_w: [50 * 45]f64 = @splat(0);
    for (0..70) |i| {
        for (0..45) |j| {
            for (0..50) |k| {
                want_out[i * 45 + j] += @as(f64, rounded(x[i * 50 + k])) * rounded(w[k * 45 + j]);
                want_x[i * 50 + k] += @as(f64, rounded(g[i * 45 + j])) * rounded(w[k * 45 + j]);
                want_w[k * 45 + j] += @as(f64, rounded(x[i * 50 + k])) * rounded(g[i * 45 + j]);
            }
        }
    }

    var out: [70 * 45]f32 = undefined;
    var grad_x: [70 * 50]f32 = undefined;
    var grad_w: [50 * 45]f32 = undefined;
    try runMatmulChain(true, gpu, &x, &w, &g, &out, &grad_x, &grad_w);
    try expectCloseToReference(&want_out, &out, 0.01, 0.02);
    try expectCloseToReference(&want_x, &grad_x, 0.01, 0.05);
    try expectCloseToReference(&want_w, &grad_w, 0.001, 0.01);
}

fn runBatchedChain(comptime half: bool, gpu: *gpu_mod.Gpu, a_values: []const f32, b_values: []const f32, g_values: []const f32, out: []f32, grad_a: []f32, grad_b: []f32) !void {
    const A = Tensor(f32, &.{ 3, 5, 7 });
    const B = Tensor(f32, &.{ 3, 4, 7 });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const a = try t.leaf(A{ .data = try gpu.alloc(f32, A.len) });
    const b = try t.leaf(B{ .data = try gpu.alloc(f32, B.len) });
    try gpu.upload(a.value.data, a_values);
    try gpu.upload(b.value.data, b_values);
    try gpu.fill(a.grad.data, 0);
    try gpu.fill(b.grad.data, 0);
    const lhs = if (half) try t.cast(a, u16) else a;
    const rhs = if (half) try t.cast(b, u16) else b;
    const y = try t.cast(try t.batchedMatmul(lhs, rhs, true), f32);
    try gpu.download(out, y.value.data);
    try gpu.upload(y.grad.data, g_values);
    try runBackwardOf(&t);
    try gpu.download(grad_a, a.grad.data);
    try gpu.download(grad_b, b.grad.data);
}

test "a batched product of bf16 matrices with a transposed right side matches a double precision reference, outputs and both gradients" {
    const gpu = try gpu_mod.Gpu.forTest(8 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(52);
    const rng = prng.random();
    var a: [3 * 5 * 7]f32 = undefined;
    var b: [3 * 4 * 7]f32 = undefined;
    var g: [3 * 5 * 4]f32 = undefined;
    for (&a) |*v| v.* = rng.floatNorm(f32);
    for (&b) |*v| v.* = rng.floatNorm(f32);
    for (&g) |*v| v.* = rng.floatNorm(f32);

    var want_out: [3 * 5 * 4]f64 = @splat(0);
    var want_a: [3 * 5 * 7]f64 = @splat(0);
    var want_b: [3 * 4 * 7]f64 = @splat(0);
    for (0..3) |s| {
        for (0..5) |i| {
            for (0..4) |j| {
                for (0..7) |k| {
                    want_out[(s * 5 + i) * 4 + j] += @as(f64, rounded(a[(s * 5 + i) * 7 + k])) * rounded(b[(s * 4 + j) * 7 + k]);
                    want_a[(s * 5 + i) * 7 + k] += @as(f64, rounded(g[(s * 5 + i) * 4 + j])) * rounded(b[(s * 4 + j) * 7 + k]);
                    want_b[(s * 4 + j) * 7 + k] += @as(f64, rounded(g[(s * 5 + i) * 4 + j])) * rounded(a[(s * 5 + i) * 7 + k]);
                }
            }
        }
    }
    var out: [3 * 5 * 4]f32 = undefined;
    var grad_a: [3 * 5 * 7]f32 = undefined;
    var grad_b: [3 * 4 * 7]f32 = undefined;
    try runBatchedChain(true, gpu, &a, &b, &g, &out, &grad_a, &grad_b);
    try expectCloseToReference(&want_out, &out, 0.01, 0.02);
    try expectCloseToReference(&want_a, &grad_a, 0.01, 0.05);
    try expectCloseToReference(&want_b, &grad_b, 0.01, 0.05);
}

test "an embedding gathers bf16 rows from an f32 table and scatters bf16 gradients back into it" {
    const gpu = try gpu_mod.Gpu.forTest(4 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(53);
    const rng = prng.random();
    var table_values: [10 * 6]f32 = undefined;
    var g: [8 * 6]f32 = undefined;
    for (&table_values) |*v| v.* = rng.floatNorm(f32);
    for (&g) |*v| v.* = rng.floatNorm(f32);
    const host_ids = [_]u32{ 3, 7, 3, 0, 9, 7, 7, 1 };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const table = try t.leaf(Tensor(f32, &.{ 10, 6 }){ .data = try gpu.alloc(f32, 60) });
    try gpu.upload(table.value.data, &table_values);
    try gpu.fill(table.grad.data, 0);
    const ids = try gpu.alloc(u32, host_ids.len);
    try gpu.upload(ids, &host_ids);
    const rows = try t.cast(try t.embedding(table, 8, ids, u16), f32);
    var got: [8 * 6]f32 = undefined;
    try gpu.download(&got, rows.value.data);
    for (host_ids, 0..) |id, i| {
        for (0..6) |j| try testing.expectEqual(rounded(table_values[id * 6 + j]), got[i * 6 + j]);
    }
    try gpu.upload(rows.grad.data, &g);
    try runBackwardOf(&t);
    var want: [10 * 6]f64 = @splat(0);
    for (host_ids, 0..) |id, i| {
        for (0..6) |j| want[id * 6 + j] += rounded(g[i * 6 + j]);
    }
    var grad: [10 * 6]f32 = undefined;
    try gpu.download(&grad, table.grad.data);
    try expectCloseToReference(&want, &grad, 1e-5, 1e-5);
}

test "cross entropy and mean squared error accept bf16 inputs and compute in f32" {
    const gpu = try gpu_mod.Gpu.forTest(4 << 20);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(54);
    const rng = prng.random();
    var logits_values: [8 * 10]f32 = undefined;
    var target_values: [8 * 10]f32 = undefined;
    for (&logits_values) |*v| v.* = rng.floatNorm(f32) * 2;
    for (&target_values) |*v| v.* = rng.floatNorm(f32);
    const host_labels = [_]u8{ 3, 0, 9, 4, 4, 1, 7, 2 };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const labels = try gpu.alloc(u8, host_labels.len);
    try gpu.upload(labels, &host_labels);

    var want_loss: f64 = 0;
    var want_grad: [8 * 10]f64 = undefined;
    for (0..8) |i| {
        var largest: f64 = -std.math.inf(f64);
        for (0..10) |j| largest = @max(largest, rounded(logits_values[i * 10 + j]));
        var total: f64 = 0;
        for (0..10) |j| total += @exp(@as(f64, rounded(logits_values[i * 10 + j])) - largest);
        for (0..10) |j| {
            const p = @exp(@as(f64, rounded(logits_values[i * 10 + j])) - largest) / total;
            want_grad[i * 10 + j] = (p - @as(f64, if (j == host_labels[i]) 1 else 0)) / 8;
        }
        want_loss -= @log(@exp(@as(f64, rounded(logits_values[i * 10 + host_labels[i]])) - largest) / total) / 8;
    }

    {
        try gpu.resetStep();
        var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
        const logits = try t.leaf(Tensor(f32, &.{ 8, 10 }){ .data = try gpu.alloc(f32, 80) });
        try gpu.upload(logits.value.data, &logits_values);
        try gpu.fill(logits.grad.data, 0);
        const loss = try t.crossEntropy(try t.cast(logits, u16), labels);
        try testing.expectApproxEqAbs(@as(f32, @floatCast(want_loss)), try t.scalar(loss), 2e-3);
        try t.backward(loss);
        var grad: [80]f32 = undefined;
        try gpu.download(&grad, logits.grad.data);
        try expectCloseToReference(&want_grad, &grad, 0.02, 2e-3);
    }
    {
        try gpu.resetStep();
        var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
        const prediction = try t.leaf(Tensor(f32, &.{ 8, 10 }){ .data = try gpu.alloc(f32, 80) });
        const target = try t.constant(Tensor(f32, &.{ 8, 10 }){ .data = try gpu.alloc(f32, 80) });
        try gpu.upload(prediction.value.data, &logits_values);
        try gpu.upload(target.value.data, &target_values);
        try gpu.fill(prediction.grad.data, 0);
        const loss = try t.mse(try t.cast(prediction, u16), try t.cast(target, u16));
        var want_mse: f64 = 0;
        var want_mse_grad: [80]f64 = undefined;
        for (0..80) |i| {
            const diff = @as(f64, rounded(logits_values[i])) - rounded(target_values[i]);
            want_mse += diff * diff / 80;
            want_mse_grad[i] = 2 * diff / 80;
        }
        try testing.expectApproxEqAbs(@as(f32, @floatCast(want_mse)), try t.scalar(loss), 2e-3 * @as(f32, @floatCast(want_mse)));
        try t.backward(loss);
        var grad: [80]f32 = undefined;
        try gpu.download(&grad, prediction.grad.data);
        try expectCloseToReference(&want_mse_grad, &grad, 0.02, 1e-3);
    }
}
