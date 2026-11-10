const std = @import("std");
const Allocator = std.mem.Allocator;
const ops = @import("../tensor/ops.zig");
const tensor = @import("../tensor/tensor.zig");
const kernels = @import("../cpu/kernels.zig");
const gpu_mod = @import("../gpu/gpu.zig");
const conv = @import("../gpu/math/conv.zig");
const drop = @import("../gpu/math/dropout.zig");
const norm = @import("../gpu/math/norm.zig");
const Tensor = tensor.Tensor;

pub const Error = gpu_mod.Error;

fn padTo4(comptime dims: []const usize, comptime fill: u32) [4]u32 {
    if (dims.len > 4) @compileError("gpu kernels support tensors of rank 4 or less");
    var out: [4]u32 = @splat(fill);
    for (dims, 0..) |d, i| out[4 - dims.len + i] = @intCast(d);
    return out;
}

fn rowMajorStrides4(comptime dims: []const usize) [4]u32 {
    var out: [4]u32 = @splat(0);
    var acc: u32 = 1;
    var i: usize = dims.len;
    while (i > 0) {
        i -= 1;
        out[4 - dims.len + i] = acc;
        acc *= @intCast(dims[i]);
    }
    return out;
}

fn reducedAxisMask(comptime target: []const usize) u32 {
    var mask: u32 = 0;
    for (padTo4(target, 1), 0..) |d, axis| {
        if (d == 1) mask |= @as(u32, 1) << @intCast(axis);
    }
    return mask;
}

pub fn Var(comptime Tn: type) type {
    return struct { value: Tn, grad: Tn, needs_grad: bool };
}

pub fn reshape(x: anytype, comptime shape: []const usize) Var(Tensor(@TypeOf(x.value).Elem, shape)) {
    const R = Tensor(@TypeOf(x.value).Elem, shape);
    comptime std.debug.assert(R.len == @TypeOf(x.value).len);
    return .{ .value = .{ .data = x.value.data }, .grad = .{ .data = x.grad.data }, .needs_grad = x.needs_grad };
}

pub const ConvOpts = struct { stride: usize = 1, pad: usize = 0 };

fn kernelSize(comptime In: type, comptime W: type) usize {
    if (In.rank != 4 or W.rank != 2) @compileError("conv2d expects (n, h, w, c) input and (k*k*c, out) weights");
    const k = std.math.sqrt(W.shape[0] / In.shape[3]);
    if (k * k * In.shape[3] != W.shape[0]) @compileError("conv2d weights must have k*k*c rows");
    return k;
}

fn ConvOut(comptime In: type, comptime W: type, comptime opts: ConvOpts) type {
    const k = kernelSize(In, W);
    return Tensor(f32, &.{ In.shape[0], (In.shape[1] + 2 * opts.pad - k) / opts.stride + 1, (In.shape[2] + 2 * opts.pad - k) / opts.stride + 1, W.shape[1] });
}

fn PoolOut(comptime In: type) type {
    if (In.rank != 4) @compileError("maxPool expects (n, h, w, c) input");
    return Tensor(f32, &.{ In.shape[0], In.shape[1] / 2, In.shape[2] / 2, In.shape[3] });
}

const Node = struct {
    ctx: *anyopaque,
    run: *const fn (*Tape, *anyopaque) Error!void,
};

pub const Tape = struct {
    gpa: Allocator,
    par: kernels.Parallel = .{},
    nodes: std.ArrayList(Node) = .empty,
    training: bool = false,
    seed: u64 = 0,

    fn newTensor(self: *Tape, comptime Tn: type) Error!Tn {
        if (self.par.gpu) |dev| return .{ .data = try dev.scratch(Tn.Elem, Tn.len) };
        return Tn.empty(self.gpa);
    }

    pub fn filledTensor(self: *Tape, comptime Tn: type, value: f32) Error!Tn {
        const filled = try self.newTensor(Tn);
        if (self.par.gpu) |dev| try dev.fill(filled.data, value) else @memset(filled.data, value);
        return filled;
    }

    pub fn zeroedTensor(self: *Tape, comptime Tn: type) Error!Tn {
        if (self.par.gpu) |dev| return .{ .data = try dev.scratchZeroed(Tn.Elem, Tn.len) };
        const tensor_zeroed = try Tn.empty(self.gpa);
        @memset(tensor_zeroed.data, 0);
        return tensor_zeroed;
    }

    fn makeVar(self: *Tape, value: anytype, needs_grad: bool) Error!Var(@TypeOf(value)) {
        const Tn = @TypeOf(value);
        var grad: Tn = undefined;
        if (needs_grad) {
            if (self.par.gpu) |dev| {
                grad = .{ .data = try dev.scratchZeroed(Tn.Elem, Tn.len) };
            } else {
                grad = try self.newTensor(Tn);
                @memset(grad.data, 0);
            }
        }
        return .{ .value = value, .grad = grad, .needs_grad = needs_grad };
    }

    pub fn scalar(self: *Tape, v: anytype) Error!@TypeOf(v.value).Elem {
        if (self.par.gpu) |dev| return dev.scalar(v.value.data);
        return v.value.data[0];
    }

    fn elementwise(self: *Tape, comptime op: gpu_mod.BinaryOp, a: anytype, b: anytype) Error!ops.BroadcastResult(@TypeOf(a), @TypeOf(b)) {
        if (self.par.gpu) |dev| {
            const out = try self.newTensor(ops.BroadcastResult(@TypeOf(a), @TypeOf(b)));
            const bc = comptime ops.broadcast(@TypeOf(a).shape, @TypeOf(b).shape);
            try dev.elementwise(op, a.data, b.data, out.data, .{ .dims = padTo4(&bc.dims, 1), .a_strides = padTo4(&bc.a, 0), .b_strides = padTo4(&bc.b, 0) });
            return out;
        }
        return switch (op) {
            .add => ops.add(self.gpa, a, b),
            .mul => ops.mul(self.gpa, a, b),
        };
    }

    pub fn leaf(self: *Tape, value: anytype) Error!Var(@TypeOf(value)) {
        return self.makeVar(value, true);
    }

    pub fn constant(self: *Tape, value: anytype) Error!Var(@TypeOf(value)) {
        return self.makeVar(value, false);
    }

    fn recordBackward(self: *Tape, ctx: anytype, comptime run: fn (*Tape, *const @TypeOf(ctx)) Error!void) Error!void {
        const Ctx = @TypeOf(ctx);
        const p = try self.gpa.create(Ctx);
        p.* = ctx;
        try self.nodes.append(self.gpa, .{
            .ctx = p,
            .run = struct {
                fn call(tape: *Tape, raw: *anyopaque) Error!void {
                    return run(tape, @ptrCast(@alignCast(raw)));
                }
            }.call,
        });
    }

    fn accumulateGrad(self: *Tape, v: anytype, g: anytype) Error!void {
        if (!v.needs_grad) return;
        const D = @TypeOf(v.grad);
        const G = @TypeOf(g);
        if (self.par.gpu) |dev| {
            if (comptime std.mem.eql(usize, D.shape, G.shape)) {
                try dev.addInPlace(v.grad.data, g.data);
            } else {
                try dev.sumTo(g.data, v.grad.data, .{ .dims = padTo4(G.shape, 1), .strides = rowMajorStrides4(G.shape), .mask = comptime reducedAxisMask(D.shape) }, true);
            }
            return;
        }
        if (comptime std.mem.eql(usize, D.shape, G.shape)) {
            for (v.grad.data, g.data) |*d, x| d.* += x;
        } else {
            const r = try ops.sumTo(self.gpa, g, D.shape);
            for (v.grad.data, r.data) |*d, x| d.* += x;
        }
    }

    pub fn add(self: *Tape, a: anytype, b: anytype) Error!Var(ops.BroadcastResult(@TypeOf(a.value), @TypeOf(b.value))) {
        const out = try self.makeVar(try self.elementwise(.add, a.value, b.value), a.needs_grad or b.needs_grad);
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                try t.accumulateGrad(c.a, c.out.grad);
                try t.accumulateGrad(c.b, c.out.grad);
            }
        }.run);
        return out;
    }

    pub fn mul(self: *Tape, a: anytype, b: anytype) Error!Var(ops.BroadcastResult(@TypeOf(a.value), @TypeOf(b.value))) {
        const out = try self.makeVar(try self.elementwise(.mul, a.value, b.value), a.needs_grad or b.needs_grad);
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (c.a.needs_grad) try t.accumulateGrad(c.a, try t.elementwise(.mul, c.out.grad, c.b.value));
                if (c.b.needs_grad) try t.accumulateGrad(c.b, try t.elementwise(.mul, c.out.grad, c.a.value));
            }
        }.run);
        return out;
    }

    pub fn matmul(self: *Tape, a: anytype, b: anytype) Error!Var(tensor.MatmulResult(@TypeOf(a.value), @TypeOf(b.value), .{})) {
        const value = try self.newTensor(tensor.MatmulResult(@TypeOf(a.value), @TypeOf(b.value), .{}));
        try tensor.matmulInto(self.gpa, self.par, a.value, b.value, value, .{});
        const out = try self.makeVar(value, a.needs_grad or b.needs_grad);
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (c.a.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.out.grad, c.b.value, c.a.grad, .{ .trans_b = true, .accumulate = true });
                if (c.b.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.a.value, c.out.grad, c.b.grad, .{ .trans_a = true, .accumulate = true });
            }
        }.run);
        return out;
    }

    pub fn relu(self: *Tape, x: anytype) Error!Var(@TypeOf(x.value)) {
        var value: @TypeOf(x.value) = undefined;
        if (self.par.gpu) |dev| {
            value = try self.newTensor(@TypeOf(x.value));
            try dev.relu(x.value.data, value.data);
        } else {
            value = try ops.relu(self.gpa, x.value);
        }
        const out = try self.makeVar(value, x.needs_grad);
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.reluBackward(c.x.value.data, c.out.grad.data, c.x.grad.data);
                for (c.x.grad.data, c.x.value.data, c.out.grad.data) |*gx, v, g| {
                    gx.* += if (v > 0) g else 0;
                }
            }
        }.run);
        return out;
    }

    fn dropKernel(self: *Tape, src: []const f32, dst: []f32, seed: u64, keep: f32, accumulate: bool) Error!void {
        if (self.par.gpu) |dev| return dev.dropout(src, dst, seed, keep, accumulate);
        kernels.parallelEach(self.par, src.len, drop.apply, .{ src, dst, seed, keep, accumulate });
    }
    pub fn dropout(self: *Tape, x: anytype, keep: f32) Error!Var(@TypeOf(x.value)) {
        if (!self.training) return x;
        const seed = self.seed;
        self.seed += 1;
        const value = try self.newTensor(@TypeOf(x.value));
        try self.dropKernel(x.value.data, value.data, seed, keep, false);
        const out = try self.makeVar(value, x.needs_grad);
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out), seed: u64, keep: f32 };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out, .seed = seed, .keep = keep }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                try t.dropKernel(c.out.grad.data, c.x.grad.data, c.seed, c.keep, true);
            }
        }.run);
        return out;
    }

    pub fn batchNorm(self: *Tape, x: anytype, gamma: anytype, beta: anytype, running: anytype) Error!Var(@TypeOf(x.value)) {
        const In = @TypeOf(x.value);
        const c = In.shape[In.rank - 1];
        const Channels = Tensor(f32, &.{c});
        const rows = In.len / c;
        const mean = try self.newTensor(Channels);
        const inv_std = try self.newTensor(Channels);
        const value = try self.newTensor(In);
        if (self.par.gpu) |dev| {
            try dev.batchNormStats(x.value.data, mean.data, inv_std.data, running.mean.data, running.variance.data, self.training);
            try dev.batchNormApply(x.value.data, value.data, mean.data, inv_std.data, gamma.value.data, beta.value.data);
        } else {
            norm.stats(x.value.data, mean.data, inv_std.data, running.mean.data, running.variance.data, self.training);
            kernels.parallelEach(self.par, In.len, norm.apply, .{ x.value.data, value.data, c, mean.data, inv_std.data, gamma.value.data, beta.value.data });
        }
        const out = try self.makeVar(value, x.needs_grad or gamma.needs_grad or beta.needs_grad);

        const Ctx = struct { x: @TypeOf(x), gamma: @TypeOf(gamma), beta: @TypeOf(beta), mean: Channels, inv_std: Channels, out: @TypeOf(out) };
        if (out.needs_grad and self.training) try self.recordBackward(Ctx{ .x = x, .gamma = gamma, .beta = beta, .mean = mean, .inv_std = inv_std, .out = out }, struct {
            fn run(t: *Tape, k: *const Ctx) Error!void {
                const sums = try t.zeroedTensor(Tensor(f32, &.{2 * c}));
                if (t.par.gpu) |dev| {
                    try dev.batchNormGradSums(k.x.value.data, k.out.grad.data, k.mean.data, k.inv_std.data, sums.data, k.gamma.grad.data, k.beta.grad.data);
                    if (k.x.needs_grad) try dev.batchNormGradInput(k.x.value.data, k.out.grad.data, k.x.grad.data, k.mean.data, k.inv_std.data, k.gamma.value.data, sums.data);
                } else {
                    norm.gradSums(k.x.value.data, k.out.grad.data, k.mean.data, k.inv_std.data, sums.data, k.gamma.grad.data, k.beta.grad.data);
                    if (k.x.needs_grad) kernels.parallelEach(t.par, In.len, norm.gradInput, .{ k.x.value.data, k.out.grad.data, k.x.grad.data, c, rows, k.mean.data, k.inv_std.data, k.gamma.value.data, sums.data });
                }
            }
        }.run);
        return out;
    }

    pub fn avgPool(self: *Tape, x: anytype) Error!Var(PoolOut(@TypeOf(x.value))) {
        const In = @TypeOf(x.value);
        const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = In.shape[3], .k = 2 };
        const value = try self.newTensor(PoolOut(In));
        try self.convKernel(.avgPool, .{ x.value.data, value.data }, d);
        const out = try self.makeVar(value, x.needs_grad);
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, k: *const Ctx) Error!void {
                try t.convKernel(.avgPoolBackward, .{ k.out.grad.data, k.x.grad.data }, d);
            }
        }.run);
        return out;
    }

    fn convKernel(self: *Tape, comptime op: conv.Op, slices: anytype, d: conv.Dims) Error!void {
        if (self.par.gpu) |dev| return dev.convKernel(op, slices, d);
        kernels.parallelEach(self.par, conv.workItems(op, d), @field(conv, @tagName(op)), slices ++ .{d});
    }
    pub fn conv2d(self: *Tape, x: anytype, w: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts)) {
        const In = @TypeOf(x.value);
        const Out = ConvOut(In, @TypeOf(w.value), opts);
        const k = comptime kernelSize(In, @TypeOf(w.value));
        const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = In.shape[3], .k = k, .stride = opts.stride, .pad = opts.pad };
        const rows = Out.len / Out.shape[3];
        const Col = Tensor(f32, &.{ rows, k * k * In.shape[3] });
        const Flat = Tensor(f32, &.{ rows, Out.shape[3] });

        const col = try self.newTensor(Col);
        try self.convKernel(.im2col, .{ x.value.data, col.data }, d);
        const value = try self.newTensor(Out);
        try tensor.matmulInto(self.gpa, self.par, col, w.value, Flat{ .data = value.data }, .{});
        const out = try self.makeVar(value, x.needs_grad or w.needs_grad);

        const Ctx = struct { x: @TypeOf(x), w: @TypeOf(w), col: Col, out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .w = w, .col = col, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                const grad: Flat = .{ .data = c.out.grad.data };
                if (c.x.needs_grad) {
                    const dcol = try t.newTensor(Col);
                    try tensor.matmulInto(t.gpa, t.par, grad, c.w.value, dcol, .{ .trans_b = true });
                    try t.convKernel(.col2im, .{ dcol.data, c.x.grad.data }, d);
                }
                if (c.w.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.col, grad, c.w.grad, .{ .trans_a = true, .accumulate = true });
            }
        }.run);
        return out;
    }

    pub fn maxPool(self: *Tape, x: anytype) Error!Var(PoolOut(@TypeOf(x.value))) {
        const In = @TypeOf(x.value);
        const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = In.shape[3], .k = 2 };
        const value = try self.newTensor(PoolOut(In));
        try self.convKernel(.maxPool, .{ x.value.data, value.data }, d);
        const out = try self.makeVar(value, x.needs_grad);
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                try t.convKernel(.maxPoolBackward, .{ c.x.value.data, c.out.grad.data, c.x.grad.data }, d);
            }
        }.run);
        return out;
    }

    pub fn crossEntropy(self: *Tape, logits: anytype, labels: []const u8) Error!Var(Tensor(@TypeOf(logits.value).Elem, &.{})) {
        const L = @TypeOf(logits.value);
        comptime {
            if (L.rank != 2) @compileError("crossEntropy expects (batch, classes) logits");
        }
        const rows = L.shape[0];
        const cols = L.shape[1];
        std.debug.assert(labels.len == rows);

        const Scalar = Tensor(L.Elem, &.{});
        var logp: L = undefined;
        var value: Scalar = undefined;
        if (self.par.gpu) |dev| {
            logp = try self.newTensor(L);
            value = try self.newTensor(Scalar);
            try dev.crossEntropy(logits.value.data, logp.data, labels, value.data, cols);
        } else {
            logp = try ops.logSoftmax(self.gpa, logits.value);
            var total: f64 = 0;
            for (labels, 0..) |y, i| total -= logp.data[i * cols + y];
            value = try Scalar.empty(self.gpa);
            value.data[0] = @floatCast(total / rows);
        }
        const out = try self.makeVar(value, logits.needs_grad);

        const Ctx = struct { logits: @TypeOf(logits), logp: L, labels: []const u8, out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .logits = logits, .logp = logp, .labels = labels, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.crossEntropyBackward(c.logp.data, c.labels, c.logits.grad.data, c.out.grad.data, cols);
                const scale = c.out.grad.data[0] / rows;
                for (0..rows) |i| {
                    for (0..cols) |j| {
                        var p = @exp(c.logp.data[i * cols + j]);
                        if (j == c.labels[i]) p -= 1;
                        c.logits.grad.data[i * cols + j] += p * scale;
                    }
                }
            }
        }.run);
        return out;
    }

    pub fn backward(self: *Tape, loss: anytype) Error!void {
        comptime {
            if (@TypeOf(loss.value).rank != 0) @compileError("backward needs a scalar loss");
        }
        if (self.par.gpu) |dev| try dev.fill(loss.grad.data, 1) else loss.grad.data[0] = 1;
        var i = self.nodes.items.len;
        while (i > 0) {
            i -= 1;
            const node = self.nodes.items[i];
            try node.run(self, node.ctx);
        }
    }
};

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

test "gpu dropout draws the same mask as the cpu" {
    const gpu = gpu_mod.Gpu.init(testing.allocator, 1 << 20) catch |err| switch (err) {
        error.DriverNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer gpu.deinit();
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
    var gpu_tape: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true, .seed = 3 };
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
        const flat = reshape(try t.maxPool(act), &.{ 2, 12 });
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

fn ConvCase(comptime opts: ConvOpts) type {
    return struct {
        const In = Tensor(f32, &.{ 2, 5, 5, 2 });
        const Wt = Tensor(f32, &.{ 18, 3 });
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
                        for (0..3) |oc| {
                            var want: f32 = 0;
                            for (0..3) |ky| {
                                for (0..3) |kx| {
                                    const py = oy * opts.stride + ky;
                                    const px = ox * opts.stride + kx;
                                    if (py < opts.pad or px < opts.pad or py - opts.pad >= 5 or px - opts.pad >= 5) continue;
                                    for (0..2) |c| {
                                        want += p.x.data[((n * 5 + py - opts.pad) * 5 + px - opts.pad) * 2 + c] * p.w.data[((ky * 3 + kx) * 2 + c) * 3 + oc];
                                    }
                                }
                            }
                            try testing.expectApproxEqAbs(want, run.y.value.data[((n * oh + oy) * ow + ox) * 3 + oc], 1e-5);
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

test "gpu and cpu agree on a small network" {
    try gradcheck.expectGpuMatchesCpu(SmallNet, 21, 1e-5);
}

test "gradients of a small network match finite differences" {
    try gradcheck.expectFiniteDifferences(SmallNet, 7, 5e-3, 3e-3);
}

test "gradients of a convolution, pooling and linear network match finite differences and the gpu" {
    try gradcheck.expectFiniteDifferences(ConvPoolNet, 5, 1e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(ConvPoolNet, 8, 1e-5);
}

test "conv2d with padding and stride matches a direct convolution, finite differences and the gpu" {
    inline for (.{ ConvOpts{}, ConvOpts{ .stride = 1, .pad = 1 }, ConvOpts{ .stride = 2, .pad = 1 }, ConvOpts{ .stride = 2, .pad = 0 } }) |opts| {
        try ConvCase(opts).expectDirectConvolution();
        try gradcheck.expectFiniteDifferences(ConvCase(opts), 31, 1e-3, 3e-3);
        try gradcheck.expectGpuMatchesCpu(ConvCase(opts), 32, 1e-5);
    }
}

test "batch norm gradients match finite differences, and the gpu matches the cpu including running statistics" {
    try gradcheck.expectFiniteDifferences(BnCase, 41, 1e-3, 3e-3);
    try gradcheck.expectGpuMatchesCpu(BnCase, 43, 1e-5);
}
test "average pooling takes window means and spreads gradients evenly, also on the gpu" {
    const gpu = gpu_mod.Gpu.init(testing.allocator, 1 << 20) catch |err| switch (err) {
        error.DriverNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer gpu.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const Img = Tensor(f32, &.{ 1, 2, 4, 2 });
    const values = [_]f32{ 1, 10, 2, 20, 5, 0, 6, 0, 3, 30, 4, 40, 7, 0, 8, 0 };

    var t: Tape = .{ .gpa = arena.allocator() };
    const x = try t.leaf(try Img.fromSlice(t.gpa, &values));
    const y = try t.avgPool(x);
    try testing.expectEqualSlices(f32, &.{ 2.5, 25, 6.5, 0 }, y.value.data);
    @memset(y.grad.data, 4);
    try t.nodes.items[0].run(&t, t.nodes.items[0].ctx);
    for (x.grad.data) |g| try testing.expectEqual(@as(f32, 1), g);

    const dx = try gpu.alloc(f32, Img.len);
    const dev_x = try gpu.alloc(f32, Img.len);
    try gpu.upload(dev_x, &values);
    try gpu.fill(dx, 0);
    var gpu_tape: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const gy = try gpu_tape.avgPool(Var(Img){ .value = .{ .data = dev_x }, .grad = .{ .data = dx }, .needs_grad = true });
    var back: [4]f32 = undefined;
    try gpu.download(&back, gy.value.data);
    try testing.expectEqualSlices(f32, y.value.data, &back);
    try gpu.fill(gy.grad.data, 4);
    try gpu_tape.nodes.items[0].run(&gpu_tape, gpu_tape.nodes.items[0].ctx);
    var grads: [16]f32 = undefined;
    try gpu.download(&grads, dx);
    for (grads) |g| try testing.expectEqual(@as(f32, 1), g);
}
