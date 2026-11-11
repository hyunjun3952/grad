const std = @import("std");
const Allocator = std.mem.Allocator;
const ops = @import("../tensor/ops.zig");
const tensor = @import("../tensor/tensor.zig");
const kernels = @import("../cpu/kernels.zig");
const gpu_mod = @import("../gpu/gpu.zig");
const ga = @import("../gpu/args.zig");
const conv = @import("../gpu/math/conv.zig");
const cpu = @import("../cpu/ops.zig");
const activation_math = @import("../gpu/math/activation.zig");
const softmax_math = @import("../gpu/math/softmax.zig");
const layernorm_math = @import("../gpu/math/layernorm.zig");
const drop = @import("../gpu/math/dropout.zig");
const norm = @import("../gpu/math/norm.zig");
const bf16_math = @import("../gpu/math/bf16.zig");
const Tensor = tensor.Tensor;
const trace_mod = @import("trace.zig");
pub const Trace = trace_mod.Trace;

pub const Error = gpu_mod.Error;

fn padToRank(comptime dims: []const usize, comptime fill: u32) [ga.max_rank]u32 {
    if (dims.len > ga.max_rank) @compileError("gpu kernels support tensors of rank 6 or less");
    var out: [ga.max_rank]u32 = @splat(fill);
    for (dims, 0..) |d, i| out[ga.max_rank - dims.len + i] = @intCast(d);
    return out;
}

fn rowMajorStrides(comptime dims: []const usize) [ga.max_rank]u32 {
    var out: [ga.max_rank]u32 = @splat(0);
    var acc: u32 = 1;
    var i: usize = dims.len;
    while (i > 0) {
        i -= 1;
        out[ga.max_rank - dims.len + i] = acc;
        acc *= @intCast(dims[i]);
    }
    return out;
}

fn reducedAxisMask(comptime target: []const usize) u32 {
    var mask: u32 = 0;
    for (padToRank(target, 1), 0..) |d, axis| {
        if (d == 1) mask |= @as(u32, 1) << @intCast(axis);
    }
    return mask;
}

/// A value on the tape: its tensor, the gradient buffer of the same shape, and whether a gradient is needed.
/// Gradients are accumulated with +=, so the buffer starts at zero.
pub fn Var(comptime Tn: type) type {
    return struct { value: Tn, grad: Tn, needs_grad: bool };
}

/// Views a variable with another shape of the same length. The data and gradient are shared, nothing is copied.
pub fn reshape(x: anytype, comptime shape: []const usize) Var(Tensor(@TypeOf(x.value).Elem, shape)) {
    const R = Tensor(@TypeOf(x.value).Elem, shape);
    comptime std.debug.assert(R.len == @TypeOf(x.value).len);
    return .{ .value = .{ .data = x.value.data }, .grad = .{ .data = x.grad.data }, .needs_grad = x.needs_grad };
}

fn batchNormGradIo(comptime E: type, k: anytype, sums: []const f32, comptime fuse_relu: bool) norm.GradIo(E) {
    return .{ .x = k.x.value.data.ptr, .dy = k.out.grad.data.ptr, .dx = k.x.grad.data.ptr, .sums = sums.ptr, .mask = if (fuse_relu) k.out.value.data.ptr else k.out.grad.data.ptr };
}

pub const Activation = activation_math.Kind;
pub const ConvOpts = struct { stride: usize = 1, pad: usize = 0, dilation: usize = 1, groups: usize = 1 };
pub const PoolOpts = struct { kernel: usize = 2, stride: usize = 2, pad: usize = 0 };
const ConvPath = enum { direct, packed_im2col, implicit, tf32, cudnn, grouped };
const Packed = gpu_mod.Gpu.Packed;

fn sliceOf(comptime Tn: type, data: anytype, index: usize) Tn {
    return .{ .data = data[index * Tn.len ..][0..Tn.len] };
}

fn kernelSize(comptime In: type, comptime W: type, comptime groups: usize) usize {
    if (In.rank != 4 or W.rank != 2) @compileError("conv2d expects (n, h, w, c) input and (k*k*c/groups, out) weights");
    if (In.shape[3] % groups != 0 or W.shape[1] % groups != 0) @compileError("conv2d groups must divide the input and output channels");
    const per_group = In.shape[3] / groups;
    const k = std.math.sqrt(W.shape[0] / per_group);
    if (k * k * per_group != W.shape[0]) @compileError("conv2d weights must have k*k*c/groups rows");
    return k;
}

fn ConvOut(comptime In: type, comptime W: type, comptime opts: ConvOpts) type {
    const span = opts.dilation * (kernelSize(In, W, opts.groups) - 1) + 1;
    return Tensor(In.Elem, &.{ In.shape[0], (In.shape[1] + 2 * opts.pad - span) / opts.stride + 1, (In.shape[2] + 2 * opts.pad - span) / opts.stride + 1, W.shape[1] });
}

fn PoolOut(comptime In: type, comptime opts: PoolOpts) type {
    if (In.rank != 4) @compileError("pooling expects (n, h, w, c) input");
    return Tensor(In.Elem, &.{ In.shape[0], (In.shape[1] + 2 * opts.pad - opts.kernel) / opts.stride + 1, (In.shape[2] + 2 * opts.pad - opts.kernel) / opts.stride + 1, In.shape[3] });
}

// ***** tape *****

const Node = struct {
    ctx: *anyopaque,
    run: *const fn (*Tape, *anyopaque) Error!void,
};

/// Records the operations of one forward pass so that backward can replay them in reverse.
/// Every op has a CPU branch and a GPU branch (when par.gpu is set); results live in the GPU scratch arena or in gpa.
pub const Tape = struct {
    gpa: Allocator,
    par: kernels.Parallel = .{},
    nodes: std.ArrayList(Node) = .empty,
    training: bool = false,
    smoothing: f32 = 0,
    seed: u64 = 0,
    trace: ?*Trace = null,

    fn note(self: *Tape, op: trace_mod.Op, inputs: anytype, output: anytype, attrs: trace_mod.Attrs) Error!void {
        if (self.trace) |t| try t.add(op, inputs, output, attrs);
    }

    fn newTensor(self: *Tape, comptime Tn: type) Error!Tn {
        if (self.par.gpu) |dev| return .{ .data = try dev.scratch(Tn.Elem, Tn.len) };
        return Tn.empty(self.gpa);
    }

    /// Returns a new tensor with every element set to value; the exporter records it as a constant.
    pub fn filledTensor(self: *Tape, comptime Tn: type, value: f32) Error!Tn {
        const filled = try self.newTensor(Tn);
        if (self.par.gpu) |dev| try dev.fill(filled.data, value) else @memset(filled.data, value);
        if (self.trace) |t| try t.addConstant(filled, value);
        return filled;
    }

    /// Returns a new tensor of zeros that the tape does not own on the GPU (it comes from the zero arena).
    pub fn zeroedTensor(self: *Tape, comptime Tn: type) Error!Tn {
        const zeroed: Tn = if (self.par.gpu) |dev| .{ .data = try dev.scratchZeroed(Tn.Elem, Tn.len) } else .{ .data = try self.gpa.alloc(Tn.Elem, Tn.len) };
        if (self.par.gpu == null) @memset(zeroed.data, 0);
        if (self.trace) |t| try t.addConstant(zeroed, 0);
        return zeroed;
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

    /// Reads a one-element variable back to the host, synchronizing the device if needed.
    pub fn scalar(self: *Tape, v: anytype) Error!@TypeOf(v.value).Elem {
        if (self.par.gpu) |dev| return dev.scalar(v.value.data);
        return v.value.data[0];
    }

    fn elementwise(self: *Tape, comptime op: gpu_mod.BinaryOp, a: anytype, b: anytype) Error!ops.BroadcastResult(@TypeOf(a), @TypeOf(b)) {
        if (self.par.gpu) |dev| {
            const out = try self.newTensor(ops.BroadcastResult(@TypeOf(a), @TypeOf(b)));
            const bc = comptime ops.broadcast(@TypeOf(a).shape, @TypeOf(b).shape);
            try dev.elementwise(op, a.data, b.data, out.data, .{ .dims = padToRank(&bc.dims, 1), .a_strides = padToRank(&bc.a, 0), .b_strides = padToRank(&bc.b, 0) });
            return out;
        }
        if (comptime @TypeOf(a).Elem != f32) return error.NoDevice;
        return switch (op) {
            .add => ops.add(self.gpa, a, b),
            .mul => ops.mul(self.gpa, a, b),
        };
    }

    /// Wraps a tensor as a trainable input whose gradient will be computed.
    pub fn leaf(self: *Tape, value: anytype) Error!Var(@TypeOf(value)) {
        return self.makeVar(value, true);
    }

    /// Wraps a tensor as an input that receives no gradient.
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
                try dev.sumTo(g.data, v.grad.data, .{ .dims = padToRank(G.shape, 1), .strides = rowMajorStrides(G.shape), .mask = comptime reducedAxisMask(D.shape) }, true);
            }
            return;
        }
        if (comptime D.Elem != f32) return error.NoDevice;
        if (comptime std.mem.eql(usize, D.shape, G.shape)) {
            for (v.grad.data, g.data) |*d, x| d.* += x;
        } else {
            const r = try ops.sumTo(self.gpa, g, D.shape);
            for (v.grad.data, r.data) |*d, x| d.* += x;
        }
    }

    /// Adds two tensors with broadcasting; the gradient of a broadcast operand is summed back to its shape.
    pub fn add(self: *Tape, a: anytype, b: anytype) Error!Var(ops.BroadcastResult(@TypeOf(a.value), @TypeOf(b.value))) {
        const out = try self.makeVar(try self.elementwise(.add, a.value, b.value), a.needs_grad or b.needs_grad);
        try self.note(.add, .{ a, b }, out, .{});
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                try t.accumulateGrad(c.a, c.out.grad);
                try t.accumulateGrad(c.b, c.out.grad);
            }
        }.run);
        return out;
    }

    /// Computes max(a + b, 0) in one pass, as used by residual connections. Both operands must have the same shape.
    pub fn addRelu(self: *Tape, a: anytype, b: anytype) Error!Var(@TypeOf(a.value)) {
        comptime if (@TypeOf(a.value) != @TypeOf(b.value)) @compileError("addRelu expects two tensors of the same shape");
        const T = @TypeOf(a.value);
        const value = try self.newTensor(T);
        if (self.par.gpu) |dev| {
            try dev.addRelu(a.value.data, b.value.data, value.data);
        } else if (T.Elem == f32) {
            for (value.data, a.value.data, b.value.data) |*o, x, y| o.* = @max(x + y, 0);
        } else return error.NoDevice;
        const out = try self.makeVar(value, a.needs_grad or b.needs_grad);
        try self.note(.add_relu, .{ a, b }, out, .{});
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.addReluBackward(c.out.value.data, c.out.grad.data, c.a.grad.data, c.b.grad.data, c.a.needs_grad, c.b.needs_grad);
                if (comptime T.Elem != f32) return error.NoDevice;
                for (c.out.value.data, c.out.grad.data, 0..) |o, g, i| {
                    const v = if (o > 0) g else 0;
                    if (c.a.needs_grad) c.a.grad.data[i] += v;
                    if (c.b.needs_grad) c.b.grad.data[i] += v;
                }
            }
        }.run);
        return out;
    }

    /// Multiplies two tensors element by element with broadcasting.
    pub fn mul(self: *Tape, a: anytype, b: anytype) Error!Var(ops.BroadcastResult(@TypeOf(a.value), @TypeOf(b.value))) {
        const out = try self.makeVar(try self.elementwise(.mul, a.value, b.value), a.needs_grad or b.needs_grad);
        try self.note(.mul, .{ a, b }, out, .{});
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (c.a.needs_grad) try t.accumulateGrad(c.a, try t.elementwise(.mul, c.out.grad, c.b.value));
                if (c.b.needs_grad) try t.accumulateGrad(c.b, try t.elementwise(.mul, c.out.grad, c.a.value));
            }
        }.run);
        return out;
    }

    /// Multiplies matrices (m, k) x (k, n); the backward pass uses the transposed products without materializing them.
    pub fn matmul(self: *Tape, a: anytype, b: anytype) Error!Var(tensor.MatmulResult(@TypeOf(a.value), @TypeOf(b.value), .{})) {
        const value = try self.newTensor(tensor.MatmulResult(@TypeOf(a.value), @TypeOf(b.value), .{}));
        try tensor.matmulInto(self.gpa, self.par, a.value, b.value, value, .{});
        const out = try self.makeVar(value, a.needs_grad or b.needs_grad);
        try self.note(.matmul, .{ a, b }, out, .{});
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (c.a.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.out.grad, c.b.value, c.a.grad, .{ .trans_b = true, .accumulate = true });
                if (c.b.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.a.value, c.out.grad, c.b.grad, .{ .trans_a = true, .accumulate = true });
            }
        }.run);
        return out;
    }

    /// Applies max(x, 0) element by element.
    pub fn relu(self: *Tape, x: anytype) Error!Var(@TypeOf(x.value)) {
        var value: @TypeOf(x.value) = undefined;
        if (self.par.gpu) |dev| {
            value = try self.newTensor(@TypeOf(x.value));
            try dev.relu(x.value.data, value.data);
        } else {
            if (comptime @TypeOf(x.value).Elem != f32) return error.NoDevice;
            value = try ops.relu(self.gpa, x.value);
        }
        const out = try self.makeVar(value, x.needs_grad);
        try self.note(.relu, .{x}, out, .{});
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

    /// Multiplies (batch, m, k) by (batch, k, n), or by (batch, n, k) when trans_b is set (attention scores).
    pub fn batchedMatmul(self: *Tape, a: anytype, b: anytype, comptime trans_b: bool) Error!Var(Tensor(f32, &.{ @TypeOf(a.value).shape[0], @TypeOf(a.value).shape[1], if (trans_b) @TypeOf(b.value).shape[1] else @TypeOf(b.value).shape[2] })) {
        const A = @TypeOf(a.value);
        const B = @TypeOf(b.value);
        comptime if (A.rank != 3 or B.rank != 3 or A.shape[0] != B.shape[0] or A.shape[2] != (if (trans_b) B.shape[2] else B.shape[1])) @compileError("batchedMatmul expects (batch, m, k) and (batch, k, n) tensors, or (batch, n, k) with trans_b");
        const batch = A.shape[0];
        const m = A.shape[1];
        const k = A.shape[2];
        const n = if (trans_b) B.shape[1] else B.shape[2];
        const Am = Tensor(f32, &.{ m, k });
        const Bm = Tensor(f32, if (trans_b) &.{ n, k } else &.{ k, n });
        const Cm = Tensor(f32, &.{ m, n });
        const C = Tensor(f32, &.{ batch, m, n });
        const value = try self.newTensor(C);
        for (0..batch) |i| try tensor.matmulInto(self.gpa, self.par, sliceOf(Am, a.value.data, i), sliceOf(Bm, b.value.data, i), sliceOf(Cm, value.data, i), .{ .trans_b = trans_b });
        const out = try self.makeVar(value, a.needs_grad or b.needs_grad);
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                for (0..batch) |i| {
                    const dc = sliceOf(Cm, c.out.grad.data, i);
                    if (c.a.needs_grad) try tensor.matmulInto(t.gpa, t.par, dc, sliceOf(Bm, c.b.value.data, i), sliceOf(Am, c.a.grad.data, i), .{ .trans_b = !trans_b, .accumulate = true });
                    if (c.b.needs_grad) {
                        if (trans_b) {
                            try tensor.matmulInto(t.gpa, t.par, dc, sliceOf(Am, c.a.value.data, i), sliceOf(Bm, c.b.grad.data, i), .{ .trans_a = true, .accumulate = true });
                        } else {
                            try tensor.matmulInto(t.gpa, t.par, sliceOf(Am, c.a.value.data, i), dc, sliceOf(Bm, c.b.grad.data, i), .{ .trans_a = true, .accumulate = true });
                        }
                    }
                }
            }
        }.run);
        return out;
    }

    /// Gathers the rows of table named by ids; the backward pass scatters the gradient back with atomic adds.
    pub fn embedding(self: *Tape, table: anytype, comptime n: usize, ids: []const u32) Error!Var(Tensor(f32, &.{ n, @TypeOf(table.value).shape[1] })) {
        const width = @TypeOf(table.value).shape[1];
        std.debug.assert(ids.len == n);
        const value = try self.newTensor(Tensor(f32, &.{ n, width }));
        if (self.par.gpu) |dev| {
            try dev.embeddingGather(table.value.data, ids, value.data);
        } else {
            for (ids, 0..) |id, i| @memcpy(value.data[i * width ..][0..width], table.value.data[@as(usize, id) * width ..][0..width]);
        }
        const out = try self.makeVar(value, table.needs_grad);
        const Ctx = struct { table: @TypeOf(table), ids: []const u32, out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .table = table, .ids = ids, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.embeddingScatter(c.out.grad.data, c.ids, c.table.grad.data);
                for (c.ids, 0..) |id, i| {
                    for (c.table.grad.data[@as(usize, id) * width ..][0..width], c.out.grad.data[i * width ..][0..width]) |*g, d| g.* += d;
                }
            }
        }.run);
        return out;
    }

    /// Normalizes each row to zero mean and unit variance, then scales by gamma and shifts by beta.
    pub fn layerNorm(self: *Tape, x: anytype, gamma: anytype, beta: anytype) Error!Var(@TypeOf(x.value)) {
        const X = @TypeOf(x.value);
        comptime if (X.rank != 2 or @TypeOf(gamma.value).Elem != f32 or @TypeOf(gamma.value).shape[0] != X.shape[1]) @compileError("layerNorm expects a (rows, columns) tensor and f32 (columns) parameters");
        const rows = X.shape[0];
        const cols = X.shape[1];
        const Stats = Tensor(f32, &.{rows});
        const mean = try self.newTensor(Stats);
        const inv_std = try self.newTensor(Stats);
        const value = try self.newTensor(X);
        if (self.par.gpu) |dev| {
            try dev.layerNorm(.{ .x = x.value.data, .gamma = gamma.value.data, .mean = mean.data, .inv_std = inv_std.data }, beta.value.data, value.data);
        } else {
            if (comptime X.Elem != f32) return error.NoDevice;
            for (0..rows) |r| {
                const row = layernorm_math.forwardRow(f32, x.value.data.ptr + r * cols, value.data.ptr + r * cols, gamma.value.data.ptr, beta.value.data.ptr, cols);
                mean.data[r] = row.mean;
                inv_std.data[r] = row.inv_std;
            }
        }
        const out = try self.makeVar(value, x.needs_grad or gamma.needs_grad or beta.needs_grad);
        const Ctx = struct { x: @TypeOf(x), gamma: @TypeOf(gamma), beta: @TypeOf(beta), mean: Stats, inv_std: Stats, out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .gamma = gamma, .beta = beta, .mean = mean, .inv_std = inv_std, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| {
                    const p: gpu_mod.Gpu.LayerNorm(X.Elem) = .{ .x = c.x.value.data, .gamma = c.gamma.value.data, .mean = c.mean.data, .inv_std = c.inv_std.data };
                    return dev.layerNormBackward(p, c.out.grad.data, if (c.x.needs_grad) c.x.grad.data else null, if (c.gamma.needs_grad) c.gamma.grad.data else null, if (c.gamma.needs_grad) c.beta.grad.data else null);
                }
                if (comptime X.Elem != f32) return error.NoDevice;
                for (0..rows) |r| {
                    if (c.x.needs_grad) layernorm_math.backwardRow(f32, c.x.value.data.ptr + r * cols, c.out.grad.data.ptr + r * cols, c.x.grad.data.ptr + r * cols, c.gamma.value.data.ptr, .{ .mean = c.mean.data[r], .inv_std = c.inv_std.data[r] }, cols);
                }
                if (c.gamma.needs_grad) for (0..cols) |j| layernorm_math.parameterGradient(f32, c.x.value.data.ptr, c.out.grad.data.ptr, c.mean.data.ptr, c.inv_std.data.ptr, c.gamma.grad.data.ptr, c.beta.grad.data.ptr, rows, cols, j);
            }
        }.run);
        return out;
    }

    /// Applies softmax over the last axis. With causal_period above zero, row i only sees columns up to i % causal_period.
    pub fn softmax(self: *Tape, x: anytype, comptime causal_period: usize) Error!Var(@TypeOf(x.value)) {
        const X = @TypeOf(x.value);
        comptime if (X.rank != 2) @compileError("softmax expects a (rows, columns) tensor");
        const cols = X.shape[1];
        const value = try self.newTensor(X);
        if (self.par.gpu) |dev| {
            try dev.softmax(x.value.data, value.data, cols, causal_period);
        } else {
            if (comptime X.Elem != f32) return error.NoDevice;
            const Job = struct { x: [*]const f32, y: [*]f32 };
            kernels.parallelFor(self.par, X.shape[0], 16, Job{ .x = x.value.data.ptr, .y = value.data.ptr }, struct {
                fn run(job: Job, lo: usize, hi: usize) void {
                    for (lo..hi) |r| softmax_math.forwardRow(f32, job.x + r * cols, job.y + r * cols, cols, softmax_math.limitOf(r, cols, causal_period));
                }
            }.run);
        }
        const out = try self.makeVar(value, x.needs_grad);
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.softmaxBackward(c.out.value.data, c.out.grad.data, c.x.grad.data, cols);
                if (comptime X.Elem != f32) return error.NoDevice;
                for (0..X.shape[0]) |r| softmax_math.backwardRow(f32, c.out.value.data.ptr + r * cols, c.out.grad.data.ptr + r * cols, c.x.grad.data.ptr + r * cols, cols);
            }
        }.run);
        return out;
    }

    /// Converts the element type (f32 to bf16 and back); the gradient is converted the other way.
    pub fn cast(self: *Tape, x: anytype, comptime To: type) Error!Var(Tensor(To, @TypeOf(x.value).shape)) {
        const X = @TypeOf(x.value);
        if (X.Elem == To) return x;
        comptime if (!((X.Elem == f32 and To == u16) or (X.Elem == u16 and To == f32))) @compileError("cast converts between f32 and bf16 (stored as u16)");
        const dev = self.par.gpu orelse return error.NoDevice;
        const value = try self.newTensor(Tensor(To, X.shape));
        if (To == u16) try dev.toHalf(x.value.data, value.data) else try dev.fromHalf(x.value.data, value.data);
        const out = try self.makeVar(value, x.needs_grad);
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                const g = t.par.gpu.?;
                if (To == u16) try g.addFromHalf(c.out.grad.data, c.x.grad.data) else try g.addToHalf(c.out.grad.data, c.x.grad.data);
            }
        }.run);
        return out;
    }

    /// Applies one of the smooth activations in Activation (gelu, silu, tanh, ...) element by element.
    pub fn activate(self: *Tape, x: anytype, comptime kind: Activation) Error!Var(@TypeOf(x.value)) {
        const value = try self.newTensor(@TypeOf(x.value));
        const Job = struct { x: []const f32, y: []f32 };
        if (self.par.gpu) |dev| {
            try dev.activation(x.value.data, value.data, kind);
        } else {
            if (comptime @TypeOf(x.value).Elem != f32) return error.NoDevice;
            kernels.parallelFor(self.par, value.data.len, 1 << 14, Job{ .x = x.value.data, .y = value.data }, struct {
                fn run(job: Job, lo: usize, hi: usize) void {
                    activation_math.apply(job.x, job.y, kind, lo, hi);
                }
            }.run);
        }
        const out = try self.makeVar(value, x.needs_grad);
        try self.note(@field(trace_mod.Op, @tagName(kind)), .{x}, out, .{});
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.activationBackward(c.x.value.data, c.out.value.data, c.out.grad.data, c.x.grad.data, kind);
                if (comptime @TypeOf(c.x.value).Elem != f32) return error.NoDevice;
                activation_math.applyBackward(c.x.value.data, c.out.value.data, c.out.grad.data, c.x.grad.data, kind, 0, c.x.grad.data.len);
            }
        }.run);
        return out;
    }

    fn dropKernel(self: *Tape, src: anytype, dst: anytype, seed: u64, keep: f32, accumulate: bool) Error!void {
        if (self.par.gpu) |dev| return dev.dropout(src, dst, seed, keep, accumulate);
        if (comptime @typeInfo(@TypeOf(dst)).pointer.child != f32) return error.NoDevice;
        kernels.parallelEach(self.par, src.len, drop.apply, .{ src, dst, seed, keep, accumulate });
    }
    /// Zeroes each element with probability 1 - keep and scales the rest by 1 / keep; does nothing unless training is set.
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

    /// Normalizes over every axis except the last using batch statistics while training and running ones otherwise.
    pub fn batchNorm(self: *Tape, x: anytype, gamma: anytype, beta: anytype, running: anytype) Error!Var(@TypeOf(x.value)) {
        return self.batchNormImpl(false, x, gamma, beta, running);
    }

    /// Batch norm followed by ReLU, fused into one kernel.
    pub fn batchNormRelu(self: *Tape, x: anytype, gamma: anytype, beta: anytype, running: anytype) Error!Var(@TypeOf(x.value)) {
        return self.batchNormImpl(true, x, gamma, beta, running);
    }

    fn batchNormImpl(self: *Tape, comptime fuse_relu: bool, x: anytype, gamma: anytype, beta: anytype, running: anytype) Error!Var(@TypeOf(x.value)) {
        const In = @TypeOf(x.value);
        const c = In.shape[In.rank - 1];
        const Channels = Tensor(f32, &.{c});
        const mean = try self.newTensor(Channels);
        const inv_std = try self.newTensor(Channels);
        const value = try self.newTensor(In);
        const st: norm.Stats = .{ .mean = mean.data.ptr, .inv_std = inv_std.data.ptr, .running_mean = running.mean.data.ptr, .running_var = running.variance.data.ptr };
        const affine: norm.Affine = .{ .mean = st.mean, .inv_std = st.inv_std, .gamma = gamma.value.data.ptr, .beta = beta.value.data.ptr };
        const layout: norm.Layout = .{ .len = In.len, .channels = c, .flag = @intFromBool(fuse_relu) };
        if (self.par.gpu) |dev| {
            try dev.batchNormStats(x.value.data, st, c, self.training);
            try dev.batchNormApply(x.value.data, value.data, affine, layout);
        } else if (In.Elem == f32) {
            cpu.normStats(self.par, x.value.data, st, c, self.training);
            cpu.normApply(self.par, x.value.data, value.data, affine, layout);
        } else return error.NoDevice;
        const out = try self.makeVar(value, x.needs_grad or gamma.needs_grad or beta.needs_grad);
        try self.note(if (fuse_relu) .batch_norm_relu else .batch_norm, .{ x, gamma, beta, running.mean, running.variance }, out, .{});

        const Ctx = struct { x: @TypeOf(x), gamma: @TypeOf(gamma), beta: @TypeOf(beta), affine: norm.Affine, out: @TypeOf(out) };
        if (out.needs_grad and self.training) try self.recordBackward(Ctx{ .x = x, .gamma = gamma, .beta = beta, .affine = affine, .out = out }, struct {
            fn run(t: *Tape, k: *const Ctx) Error!void {
                const sums = try t.zeroedTensor(Tensor(f32, &.{2 * c}));
                const mask: []const In.Elem = if (fuse_relu) k.out.value.data else &.{};
                const grads: norm.GradOut = .{ .sums = sums.data, .gamma = k.gamma.grad.data, .beta = k.beta.grad.data };
                if (t.par.gpu) |dev| {
                    try dev.batchNormGradSums(k.x.value.data, k.out.grad.data, k.affine, mask, grads);
                    if (k.x.needs_grad) try dev.batchNormGradInput(batchNormGradIo(In.Elem, k, sums.data, fuse_relu), k.affine, layout);
                } else if (In.Elem == f32) {
                    cpu.normGradSums(t.par, k.x.value.data, k.out.grad.data, k.affine, mask, grads);
                    if (k.x.needs_grad) cpu.normGradInput(t.par, batchNormGradIo(In.Elem, k, sums.data, fuse_relu), k.affine, layout);
                }
            }
        }.run);
        return out;
    }

    /// Averages kernel x kernel windows of an NHWC tensor.
    pub fn avgPool(self: *Tape, x: anytype, comptime opts: PoolOpts) Error!Var(PoolOut(@TypeOf(x.value), opts)) {
        const In = @TypeOf(x.value);
        const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = In.shape[3], .k = opts.kernel, .stride = opts.stride, .pad = opts.pad };
        const value = try self.newTensor(PoolOut(In, opts));
        try self.convKernel(if (In.Elem == u16) .avgPoolHalf else .avgPool, .{ x.value.data, value.data }, d);
        const out = try self.makeVar(value, x.needs_grad);
        try self.note(.avg_pool, .{x}, out, .{ .kernel = opts.kernel, .stride = opts.stride, .pad = opts.pad });
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, k: *const Ctx) Error!void {
                try t.convKernel(if (In.Elem == u16) .avgPoolBackwardHalf else .avgPoolBackward, .{ k.out.grad.data, k.x.grad.data }, d);
            }
        }.run);
        return out;
    }

    fn convKernel(self: *Tape, comptime op: conv.Op, slices: anytype, d: conv.Dims) Error!void {
        if (self.par.gpu) |dev| return dev.convKernel(op, slices, d);
        switch (op) {
            .im2col => return cpu.im2col(self.par, slices[0], slices[1], d),
            .col2im => return cpu.col2im(self.par, slices[0], slices[1], d),
            else => {},
        }
        kernels.parallelEach(self.par, conv.workItems(op, d), @field(conv, @tagName(op)), slices ++ .{d});
    }
    fn convHalf(self: *Tape, x: anytype, w: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts)) {
        const In = @TypeOf(x.value);
        const Out = ConvOut(In, @TypeOf(w.value), opts);
        const k = comptime kernelSize(In, @TypeOf(w.value), opts.groups);
        const c_in = In.shape[3];
        const c_out = Out.shape[3];
        comptime if (opts.stride != 1 or opts.groups != 1 or opts.dilation != 1 or c_in % 64 != 0 or c_out % 32 != 0) @compileError("bf16 activations need stride-1 convolutions with a multiple of 64 input and of 32 output channels; cast to f32 around other layers");
        const dev = self.par.gpu orelse return error.NoDevice;
        const rows = Out.len / c_out;
        const kc = k * k * c_in;
        const geometry: ga.ConvGeometry = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = k, .stride = 1, .pad = opts.pad, .out_h = Out.shape[1], .out_w = Out.shape[2], .positions = rows, .c_out = c_out };
        const reverse: ga.ConvGeometry = .{ .n = In.shape[0], .h = Out.shape[1], .w = Out.shape[2], .c = c_out, .k = k, .stride = 1, .pad = k - 1 - opts.pad, .out_h = In.shape[1], .out_w = In.shape[2], .positions = In.len / c_in, .c_out = c_in };

        const value = try self.newTensor(Out);
        {
            const mark = dev.step_used;
            defer dev.step_used = mark;
            const w_half = try dev.packMatrix(w.value.data, kc, c_out, kc, std.mem.alignForward(usize, c_out, 64), false);
            try dev.convImplicitForward(x.value.data, w_half, value.data, geometry, false);
        }
        const out = try self.makeVar(value, x.needs_grad or w.needs_grad);
        const Ctx = struct { x: @TypeOf(x), w: @TypeOf(w), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .w = w, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                const g = t.par.gpu.?;
                const mark = g.step_used;
                defer g.step_used = mark;
                if (c.x.needs_grad) {
                    const rotated = try g.packRotatedWeights(c.w.value.data, k, c_in, c_out);
                    try g.convImplicitForward(c.out.grad.data, rotated, c.x.grad.data, reverse, true);
                }
                if (c.w.needs_grad) {
                    const grad_t = try g.packMatrix(c.out.grad.data, c_out, rows, std.mem.alignForward(usize, c_out, 128), std.mem.alignForward(usize, rows, 32), true);
                    try g.convImplicitWeightGrad(grad_t, c.x.value.data, c.w.grad.data, geometry);
                }
            }
        }.run);
        return out;
    }

    /// Convolves an NHWC input with weights of shape (k*k*c/groups, out).
    /// NOTE: the algorithm (direct, packed bf16, implicit GEMM, TF32, cuDNN, grouped) is chosen once per call from the shapes and device flags.
    pub fn conv2d(self: *Tape, x: anytype, w: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts)) {
        const In = @TypeOf(x.value);
        if (comptime In.Elem == u16) return self.convHalf(x, w, opts);
        const Out = ConvOut(In, @TypeOf(w.value), opts);
        const k = comptime kernelSize(In, @TypeOf(w.value), opts.groups);
        const c_in = In.shape[3];
        const c_out = Out.shape[3];
        const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = k, .stride = opts.stride, .pad = opts.pad, .dilation = opts.dilation, .groups = opts.groups, .c_out = c_out };
        const rows = Out.len / c_out;
        const kc = k * k * c_in;
        const Col = Tensor(f32, &.{ rows, kc });
        const Flat = Tensor(f32, &.{ rows, c_out });
        const geometry: ga.ConvGeometry = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = k, .stride = opts.stride, .pad = opts.pad, .out_h = Out.shape[1], .out_w = Out.shape[2], .positions = rows, .c_out = c_out };
        const reverse: ga.ConvGeometry = .{ .n = In.shape[0], .h = Out.shape[1], .w = Out.shape[2], .c = c_out, .k = k, .stride = 1, .pad = k - 1 - opts.pad, .out_h = In.shape[1], .out_w = In.shape[2], .positions = In.len / c_in, .c_out = c_in };
        const cudnn_geometry: gpu_mod.cudnn.Geometry = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = c_out, .r = k, .pad = opts.pad, .stride = opts.stride };
        const aligned_out = std.mem.alignForward(usize, c_out, 64);

        const dev = self.par.gpu;
        const unit_stride_blocks = opts.stride == 1 and c_in % 64 == 0;
        const general = opts.groups > 1 or opts.dilation > 1;
        const path: ConvPath = if (general) .grouped else if (dev == null) .direct else if (dev.?.cudnn_lib != null) .cudnn else if (dev.?.tensorWorthwhile(rows, c_out, kc)) (if (unit_stride_blocks and c_out % 32 == 0) .implicit else .packed_im2col) else if (unit_stride_blocks and c_out % 16 == 0 and dev.?.tf32Worthwhile(rows, c_out, kc)) .tf32 else .direct;

        const value = try self.newTensor(Out);
        var col: Col = undefined;
        var col_half: Packed = .{ .data = &.{}, .rows_padded = 0, .cols_padded = 0 };
        var x_half: []const u16 = &.{};
        switch (path) {
            .grouped => try self.convKernel(.groupedForward, .{ x.value.data, w.value.data, value.data }, d),
            .direct => {
                col = try self.newTensor(Col);
                try self.convKernel(.im2col, .{ x.value.data, col.data }, d);
                try tensor.matmulInto(self.gpa, self.par, col, w.value, Flat{ .data = value.data }, .{});
            },
            .packed_im2col => {
                const g = dev.?;
                col_half = try g.im2colPacked(x.value.data, d);
                const w_half = try g.packMatrix(w.value.data, kc, c_out, col_half.cols_padded, aligned_out, false);
                try g.matmulPacked(col_half, w_half, value.data, .{ .m = rows, .n = c_out, .k = kc }, false, false);
            },
            .implicit => {
                const g = dev.?;
                x_half = (try g.packMatrix(x.value.data, In.len / c_in, c_in, In.len / c_in, c_in, false)).data;
                const w_half = try g.packMatrix(w.value.data, kc, c_out, kc, aligned_out, false);
                try g.convImplicitForward(x_half, w_half, value.data, geometry, false);
            },
            .tf32 => try dev.?.convTf32Forward(x.value.data, w.value.data, c_out, value.data, geometry, false),
            .cudnn => {
                const g = dev.?;
                const mark = g.step_used;
                defer g.step_used = mark;
                try g.cudnnForward(cudnn_geometry, x.value.data, try g.weightsToKrsc(w.value.data, k, c_in, c_out), value.data);
            },
        }
        const out = try self.makeVar(value, x.needs_grad or w.needs_grad);
        try self.note(.conv, .{ x, w }, out, .{ .kernel = k, .stride = opts.stride, .pad = opts.pad, .dilation = opts.dilation, .groups = opts.groups });

        const Ctx = struct { x: @TypeOf(x), w: @TypeOf(w), path: ConvPath, col: Col, col_half: Packed, x_half: []const u16, out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .w = w, .path = path, .col = col, .col_half = col_half, .x_half = x_half, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                const grad: Flat = .{ .data = c.out.grad.data };
                const device = t.par.gpu;
                const mark = if (device) |g| g.step_used else 0;
                defer if (device) |g| {
                    g.step_used = mark;
                };
                switch (if (general) ConvPath.grouped else c.path) {
                    .grouped => {
                        if (c.x.needs_grad) try t.convKernel(.groupedInputGrad, .{ grad.data, c.w.value.data, c.x.grad.data }, d);
                        if (c.w.needs_grad) try t.convKernel(.groupedWeightGrad, .{ c.x.value.data, grad.data, c.w.grad.data }, d);
                    },
                    .direct => {
                        if (c.x.needs_grad) {
                            const dcol = try t.newTensor(Col);
                            try tensor.matmulInto(t.gpa, t.par, grad, c.w.value, dcol, .{ .trans_b = true });
                            try t.convKernel(.col2im, .{ dcol.data, c.x.grad.data }, d);
                        }
                        if (c.w.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.col, grad, c.w.grad, .{ .trans_a = true, .accumulate = true });
                    },
                    .packed_im2col => {
                        const g = t.par.gpu.?;
                        if (c.x.needs_grad) {
                            const grad_packed = try g.packMatrix(grad.data, rows, c_out, std.mem.alignForward(usize, rows, 128), std.mem.alignForward(usize, c_out, 32), false);
                            const weight_packed = try g.packMatrix(c.w.value.data, c_out, kc, std.mem.alignForward(usize, c_out, 32), std.mem.alignForward(usize, kc, 64), true);
                            const dcol_half = try g.scratch(u16, rows * kc);
                            try g.matmulPacked(grad_packed, weight_packed, dcol_half, .{ .m = rows, .n = kc, .k = c_out }, false, false);
                            try t.convKernel(.col2imHalf, .{ dcol_half, c.x.grad.data }, d);
                        }
                        if (c.w.needs_grad) {
                            const grad_t = try g.packMatrix(grad.data, c_out, rows, std.mem.alignForward(usize, c_out, 128), c.col_half.rows_padded, true);
                            try g.matmulPacked(grad_t, c.col_half, c.w.grad.data, .{ .m = c_out, .n = kc, .k = rows }, true, true);
                        }
                    },
                    .implicit => {
                        const g = t.par.gpu.?;
                        if (c.x.needs_grad) {
                            const grad_half = try g.packMatrix(grad.data, rows, c_out, rows, c_out, false);
                            const rotated = try g.packRotatedWeights(c.w.value.data, k, c_in, c_out);
                            try g.convImplicitForward(grad_half.data, rotated, c.x.grad.data, reverse, true);
                        }
                        if (c.w.needs_grad) {
                            const grad_t = try g.packMatrix(grad.data, c_out, rows, std.mem.alignForward(usize, c_out, 128), std.mem.alignForward(usize, rows, 32), true);
                            try g.convImplicitWeightGrad(grad_t, c.x_half, c.w.grad.data, geometry);
                        }
                    },
                    .tf32 => {
                        const g = t.par.gpu.?;
                        if (c.x.needs_grad) {
                            const rotated = try g.rotateWeightsF32(c.w.value.data, k, c_in, c_out);
                            try g.convTf32Forward(grad.data, rotated, std.mem.alignForward(usize, c_in, 64), c.x.grad.data, reverse, true);
                        }
                        if (c.w.needs_grad) {
                            const positions_padded = std.mem.alignForward(usize, rows, 32);
                            const grad_t = try g.transposeF32(grad.data, c_out, rows, std.mem.alignForward(usize, c_out, 32), positions_padded);
                            try g.convTf32WeightGrad(grad_t, c.x.value.data, c.w.grad.data, geometry);
                        }
                    },
                    .cudnn => {
                        const g = t.par.gpu.?;
                        const w_krsc = try g.weightsToKrsc(c.w.value.data, k, c_in, c_out);
                        if (c.x.needs_grad) try g.cudnnBackwardData(cudnn_geometry, w_krsc, c.out.grad.data, c.x.grad.data);
                        if (c.w.needs_grad) {
                            const dw_krsc = try g.scratch(f32, w_krsc.len);
                            try g.cudnnBackwardFilter(cudnn_geometry, c.x.value.data, c.out.grad.data, dw_krsc);
                            try g.addKrscGradient(dw_krsc, c.w.grad.data, k, c_in, c_out);
                        }
                    },
                }
            }
        }.run);
        return out;
    }
    /// Takes the maximum over kernel x kernel windows of an NHWC tensor.
    pub fn maxPool(self: *Tape, x: anytype, comptime opts: PoolOpts) Error!Var(PoolOut(@TypeOf(x.value), opts)) {
        const In = @TypeOf(x.value);
        const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = In.shape[3], .k = opts.kernel, .stride = opts.stride, .pad = opts.pad };
        const value = try self.newTensor(PoolOut(In, opts));
        const index = if (self.par.gpu) |dev| try dev.scratch(u32, value.data.len) else try self.gpa.alloc(u32, value.data.len);
        try self.convKernel(if (In.Elem == u16) .maxPoolHalf else .maxPool, .{ x.value.data, value.data, index }, d);
        const out = try self.makeVar(value, x.needs_grad);
        try self.note(.max_pool, .{x}, out, .{ .kernel = opts.kernel, .stride = opts.stride, .pad = opts.pad });
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out), index: []u32 };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out, .index = index }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                try t.convKernel(if (In.Elem == u16) .maxPoolBackwardHalf else .maxPoolBackward, .{ c.index, c.out.grad.data, c.x.grad.data }, d);
            }
        }.run);
        return out;
    }

    /// Returns the mean softmax cross-entropy of logits against class labels, with label smoothing from smoothing.
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
            try dev.crossEntropy(logits.value.data, logp.data, labels, value.data, cols, self.smoothing);
        } else {
            logp = try ops.logSoftmax(self.gpa, logits.value);
            var total: f64 = 0;
            for (labels, 0..) |y, i| {
                total -= (1 - self.smoothing) * logp.data[i * cols + y];
                for (logp.data[i * cols ..][0..cols]) |lp| total -= self.smoothing / cols * lp;
            }
            value = try Scalar.empty(self.gpa);
            value.data[0] = @floatCast(total / rows);
        }
        const out = try self.makeVar(value, logits.needs_grad);

        const Ctx = struct { logits: @TypeOf(logits), logp: L, labels: []const u8, smoothing: f32, out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .logits = logits, .logp = logp, .labels = labels, .smoothing = self.smoothing, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.crossEntropyBackward(c.logp.data, c.labels, c.logits.grad.data, c.out.grad.data, cols, c.smoothing);
                const scale = c.out.grad.data[0] / rows;
                for (0..rows) |i| {
                    for (0..cols) |j| {
                        var p = @exp(c.logp.data[i * cols + j]) - c.smoothing / cols;
                        if (j == c.labels[i]) p -= 1 - c.smoothing;
                        c.logits.grad.data[i * cols + j] += p * scale;
                    }
                }
            }
        }.run);
        return out;
    }

    /// Returns the mean squared error between prediction and target.
    pub fn mse(self: *Tape, prediction: anytype, target: anytype) Error!Var(Tensor(@TypeOf(prediction.value).Elem, &.{})) {
        const P = @TypeOf(prediction.value);
        comptime {
            if (P.rank != 2 or !std.mem.eql(usize, P.shape, @TypeOf(target.value).shape)) @compileError("mse expects two (batch, features) tensors of the same shape");
        }
        const minus_one = try self.constant(try self.filledTensor(Tensor(P.Elem, &.{}), -1));
        const difference = try self.add(prediction, try self.mul(target, minus_one));
        const squared = try self.mul(difference, difference);
        const row_means = try self.matmul(squared, try self.constant(try self.filledTensor(Tensor(P.Elem, &.{ P.shape[1], 1 }), 1.0 / @as(f32, P.shape[1]))));
        const mean = try self.matmul(try self.constant(try self.filledTensor(Tensor(P.Elem, &.{ 1, P.shape[0] }), 1.0 / @as(f32, P.shape[0]))), row_means);
        return reshape(mean, &.{});
    }

    /// Seeds the gradient of the scalar loss with one and runs the recorded backward closures in reverse order.
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
    const gpu = try gpu_mod.Gpu.forTest(1 << 20);
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

const LayerNormNet = struct {
    const X = Tensor(f32, &.{ 5, 6 });
    const C = Tensor(f32, &.{6});

    pub const Inputs = struct { x: X, gamma: C, beta: C, y: X };
    pub const names = .{ "x", "gamma", "beta", "y" };
    pub const labels: [5]u8 = @splat(0);
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

fn SoftmaxNet(comptime period: usize) type {
    return struct {
        const X = Tensor(f32, &.{ 6, 3 });
        pub const Inputs = struct { x: X, y: X };
        pub const names = .{ "x", "y" };
        pub const labels: [6]u8 = @splat(0);
        const Result = struct { loss: Var(Tensor(f32, &.{})), x: Var(X), y: Var(X) };

        pub fn build(t: *Tape, p: Inputs, _: []const u8) !Result {
            const x = try t.leaf(p.x);
            const y = try t.leaf(p.y);
            return .{ .loss = try t.mse(try t.softmax(x, period), y), .x = x, .y = y };
        }

        pub fn acceptable(_: Allocator, _: Inputs) !bool {
            return true;
        }
    };
}

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

fn runBatchNorm(comptime half: bool, gpu: *gpu_mod.Gpu, x_values: []const f32, dy_values: []const f32, out_values: []f32, dx_values: []f32, running: []f32) !void {
    const X = Tensor(f32, &.{ 600, 16 });
    const Channels = Tensor(f32, &.{16});
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try gpu.resetStep();
    var t: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .training = true };
    const x = try t.leaf(X{ .data = try gpu.alloc(f32, X.len) });
    try gpu.upload(x.value.data, x_values);
    try gpu.fill(x.grad.data, 0);
    const gamma = try t.leaf(Channels{ .data = try gpu.alloc(f32, 16) });
    const beta = try t.leaf(Channels{ .data = try gpu.alloc(f32, 16) });
    try gpu.fill(gamma.value.data, 1.5);
    try gpu.fill(beta.value.data, 0.25);
    try gpu.fill(gamma.grad.data, 0);
    try gpu.fill(beta.grad.data, 0);
    const mean = Channels{ .data = try gpu.alloc(f32, 16) };
    const variance = Channels{ .data = try gpu.alloc(f32, 16) };
    try gpu.fill(mean.data, 0);
    try gpu.fill(variance.data, 1);

    const input = if (half) try t.cast(x, u16) else x;
    const out = try t.batchNormRelu(input, gamma, beta, .{ .mean = mean, .variance = variance });
    const back = try t.cast(out, f32);
    try gpu.download(out_values, back.value.data);
    try gpu.upload(back.grad.data, dy_values);
    var at = t.nodes.items.len;
    while (at > 0) {
        at -= 1;
        try t.nodes.items[at].run(&t, t.nodes.items[at].ctx);
    }
    try gpu.download(dx_values, x.grad.data);
    try gpu.download(running[0..16], mean.data);
    try gpu.download(running[16..], variance.data);
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
    try runBatchNorm(false, gpu, &x, &dy, &wide_out, &wide_dx, &wide_running);
    try runBatchNorm(true, gpu, &x, &dy, &half_out, &half_dx, &half_running);
    for (wide_out, half_out) |want, got| try testing.expectApproxEqAbs(want, got, 0.04 + 0.01 * @abs(want));
    var flipped: usize = 0;
    for (wide_dx, half_dx) |want, got| {
        if (@abs(want - got) > 0.03 + 0.02 * @abs(want)) flipped += 1;
    }
    try testing.expect(flipped <= 3);
    for (wide_running, half_running) |want, got| try testing.expectApproxEqAbs(want, got, 0.02 + 0.01 * @abs(want));
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

test "softmax rows, with and without a causal mask, have correct values and gradients on the cpu and the gpu" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator() };
    const x = try t.leaf(try Tensor(f32, &.{ 3, 3 }).fromSlice(t.gpa, &.{ 0, 0, 0, 0, 0, 0, 1, 2, 3 }));
    const p = try t.softmax(x, 3);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 0, 0.5, 0.5, 0, 0.09003057, 0.24472848, 0.66524094 }, p.value.data);
    inline for (.{ 0, 3 }) |period| {
        try gradcheck.expectFiniteDifferences(SoftmaxNet(period), 7, 1e-2, 3e-3);
        try gradcheck.expectGpuMatchesCpu(SoftmaxNet(period), 8, 1e-5);
    }
}

test "layer norm has correct gradients for the input, the scale and the shift, on the cpu and the gpu" {
    try gradcheck.expectFiniteDifferences(LayerNormNet, 9, 1e-2, 5e-3);
    try gradcheck.expectGpuMatchesCpu(LayerNormNet, 10, 1e-4);
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
    const rows = try on_cpu.embedding(table, 4, &host_ids);
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
    const gathered = try on_gpu.embedding(gpu_table, 4, device_ids);
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
    const probabilities = try t.cast(try t.softmax(sharpened, 4), f32);
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
}
