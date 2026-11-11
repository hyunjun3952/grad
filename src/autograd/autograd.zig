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

pub fn ConvOut(comptime In: type, comptime W: type, comptime opts: ConvOpts) type {
    const span = opts.dilation * (kernelSize(In, W, opts.groups) - 1) + 1;
    return Tensor(In.Elem, &.{ In.shape[0], (In.shape[1] + 2 * opts.pad - span) / opts.stride + 1, (In.shape[2] + 2 * opts.pad - span) / opts.stride + 1, W.shape[1] });
}

pub fn PoolOut(comptime In: type, comptime opts: PoolOpts) type {
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
    pub fn batchedMatmul(self: *Tape, a: anytype, b: anytype, comptime trans_b: bool) Error!Var(Tensor(@TypeOf(a.value).Elem, &.{ @TypeOf(a.value).shape[0], @TypeOf(a.value).shape[1], if (trans_b) @TypeOf(b.value).shape[1] else @TypeOf(b.value).shape[2] })) {
        const A = @TypeOf(a.value);
        const B = @TypeOf(b.value);
        comptime if (A.rank != 3 or B.rank != 3 or A.Elem != B.Elem or A.shape[0] != B.shape[0] or A.shape[2] != (if (trans_b) B.shape[2] else B.shape[1])) @compileError("batchedMatmul expects (batch, m, k) and (batch, k, n) tensors, or (batch, n, k) with trans_b, of one element type");
        // NOTE: attention-sized matrices are too small for the tensor core path (packing each one costs more than the product),
        // so bf16 operands are widened once, multiplied per matrix in f32 and rounded once; the casts carry the gradients.
        if (comptime A.Elem == u16) return self.cast(try self.batchedMatmul(try self.cast(a, f32), try self.cast(b, f32), trans_b), u16);
        const batch = A.shape[0];
        const m = A.shape[1];
        const k = A.shape[2];
        const n = if (trans_b) B.shape[1] else B.shape[2];
        const Am = Tensor(A.Elem, &.{ m, k });
        const Bm = Tensor(A.Elem, if (trans_b) &.{ n, k } else &.{ k, n });
        const Cm = Tensor(A.Elem, &.{ m, n });
        const C = Tensor(A.Elem, &.{ batch, m, n });
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

    /// Gathers the rows of the f32 table named by ids into a tensor of element type Out (f32 or bf16); the backward pass scatters the gradient back with atomic adds.
    /// NOTE: the table is a parameter and stays f32, because the scatter needs f32 atomics; Out only sets the type of the activations.
    pub fn embedding(self: *Tape, table: anytype, comptime n: usize, ids: []const u32, comptime Out: type) Error!Var(Tensor(Out, &.{ n, @TypeOf(table.value).shape[1] })) {
        const width = @TypeOf(table.value).shape[1];
        std.debug.assert(ids.len == n);
        const value = try self.newTensor(Tensor(Out, &.{ n, width }));
        if (self.par.gpu) |dev| {
            try dev.embeddingGather(table.value.data, ids, value.data);
        } else {
            if (comptime Out != f32) return error.NoDevice;
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
        try self.note(.cast, .{x}, out, .{});
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
        try self.note(.conv, .{ x, w }, out, .{ .kernel = k, .pad = opts.pad });
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
    /// bf16 logits are converted to f32 first, so the loss and its statistics are always computed in f32.
    pub fn crossEntropy(self: *Tape, raw_logits: anytype, labels: []const u8) Error!Var(Tensor(f32, &.{})) {
        const logits = try self.cast(raw_logits, f32);
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
    /// bf16 inputs are converted to f32 first, so the loss is always computed in f32.
    pub fn mse(self: *Tape, raw_prediction: anytype, raw_target: anytype) Error!Var(Tensor(f32, &.{})) {
        const prediction = try self.cast(raw_prediction, f32);
        const target = try self.cast(raw_target, f32);
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
