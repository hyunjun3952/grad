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
const gru_math = @import("../gpu/math/gru.zig");
const softmax_math = @import("../gpu/math/softmax.zig");
const layernorm_math = @import("../gpu/math/layernorm.zig");
const drop = @import("../gpu/math/dropout.zig");
const norm = @import("../gpu/math/norm.zig");
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

// What batch norm fuses after its affine map: nothing, a ReLU, or a residual addition followed by a ReLU.
const NormFuse = enum { none, relu, add_relu };

fn batchNormGradIo(comptime E: type, k: anytype, sums: []const f32, comptime fuse: NormFuse) norm.GradIo(E) {
    const dr = if (fuse == .add_relu) k.residual.grad.data.ptr else k.x.grad.data.ptr;
    return .{ .x = k.x.value.data.ptr, .dy = k.out.grad.data.ptr, .dx = k.x.grad.data.ptr, .sums = sums.ptr, .mask = if (fuse != .none) k.out.value.data.ptr else k.out.grad.data.ptr, .dr = dr };
}

pub const Activation = activation_math.Kind;
pub const ConvOpts = struct { stride: usize = 1, pad: usize = 0, dilation: usize = 1, groups: usize = 1 };
pub const PoolOpts = struct { kernel: usize = 2, stride: usize = 2, pad: usize = 0 };
pub const SoftmaxOpts = struct { causal_period: usize = 0, scale: f32 = 1 };
const ConvPath = enum { cpu, direct, packed_im2col, implicit, f32_implicit, cudnn, grouped, grouped_implicit };
const Packed = gpu_mod.Gpu.Packed;

fn sliceOf(comptime Tn: type, data: anytype, index: usize) Tn {
    return .{ .data = data[index * Tn.len ..][0..Tn.len] };
}

// Multiplies batch consecutive matrices of the given types: one launch with a batch axis on the device, one product per matrix on the CPU.
fn batchedInto(gpa: Allocator, par: kernels.Parallel, comptime batch: usize, comptime Am: type, comptime Bm: type, comptime Cm: type, a: anytype, b: anytype, c: anytype, comptime opts: gpu_mod.MatmulOpts) Error!void {
    if (par.gpu) |dev| {
        const k = if (opts.trans_a) Am.shape[0] else Am.shape[1];
        return dev.matmul(a, b, c, .{ .m = Cm.shape[0], .n = Cm.shape[1], .k = k, .batch = batch }, opts);
    }
    for (0..batch) |i| try tensor.matmulInto(gpa, par, sliceOf(Am, a, i), sliceOf(Bm, b, i), sliceOf(Cm, c, i), opts);
}

fn kernelSize(comptime In: type, comptime W: type, comptime groups: usize) usize {
    if (In.rank != 4 or W.rank != 2) @compileError("conv2d expects (n, h, w, c) input and (k*k*c/groups, out) weights");
    if (In.shape[3] % groups != 0 or W.shape[1] % groups != 0) @compileError("conv2d groups must divide the input and output channels");
    const per_group = In.shape[3] / groups;
    const k = std.math.sqrt(W.shape[0] / per_group);
    if (k * k * per_group != W.shape[0]) @compileError("conv2d weights must have k*k*c/groups rows");
    return k;
}

const ConvGeometryOpts = struct { reverse: bool = false, per_group: bool = false };

fn convGeometry(d: conv.Dims, opts: ConvGeometryOpts) ga.ConvGeometry {
    const out_h = conv.outHeight(d);
    const out_w = conv.outWidth(d);
    if (opts.reverse) {
        // NOTE: rotated weights span dilation * (k - 1) + 1; a negative reverse pad is clamped because the gather cannot crop.
        const reverse_pad = if (d.dilation * (d.k - 1) < d.pad) 0 else d.dilation * (d.k - 1) - d.pad;
        return .{
            .n = d.n,
            .h = out_h,
            .w = out_w,
            .c = if (opts.per_group) d.c_out / d.groups else d.c_out,
            .k = d.k,
            .stride = 1,
            .pad = reverse_pad,
            .out_h = d.h,
            .out_w = d.w,
            .positions = d.n * d.h * d.w,
            .c_out = if (opts.per_group) d.c / d.groups else d.c,
            .dilation = d.dilation,
            .c_step = if (opts.per_group) d.c_out else 0,
        };
    }
    return .{
        .n = d.n,
        .h = d.h,
        .w = d.w,
        .c = if (opts.per_group) d.c / d.groups else d.c,
        .k = d.k,
        .stride = d.stride,
        .pad = d.pad,
        .out_h = out_h,
        .out_w = out_w,
        .positions = d.n * out_h * out_w,
        .c_out = if (opts.per_group) d.c_out / d.groups else d.c_out,
        .dilation = d.dilation,
        .c_step = if (opts.per_group) d.c else 0,
    };
}

fn chooseConvPath(d: conv.Dims, dev: ?*gpu_mod.Gpu) ConvPath {
    const rows = @as(usize, d.n) * conv.outHeight(d) * conv.outWidth(d);
    const taps: usize = conv.patchLen(d);
    const unit_stride_blocks = d.stride == 1 and d.c % 64 == 0;
    const general = d.dilation * (d.k - 1) < d.pad;
    const per_in = d.c / d.groups;
    const per_out = d.c_out / d.groups;
    // NOTE: a grouped stride-1 convolution is one implicit-GEMM launch per group when the per-group channels are
    // multiples of 16 and the products are large enough; everything else about it keeps the simple kernel.
    if (d.groups > 1 and d.stride == 1 and !general and per_in % 16 == 0 and per_out % 16 == 0 and
        dev != null and dev.?.f32ImplicitWorthwhile(rows, per_out, d.k * d.k * per_in)) return .grouped_implicit;
    if (d.groups > 1 or general) return .grouped;
    const g = dev orelse return .cpu;
    if (g.cudnn_lib != null and d.dilation == 1) return .cudnn;
    if (g.tensorWorthwhile(rows, d.c_out, taps)) {
        return if (unit_stride_blocks and d.c_out % 32 == 0) .implicit else .packed_im2col;
    }
    if (unit_stride_blocks and d.c_out % 16 == 0 and g.f32ImplicitWorthwhile(rows, d.c_out, taps)) return .f32_implicit;
    return .direct;
}

pub fn ConvOut(comptime In: type, comptime W: type, comptime opts: ConvOpts) type {
    const span = opts.dilation * (kernelSize(In, W, opts.groups) - 1) + 1;
    return Tensor(In.Elem, &.{ In.shape[0], (In.shape[1] + 2 * opts.pad - span) / opts.stride + 1, (In.shape[2] + 2 * opts.pad - span) / opts.stride + 1, W.shape[1] });
}

pub fn PoolOut(comptime In: type, comptime opts: PoolOpts) type {
    if (In.rank != 4) @compileError("pooling expects (n, h, w, c) input");
    return Tensor(In.Elem, &.{ In.shape[0], (In.shape[1] + 2 * opts.pad - opts.kernel) / opts.stride + 1, (In.shape[2] + 2 * opts.pad - opts.kernel) / opts.stride + 1, In.shape[3] });
}

// ***** cpu element-wise kernels *****

const PairJob = struct { dst: []f32, src: []const f32 };

fn addRange(job: PairJob, lo: usize, hi: usize) void {
    for (job.dst[lo..hi], job.src[lo..hi]) |*d, s| d.* += s;
}

fn reluRange(job: PairJob, lo: usize, hi: usize) void {
    for (job.dst[lo..hi], job.src[lo..hi]) |*d, s| d.* = @max(s, 0);
}

const ReluGradJob = struct { dst: []f32, x: []const f32, g: []const f32 };

fn reluGradRange(job: ReluGradJob, lo: usize, hi: usize) void {
    for (job.dst[lo..hi], job.x[lo..hi], job.g[lo..hi]) |*d, v, g| d.* += if (v > 0) g else 0;
}

// out = a op b over rows of cols values; with repeat_b every row uses the same vector b instead of its own row of b.
const RowJob = struct { a: [*]const f32, b: [*]const f32, out: [*]f32, cols: usize, repeat_b: bool, multiply: bool };

fn rowElementwise(job: RowJob, lo: usize, hi: usize) void {
    for (lo..hi) |row| {
        const a = job.a[row * job.cols ..][0..job.cols];
        const b = if (job.repeat_b) job.b[0..job.cols] else job.b[row * job.cols ..][0..job.cols];
        const out = job.out[row * job.cols ..][0..job.cols];
        if (job.multiply) {
            for (out, a, b) |*o, x, y| o.* = x * y;
        } else {
            for (out, a, b) |*o, x, y| o.* = x + y;
        }
    }
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
        const zeroed = try self.zeros(Tn);
        if (self.trace) |t| try t.addConstant(zeroed, 0);
        return zeroed;
    }

    // NOTE: gradient buffers must start at zero because every backward closure adds into them; on the GPU they come from the zero arena.
    fn zeros(self: *Tape, comptime Tn: type) Error!Tn {
        if (self.par.gpu) |dev| return .{ .data = try dev.scratchZeroed(Tn.Elem, Tn.len) };
        const t = try self.newTensor(Tn);
        @memset(t.data, 0);
        return t;
    }

    fn makeVar(self: *Tape, value: anytype, needs_grad: bool) Error!Var(@TypeOf(value)) {
        if (!needs_grad) return .{ .value = value, .grad = undefined, .needs_grad = false };
        return .{ .value = value, .grad = try self.zeros(@TypeOf(value)), .needs_grad = true };
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
        const A = @TypeOf(a);
        const B = @TypeOf(b);
        const R = ops.BroadcastResult(A, B);
        // NOTE: the two shapes that dominate networks (equal shapes, and a vector added or multiplied to every row) run on
        // the worker pool; anything else takes the general broadcasting loop of ops.
        const equal = comptime std.mem.eql(usize, A.shape, B.shape);
        const per_row = comptime A.rank >= 2 and B.rank == 1 and B.shape[0] == A.shape[A.rank - 1] and std.mem.eql(usize, R.shape, A.shape);
        if (comptime equal or per_row) {
            const out = try R.empty(self.gpa);
            const cols = if (per_row) B.len else A.len;
            const job: RowJob = .{ .a = a.data.ptr, .b = b.data.ptr, .out = out.data.ptr, .cols = cols, .repeat_b = per_row, .multiply = op == .mul };
            kernels.parallelFor(self.par, A.len / cols, @max(1, (1 << 14) / cols), job, rowElementwise);
            return out;
        }
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
            kernels.parallelFor(self.par, D.len, 1 << 15, PairJob{ .dst = v.grad.data, .src = g.data }, addRange);
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

    /// Multiplies x (rows, n_in) by w (n_in, n_out) and adds the bias b (n_out) to every row: the body of a fully connected layer.
    /// NOTE: one op instead of matmul followed by add: the bias goes into the product in place and the backward pass writes
    /// the gradients of x and w straight from the output gradient, which saves a buffer and a launch per layer and step.
    pub fn linear(self: *Tape, x: anytype, w: anytype, b: anytype) Error!Var(tensor.MatmulResult(@TypeOf(x.value), @TypeOf(w.value), .{})) {
        const Out = tensor.MatmulResult(@TypeOf(x.value), @TypeOf(w.value), .{});
        const B = @TypeOf(b.value);
        comptime if (Out.Elem != f32 or B.Elem != f32 or B.rank != 1 or B.shape[0] != Out.shape[1]) @compileError("linear takes f32 activations and an f32 bias with one value per output column");
        const value = try self.newTensor(Out);
        try tensor.matmulInto(self.gpa, self.par, x.value, w.value, value, .{});
        if (self.par.gpu) |dev| {
            const bc = comptime ops.broadcast(Out.shape, B.shape);
            // Reading and writing the same element is safe, so the sum is formed in the product's own buffer.
            try dev.elementwise(.add, value.data, b.value.data, value.data, .{ .dims = padToRank(&bc.dims, 1), .a_strides = padToRank(&bc.a, 0), .b_strides = padToRank(&bc.b, 0) });
        } else {
            for (0..Out.shape[0]) |r| {
                for (value.data[r * B.len ..][0..B.len], b.value.data) |*v, bias| v.* += bias;
            }
        }
        const out = try self.makeVar(value, x.needs_grad or w.needs_grad or b.needs_grad);
        try self.note(.linear, .{ x, w, b }, out, .{});
        const Ctx = struct { x: @TypeOf(x), w: @TypeOf(w), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .w = w, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (c.x.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.out.grad, c.w.value, c.x.grad, .{ .trans_b = true, .accumulate = true });
                if (c.w.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.x.value, c.out.grad, c.w.grad, .{ .trans_a = true, .accumulate = true });
                try t.accumulateGrad(c.b, c.out.grad);
            }
        }.run);
        return out;
    }

    /// Applies max(x, 0) element by element.
    pub fn relu(self: *Tape, x: anytype) Error!Var(@TypeOf(x.value)) {
        const value = try self.newTensor(@TypeOf(x.value));
        if (self.par.gpu) |dev| {
            try dev.relu(x.value.data, value.data);
        } else {
            if (comptime @TypeOf(x.value).Elem != f32) return error.NoDevice;
            kernels.parallelFor(self.par, value.data.len, 1 << 15, PairJob{ .dst = value.data, .src = x.value.data }, reluRange);
        }
        const out = try self.makeVar(value, x.needs_grad);
        try self.note(.relu, .{x}, out, .{});
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.reluBackward(c.x.value.data, c.out.grad.data, c.x.grad.data);
                if (comptime @TypeOf(c.x.value).Elem != f32) return error.NoDevice;
                kernels.parallelFor(t.par, c.x.grad.data.len, 1 << 15, ReluGradJob{ .dst = c.x.grad.data, .x = c.x.value.data, .g = c.out.grad.data }, reluGradRange);
            }
        }.run);
        return out;
    }

    /// Multiplies (batch, m, k) by (batch, k, n), or by (batch, n, k) when trans_b is set (attention scores).
    pub fn batchedMatmul(self: *Tape, a: anytype, b: anytype, comptime trans_b: bool) Error!Var(Tensor(
        @TypeOf(a.value).Elem,
        &.{ @TypeOf(a.value).shape[0], @TypeOf(a.value).shape[1], if (trans_b) @TypeOf(b.value).shape[1] else @TypeOf(b.value).shape[2] },
    )) {
        const A = @TypeOf(a.value);
        const B = @TypeOf(b.value);
        comptime if (A.rank != 3 or B.rank != 3 or A.Elem != B.Elem or A.shape[0] != B.shape[0] or
            A.shape[2] != (if (trans_b) B.shape[2] else B.shape[1]))
            @compileError("batchedMatmul expects (batch, m, k) and (batch, k, n) tensors, or (batch, n, k) with trans_b, of one element type");
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
        try batchedInto(self.gpa, self.par, batch, Am, Bm, Cm, a.value.data, b.value.data, value.data, .{ .trans_b = trans_b });
        const out = try self.makeVar(value, a.needs_grad or b.needs_grad);
        try self.note(.batched_matmul, .{ a, b }, out, .{ .trans_b = trans_b });
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (c.a.needs_grad) try batchedInto(t.gpa, t.par, batch, Cm, Bm, Am, c.out.grad.data, c.b.value.data, c.a.grad.data, .{ .trans_b = !trans_b, .accumulate = true });
                if (c.b.needs_grad) {
                    if (trans_b) {
                        try batchedInto(t.gpa, t.par, batch, Cm, Am, Bm, c.out.grad.data, c.a.value.data, c.b.grad.data, .{ .trans_a = true, .accumulate = true });
                    } else {
                        try batchedInto(t.gpa, t.par, batch, Am, Cm, Bm, c.a.value.data, c.out.grad.data, c.b.grad.data, .{ .trans_a = true, .accumulate = true });
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
        try self.note(.embedding, .{ table, trace_mod.View.ofIds(ids) }, out, .{});
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
        try self.note(.layer_norm, .{ x, gamma, beta }, out, .{});
        const Ctx = struct { x: @TypeOf(x), gamma: @TypeOf(gamma), beta: @TypeOf(beta), mean: Stats, inv_std: Stats, out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .gamma = gamma, .beta = beta, .mean = mean, .inv_std = inv_std, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| {
                    const p: gpu_mod.Gpu.LayerNorm(X.Elem) = .{ .x = c.x.value.data, .gamma = c.gamma.value.data, .mean = c.mean.data, .inv_std = c.inv_std.data };
                    return dev.layerNormBackward(p, c.out.grad.data, if (c.x.needs_grad) c.x.grad.data else null, if (c.gamma.needs_grad) c.gamma.grad.data else null, if (c.gamma.needs_grad) c.beta.grad.data else null);
                }
                if (comptime X.Elem != f32) return error.NoDevice;
                for (0..rows) |r| {
                    if (c.x.needs_grad) layernorm_math.backwardRow(
                        f32,
                        c.x.value.data.ptr + r * cols,
                        c.out.grad.data.ptr + r * cols,
                        c.x.grad.data.ptr + r * cols,
                        c.gamma.value.data.ptr,
                        .{ .mean = c.mean.data[r], .inv_std = c.inv_std.data[r] },
                        cols,
                    );
                }
                if (c.gamma.needs_grad) for (0..cols) |j| layernorm_math.parameterGradient(
                    f32,
                    c.x.value.data.ptr,
                    c.out.grad.data.ptr,
                    c.mean.data.ptr,
                    c.inv_std.data.ptr,
                    c.gamma.grad.data.ptr,
                    c.beta.grad.data.ptr,
                    rows,
                    cols,
                    j,
                );
            }
        }.run);
        return out;
    }

    /// Applies softmax over the last axis to scale * x (attention folds its 1 / sqrt(depth) in here instead of a separate
    /// multiply). With causal_period above zero, row i only sees columns up to i % causal_period.
    pub fn softmax(self: *Tape, x: anytype, comptime opts: SoftmaxOpts) Error!Var(@TypeOf(x.value)) {
        const X = @TypeOf(x.value);
        comptime if (X.rank != 2) @compileError("softmax expects a (rows, columns) tensor");
        comptime if (!(opts.scale > 0)) @compileError("softmax needs a scale above zero");
        const cols = X.shape[1];
        const period = opts.causal_period;
        const value = try self.newTensor(X);
        if (self.par.gpu) |dev| {
            try dev.softmax(x.value.data, value.data, cols, period, opts.scale);
        } else {
            if (comptime X.Elem != f32) return error.NoDevice;
            const Job = struct { x: [*]const f32, y: [*]f32 };
            kernels.parallelFor(self.par, X.shape[0], 16, Job{ .x = x.value.data.ptr, .y = value.data.ptr }, struct {
                fn run(job: Job, lo: usize, hi: usize) void {
                    for (lo..hi) |r| softmax_math.forwardRow(f32, job.x + r * cols, job.y + r * cols, cols, softmax_math.limitOf(r, cols, period), opts.scale);
                }
            }.run);
        }
        const out = try self.makeVar(value, x.needs_grad);
        try self.note(.softmax, .{x}, out, .{ .period = period, .scale = opts.scale });
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.softmaxBackward(c.out.value.data, c.out.grad.data, c.x.grad.data, cols, opts.scale);
                if (comptime X.Elem != f32) return error.NoDevice;
                for (0..X.shape[0]) |r| softmax_math.backwardRow(f32, c.out.value.data.ptr + r * cols, c.out.grad.data.ptr + r * cols, c.x.grad.data.ptr + r * cols, cols, opts.scale);
            }
        }.run);
        return out;
    }

    fn gruChecks(comptime Gx: type, comptime Other: type, comptime H: type, comptime other_columns: usize) void {
        comptime {
            if (H.rank != 2 or H.Elem != f32 or Gx.Elem != f32 or Other.Elem != f32) @compileError("the GRU ops work on f32 matrices");
            if (Gx.shape[0] != H.shape[0] or Gx.shape[1] != 3 * H.shape[1] or
                Other.shape[0] != H.shape[0] or Other.shape[1] != other_columns * H.shape[1])
                @compileError("GRU shapes: gx is (rows, 3 * hidden), the other projection (rows, columns * hidden), the state (rows, hidden)");
        }
    }

    /// First half of a GRU step. gx is the input projection (rows, 3 * hidden: update | reset | candidate), gh the recurrent
    /// projection (rows, 2 * hidden: update | reset) and h the state (rows, hidden). Returns the update gate and the reset gate times h.
    /// NOTE: one kernel replaces the six slices, additions and activations of the same step; backward recomputes the gates.
    pub fn gruGates(self: *Tape, gx: anytype, gh: anytype, h: anytype) Error!struct { Var(@TypeOf(h.value)), Var(@TypeOf(h.value)) } {
        const H = @TypeOf(h.value);
        gruChecks(@TypeOf(gx.value), @TypeOf(gh.value), H, 2);
        const hidden = H.shape[1];
        const update_value = try self.newTensor(H);
        const reset_value = try self.newTensor(H);
        if (self.par.gpu) |dev| {
            try dev.gruGates(gx.value.data, gh.value.data, h.value.data, update_value.data, reset_value.data, hidden);
        } else {
            for (0..H.len) |i| gru_math.gatesForward(gx.value.data.ptr, gh.value.data.ptr, h.value.data.ptr, update_value.data.ptr, reset_value.data.ptr, i, hidden);
        }
        const needs = gx.needs_grad or gh.needs_grad or h.needs_grad;
        const update = try self.makeVar(update_value, needs);
        const reset_h = try self.makeVar(reset_value, needs);
        try self.note(.gru_update, .{ gx, gh }, update, .{});
        try self.note(.gru_reset_h, .{ gx, gh, h }, reset_h, .{});
        const Ctx = struct { gx: @TypeOf(gx), gh: @TypeOf(gh), h: @TypeOf(h), update: @TypeOf(update), reset_h: @TypeOf(reset_h) };
        if (needs) try self.recordBackward(Ctx{ .gx = gx, .gh = gh, .h = h, .update = update, .reset_h = reset_h }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                const flags: u32 = (if (c.gx.needs_grad) gru_math.needs_gx else 0) | (if (c.gh.needs_grad) gru_math.needs_gh else 0) | (if (c.h.needs_grad) gru_math.needs_h else 0);
                const grad: gru_math.GatesGrad = .{ .gx = c.gx.grad.data.ptr, .gh = c.gh.grad.data.ptr, .h = c.h.grad.data.ptr };
                if (t.par.gpu) |dev| return dev.gruGatesBackward(c.gx.value.data, c.gh.value.data, c.h.value.data, c.update.grad.data, c.reset_h.grad.data, grad, flags, hidden);
                for (0..H.len) |i| gru_math.gatesBackward(c.gx.value.data.ptr, c.gh.value.data.ptr, c.h.value.data.ptr, c.update.grad.data.ptr, c.reset_h.grad.data.ptr, grad, flags, i, hidden);
            }
        }.run);
        return .{ update, reset_h };
    }

    /// Second half of a GRU step: the new state c + update * (h - c), where c = tanh(candidate part of gx + gc) and gc is the
    /// candidate projection of the reset state (rows, hidden).
    pub fn gruNext(self: *Tape, gx: anytype, gc: anytype, update: anytype, h: anytype) Error!Var(@TypeOf(h.value)) {
        const H = @TypeOf(h.value);
        gruChecks(@TypeOf(gx.value), @TypeOf(gc.value), H, 1);
        comptime if (@TypeOf(update.value) != H) @compileError("the update gate must have the shape of the state");
        const hidden = H.shape[1];
        const value = try self.newTensor(H);
        if (self.par.gpu) |dev| {
            try dev.gruNext(gx.value.data, gc.value.data, update.value.data, h.value.data, value.data, hidden);
        } else {
            for (0..H.len) |i| gru_math.nextForward(gx.value.data.ptr, gc.value.data.ptr, update.value.data.ptr, h.value.data.ptr, value.data.ptr, i, hidden);
        }
        const out = try self.makeVar(value, gx.needs_grad or gc.needs_grad or update.needs_grad or h.needs_grad);
        try self.note(.gru_next, .{ gx, gc, update, h }, out, .{});
        const Ctx = struct { gx: @TypeOf(gx), gc: @TypeOf(gc), update: @TypeOf(update), h: @TypeOf(h), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .gx = gx, .gc = gc, .update = update, .h = h, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                const flags: u32 = (if (c.gx.needs_grad) gru_math.needs_gx else 0) |
                    (if (c.gc.needs_grad) gru_math.needs_gc else 0) |
                    (if (c.update.needs_grad) gru_math.needs_update else 0) |
                    (if (c.h.needs_grad) gru_math.needs_h else 0);
                const grad: gru_math.NextGrad = .{ .gx = c.gx.grad.data.ptr, .gc = c.gc.grad.data.ptr, .update = c.update.grad.data.ptr, .h = c.h.grad.data.ptr };
                if (t.par.gpu) |dev| return dev.gruNextBackward(c.gx.value.data, c.gc.value.data, c.update.value.data, c.h.value.data, c.out.grad.data, grad, flags, hidden);
                for (0..H.len) |i| gru_math.nextBackward(c.gx.value.data.ptr, c.gc.value.data.ptr, c.update.value.data.ptr, c.h.value.data.ptr, c.out.grad.data.ptr, grad, flags, i, hidden);
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
        return self.batchNormImpl(.none, x, {}, gamma, beta, running);
    }

    /// Batch norm followed by ReLU, fused into one kernel.
    pub fn batchNormRelu(self: *Tape, x: anytype, gamma: anytype, beta: anytype, running: anytype) Error!Var(@TypeOf(x.value)) {
        return self.batchNormImpl(.relu, x, {}, gamma, beta, running);
    }

    /// Computes max(batchNorm(x) + residual, 0), the end of a residual block, in one pass over the activations.
    /// residual must have the shape and element type of x.
    pub fn batchNormAddRelu(self: *Tape, x: anytype, residual: anytype, gamma: anytype, beta: anytype, running: anytype) Error!Var(@TypeOf(x.value)) {
        comptime if (@TypeOf(x.value) != @TypeOf(residual.value)) @compileError("batchNormAddRelu expects a residual of the same shape as x");
        return self.batchNormImpl(.add_relu, x, residual, gamma, beta, running);
    }

    fn batchNormImpl(self: *Tape, comptime fuse: NormFuse, x: anytype, residual: anytype, gamma: anytype, beta: anytype, running: anytype) Error!Var(@TypeOf(x.value)) {
        const In = @TypeOf(x.value);
        const c = In.shape[In.rank - 1];
        const Channels = Tensor(f32, &.{c});
        const mean = try self.newTensor(Channels);
        const inv_std = try self.newTensor(Channels);
        const value = try self.newTensor(In);
        const st: norm.Stats = .{ .mean = mean.data.ptr, .inv_std = inv_std.data.ptr, .running_mean = running.mean.data.ptr, .running_var = running.variance.data.ptr };
        const affine: norm.Affine = .{ .mean = st.mean, .inv_std = st.inv_std, .gamma = gamma.value.data.ptr, .beta = beta.value.data.ptr };
        const flag: u32 = switch (fuse) {
            .none => 0,
            .relu => norm.relu_flag,
            .add_relu => norm.relu_flag | norm.residual_flag,
        };
        const layout: norm.Layout = .{ .len = In.len, .channels = c, .flag = flag };
        const added = if (fuse == .add_relu) residual.value.data else x.value.data;
        if (self.par.gpu) |dev| {
            try dev.batchNormStats(x.value.data, st, c, self.training);
            try dev.batchNormApply(x.value.data, added, value.data, affine, layout);
        } else if (In.Elem == f32) {
            cpu.normStats(self.par, x.value.data, st, c, self.training);
            cpu.normApply(self.par, x.value.data, added, value.data, affine, layout);
        } else return error.NoDevice;
        const residual_grad = fuse == .add_relu and residual.needs_grad;
        const out = try self.makeVar(value, x.needs_grad or gamma.needs_grad or beta.needs_grad or residual_grad);
        const norm_inputs = .{ x, gamma, beta, running.mean, running.variance };
        switch (fuse) {
            .none => try self.note(.batch_norm, norm_inputs, out, .{}),
            .relu => try self.note(.batch_norm_relu, norm_inputs, out, .{}),
            .add_relu => try self.note(.batch_norm_add_relu, norm_inputs ++ .{residual}, out, .{}),
        }

        const Ctx = struct { x: @TypeOf(x), residual: @TypeOf(residual), gamma: @TypeOf(gamma), beta: @TypeOf(beta), affine: norm.Affine, out: @TypeOf(out) };
        if (out.needs_grad and self.training) try self.recordBackward(Ctx{ .x = x, .residual = residual, .gamma = gamma, .beta = beta, .affine = affine, .out = out }, struct {
            fn run(t: *Tape, k: *const Ctx) Error!void {
                const sums = try t.zeroedTensor(Tensor(f32, &.{2 * c}));
                const mask: []const In.Elem = if (fuse != .none) k.out.value.data else &.{};
                const grads: norm.GradOut = .{ .sums = sums.data, .gamma = k.gamma.grad.data, .beta = k.beta.grad.data };
                const to_residual = fuse == .add_relu and k.residual.needs_grad;
                var grad_layout = layout;
                if (!to_residual) grad_layout.flag &= ~norm.residual_flag;
                const io = batchNormGradIo(In.Elem, k, sums.data, fuse);
                if (t.par.gpu) |dev| {
                    try dev.batchNormGradSums(k.x.value.data, k.out.grad.data, k.affine, mask, grads);
                    if (k.x.needs_grad) return dev.batchNormGradInput(io, k.affine, grad_layout);
                    // NOTE: without an input gradient the residual still needs the masked output gradient.
                    if (to_residual) try dev.addReluBackward(k.out.value.data, k.out.grad.data, k.residual.grad.data, k.residual.grad.data, true, false);
                    return;
                }
                if (comptime In.Elem != f32) return error.NoDevice;
                cpu.normGradSums(t.par, k.x.value.data, k.out.grad.data, k.affine, mask, grads);
                if (k.x.needs_grad) return cpu.normGradInput(t.par, io, k.affine, grad_layout);
                if (!to_residual) return;
                for (k.out.value.data, k.out.grad.data, k.residual.grad.data) |o, g, *d| d.* += if (o > 0) g else 0;
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
            .maxPool => return cpu.maxPool(self.par, slices[0], slices[1], slices[2], d),
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
        comptime if (opts.stride != 1 or opts.groups != 1 or opts.dilation != 1 or
            c_in % 64 != 0 or c_out % 32 != 0)
            @compileError("bf16 activations need stride-1 convolutions with a multiple of 64 input and of 32 output channels; cast to f32 around other layers");
        const dev = self.par.gpu orelse return error.NoDevice;
        const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = k, .pad = opts.pad, .c_out = c_out };
        const kc = k * k * c_in;
        const geometry: ga.ConvGeometry = comptime convGeometry(d, .{});
        const reverse: ga.ConvGeometry = comptime convGeometry(d, .{ .reverse = true });

        const value = try self.newTensor(Out);
        {
            const scope = dev.scratchScope();
            defer scope.restore();
            const w_half = try dev.packMatrix(w.value.data, kc, c_out, .{ .cols_padded = gpu_mod.padTo(c_out, .columns) });
            try dev.convImplicitForward(x.value.data, w_half, value.data, geometry, false);
        }
        const out = try self.makeVar(value, x.needs_grad or w.needs_grad);
        try self.note(.conv, .{ x, w }, out, .{ .kernel = k, .pad = opts.pad });
        const Ctx = struct { x: @TypeOf(x), w: @TypeOf(w), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .w = w, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                const g = t.par.gpu.?;
                const scope = g.scratchScope();
                defer scope.restore();
                if (c.x.needs_grad) {
                    const rotated = try g.packRotatedWeights(c.w.value.data, k, c_in, c_out);
                    try g.convImplicitForward(c.out.grad.data, rotated, c.x.grad.data, reverse, true);
                }
                if (c.w.needs_grad) try g.convImplicitWeightGrad(c.out.grad.data, c.x.value.data, c.w.grad.data, geometry);
            }
        }.run);
        return out;
    }

    fn Conv2dOps(comptime In: type, comptime W: type, comptime opts: ConvOpts) type {
        return struct {
            const Out = ConvOut(In, W, opts);
            const k = kernelSize(In, W, opts.groups);
            const c_in = In.shape[3];
            const c_out = Out.shape[3];
            const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = k, .stride = opts.stride, .pad = opts.pad, .dilation = opts.dilation, .groups = opts.groups, .c_out = c_out };
            const rows = Out.len / c_out;
            const kc = k * k * c_in;
            const Col = Tensor(f32, &.{ rows, kc });
            const Flat = Tensor(f32, &.{ rows, c_out });
            const geometry: ga.ConvGeometry = convGeometry(d, .{});
            const reverse: ga.ConvGeometry = convGeometry(d, .{ .reverse = true });
            const cudnn_geometry: gpu_mod.cudnn.Geometry = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = c_out, .r = k, .pad = opts.pad, .stride = opts.stride };
            const aligned_out = gpu_mod.padTo(c_out, .columns);
            const general = opts.dilation * (k - 1) < opts.pad;
            const per_in = c_in / opts.groups;
            const per_out = c_out / opts.groups;
            const dense_only = opts.groups > 1 or general;

            const Cache = struct { col: []f32 = &.{}, col_half: Packed = .{ .data = &.{}, .rows_padded = 0, .cols_padded = 0 }, x_half: []const u16 = &.{} };
            const Ctx = struct { x: Var(In), w: Var(W), path: ConvPath, cache: Cache, out: Var(Out) };

            fn forward(tape: *Tape, x: Var(In), w: Var(W), value: Out, path: ConvPath) Error!Cache {
                const dev = tape.par.gpu;
                var cache: Cache = .{};
                switch (path) {
                    .grouped => try tape.convKernel(.groupedForward, .{ x.value.data, w.value.data, value.data }, d),
                    // NOTE: the im2col paths are typed and shaped for dense weights and do not compile for grouped shapes; the
                    // comptime condition below prunes them, so the dense prongs live in the nested switch.
                    .cpu, .direct, .packed_im2col, .implicit, .f32_implicit, .cudnn => if (dense_only) unreachable else switch (path) {
                        .cpu => {
                            // NOTE: the CPU builds the im2col matrix in cache-sized chunks instead of one buffer of positions x k*k*c;
                            // the weight gradient rebuilds it the same way.
                            try cpu.convForward(tape.gpa, tape.par, x.value.data, w.value.data, value.data, d);
                        },
                        .direct => {
                            const g = dev.?;
                            const columns: Col = .{ .data = try g.scratch(f32, Col.len) };
                            try tape.convKernel(.im2col, .{ x.value.data, columns.data }, d);
                            try tensor.matmulInto(tape.gpa, tape.par, columns, w.value, Flat{ .data = value.data }, .{});
                            cache.col = columns.data;
                        },
                        .packed_im2col => {
                            const g = dev.?;
                            const columns = try g.im2colPacked(x.value.data, d);
                            const w_half = try g.packMatrix(w.value.data, kc, c_out, .{ .rows_padded = columns.cols_padded, .cols_padded = aligned_out });
                            try g.matmulPacked(columns, w_half, value.data, .{ .m = rows, .n = c_out, .k = kc }, false, false);
                            cache.col_half = columns;
                        },
                        .implicit => {
                            const g = dev.?;
                            cache.x_half = (try g.packMatrix(x.value.data, In.len / c_in, c_in, .{})).data;
                            const w_half = try g.packMatrix(w.value.data, kc, c_out, .{ .cols_padded = aligned_out });
                            try g.convImplicitForward(cache.x_half, w_half, value.data, geometry, false);
                        },
                        .f32_implicit => try dev.?.convF32Forward(x.value.data, w.value.data, value.data, geometry, .{}),
                        .cudnn => {
                            const g = dev.?;
                            const scope = g.scratchScope();
                            defer scope.restore();
                            try g.cudnnForward(cudnn_geometry, x.value.data, try g.weightsToKrsc(w.value.data, k, c_in, c_out), value.data);
                        },
                        .grouped, .grouped_implicit => unreachable,
                    },
                    .grouped_implicit => {
                        const g = dev.?;
                        for (0..opts.groups) |grp| {
                            const geo_g = comptime convGeometry(d, .{ .per_group = true });
                            try g.convF32Forward(x.value.data[grp * per_in ..], w.value.data[grp * per_out ..], value.data, geo_g, .{ .w_pitch = c_out, .c_layout = .{ .pitch = c_out, .offset = @intCast(grp * per_out) } });
                        }
                    },
                }
                return cache;
            }

            fn backward(t: *Tape, c: *const Ctx) Error!void {
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
                    .cpu, .direct, .packed_im2col, .implicit, .f32_implicit, .cudnn => if (dense_only) unreachable else switch (c.path) {
                        .cpu => {
                            if (c.x.needs_grad) try cpu.convInputGrad(t.gpa, t.par, grad.data, c.w.value.data, c.x.grad.data, d);
                            if (c.w.needs_grad) try cpu.convWeightGrad(t.gpa, t.par, c.x.value.data, grad.data, c.w.grad.data, d);
                        },
                        .direct => {
                            if (c.x.needs_grad) {
                                const dcol = try t.newTensor(Col);
                                try tensor.matmulInto(t.gpa, t.par, grad, c.w.value, dcol, .{ .trans_b = true });
                                try t.convKernel(.col2im, .{ dcol.data, c.x.grad.data }, d);
                            }
                            if (c.w.needs_grad) {
                                const columns: Col = .{ .data = c.cache.col };
                                try tensor.matmulInto(t.gpa, t.par, columns, grad, c.w.grad, .{ .trans_a = true, .accumulate = true });
                            }
                        },
                        .packed_im2col => {
                            const g = t.par.gpu.?;
                            if (c.x.needs_grad) {
                                const grad_packed = try g.packMatrix(grad.data, rows, c_out, .{ .rows_padded = gpu_mod.padTo(rows, .rows), .cols_padded = gpu_mod.padTo(c_out, .depth) });
                                const weight_packed = try g.packMatrix(c.w.value.data, c_out, kc, .{ .rows_padded = gpu_mod.padTo(c_out, .depth), .cols_padded = gpu_mod.padTo(kc, .columns), .transpose = true });
                                const dcol_half = try g.scratch(u16, rows * kc);
                                try g.matmulPacked(grad_packed, weight_packed, dcol_half, .{ .m = rows, .n = kc, .k = c_out }, false, false);
                                try t.convKernel(.col2imHalf, .{ dcol_half, c.x.grad.data }, d);
                            }
                            if (c.w.needs_grad) {
                                const columns = c.cache.col_half;
                                const grad_t = try g.packMatrix(grad.data, c_out, rows, .{ .rows_padded = gpu_mod.padTo(c_out, .rows), .cols_padded = columns.rows_padded, .transpose = true });
                                try g.matmulPacked(grad_t, columns, c.w.grad.data, .{ .m = c_out, .n = kc, .k = rows }, true, true);
                            }
                        },
                        .implicit => {
                            const g = t.par.gpu.?;
                            // NOTE: one bf16 copy of the output gradient serves both products.
                            const grad_half = try g.packMatrix(grad.data, rows, c_out, .{});
                            if (c.x.needs_grad) {
                                const rotated = try g.packRotatedWeights(c.w.value.data, k, c_in, c_out);
                                try g.convImplicitForward(grad_half.data, rotated, c.x.grad.data, reverse, true);
                            }
                            if (c.w.needs_grad) try g.convImplicitWeightGrad(grad_half.data, c.cache.x_half, c.w.grad.data, geometry);
                        },
                        .f32_implicit => {
                            const g = t.par.gpu.?;
                            if (c.x.needs_grad) {
                                const rotated = try g.rotateWeightsF32(c.w.value.data, k, c_in, c_out, null);
                                try g.convF32Forward(grad.data, rotated, c.x.grad.data, reverse, .{ .w_pitch = gpu_mod.padTo(c_in, .columns), .accumulate = true });
                            }
                            if (c.w.needs_grad) {
                                const positions_padded = gpu_mod.padTo(rows, .depth);
                                const grad_t = try g.transposeF32(grad.data, c_out, rows, gpu_mod.padTo(c_out, .depth), positions_padded, null);
                                try g.convF32WeightGrad(grad_t, c.x.value.data, c.w.grad.data, geometry, null);
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
                        .grouped, .grouped_implicit => unreachable,
                    },
                    .grouped_implicit => {
                        const g = t.par.gpu.?;
                        for (0..opts.groups) |grp| {
                            const geo_g = comptime convGeometry(d, .{ .per_group = true });
                            const reverse_g = comptime convGeometry(d, .{ .reverse = true, .per_group = true });
                            if (c.x.needs_grad) {
                                const rotated = try g.rotateWeightsF32(c.w.value.data, k, per_in, per_out, .{ .pitch = c_out, .offset = @intCast(grp * per_out) });
                                try g.convF32Forward(grad.data[grp * per_out ..], rotated, c.x.grad.data, reverse_g, .{ .w_pitch = gpu_mod.padTo(per_in, .columns), .accumulate = true, .c_layout = .{ .pitch = c_in, .offset = @intCast(grp * per_in) } });
                            }
                            if (c.w.needs_grad) {
                                const positions_padded = gpu_mod.padTo(rows, .depth);
                                const grad_t = try g.transposeF32(grad.data, per_out, rows, gpu_mod.padTo(per_out, .depth), positions_padded, .{ .pitch = c_out, .offset = @intCast(grp * per_out) });
                                try g.convF32WeightGrad(grad_t, c.x.value.data[grp * per_in ..], c.w.grad.data, geo_g, .{ .pitch = c_out, .offset = @intCast(grp * per_out) });
                            }
                        }
                    },
                }
            }
        };
    }

    /// Convolves an NHWC input with weights of shape (k*k*c/groups, out).
    /// NOTE: the algorithm (direct, packed bf16, implicit GEMM, TF32, cuDNN, grouped) is chosen once per call from the shapes and device flags.
    pub fn conv2d(self: *Tape, x: anytype, w: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts)) {
        const In = @TypeOf(x.value);
        if (comptime In.Elem == u16) return self.convHalf(x, w, opts);
        const Ops = Conv2dOps(In, @TypeOf(w.value), opts);
        // NOTE: the gathers space the taps by dilation and the reverse geometry pads by dilation * (k - 1) - pad, so a
        // dilated convolution can take the implicit GEMM; it cannot take cuDNN (no dilation in its descriptor) or any
        // path when the reverse pad would be negative.
        const path = chooseConvPath(Ops.d, self.par.gpu);
        const value = try self.newTensor(Ops.Out);
        // NOTE: the column matrix is k*k times the input and is kept for the weight gradient.
        const cache = try Ops.forward(self, x, w, value, path);
        const out = try self.makeVar(value, x.needs_grad or w.needs_grad);
        try self.note(.conv, .{ x, w }, out, .{ .kernel = Ops.k, .stride = opts.stride, .pad = opts.pad, .dilation = opts.dilation, .groups = opts.groups });

        const Ctx = Ops.Ctx;
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .w = w, .path = path, .cache = cache, .out = out }, Ops.backward);
        return out;
    }
    /// Convolution followed by batch normalization; on the bf16 GPU training path the conv GEMM's epilogue
    /// accumulates the batch statistics, saving a pass over the feature maps.
    pub fn convBatchNorm(self: *Tape, x: anytype, w: anytype, gamma: anytype, beta: anytype, running: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts)) {
        return self.convBn(.none, x, w, {}, gamma, beta, running, opts);
    }

    /// Convolution followed by batch normalization and ReLU; see convBatchNorm.
    pub fn convBatchNormRelu(self: *Tape, x: anytype, w: anytype, gamma: anytype, beta: anytype, running: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts)) {
        return self.convBn(.relu, x, w, {}, gamma, beta, running, opts);
    }

    /// Convolution, batch norm, addition of residual and ReLU; see convBatchNorm. residual must have the
    /// shape and element type of the convolution's output.
    pub fn convBatchNormAddRelu(self: *Tape, x: anytype, w: anytype, residual: anytype, gamma: anytype, beta: anytype, running: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts)) {
        const Out = ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts);
        comptime if (@TypeOf(residual.value) != Out) @compileError("convBatchNormAddRelu expects a residual of the convolution's output shape");
        return self.convBn(.add_relu, x, w, residual, gamma, beta, running, opts);
    }

    fn convBn(self: *Tape, comptime fuse: NormFuse, x: anytype, w: anytype, residual: anytype, gamma: anytype, beta: anytype, running: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(@TypeOf(x.value), @TypeOf(w.value), opts)) {
        const In = @TypeOf(x.value);
        const Out = ConvOut(In, @TypeOf(w.value), opts);
        if (comptime In.Elem != u16) {
            const y = try self.conv2d(x, w, opts);
            return switch (fuse) {
                .none => self.batchNorm(y, gamma, beta, running),
                .relu => self.batchNormRelu(y, gamma, beta, running),
                .add_relu => self.batchNormAddRelu(y, residual, gamma, beta, running),
            };
        }
        // NOTE: only the bf16 implicit GEMM accumulates the statistics in its epilogue; evaluation uses the
        // running statistics and needs no sums, so both take the unfused ops.
        if (self.par.gpu == null or !self.training) {
            const y = try self.conv2d(x, w, opts);
            return switch (fuse) {
                .none => self.batchNorm(y, gamma, beta, running),
                .relu => self.batchNormRelu(y, gamma, beta, running),
                .add_relu => self.batchNormAddRelu(y, residual, gamma, beta, running),
            };
        }
        const dev = self.par.gpu.?;
        const ks = comptime kernelSize(In, @TypeOf(w.value), 1);
        const c_in = In.shape[3];
        const c_out = Out.shape[3];
        comptime if (opts.stride != 1 or opts.groups != 1 or opts.dilation != 1 or
            c_in % 64 != 0 or c_out % 32 != 0)
            @compileError("bf16 activations need stride-1 convolutions with a multiple of 64 input and of 32 output channels; cast to f32 around other layers");
        const d: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = ks, .pad = opts.pad, .c_out = c_out };
        const kc = ks * ks * c_in;
        const geometry: ga.ConvGeometry = comptime convGeometry(d, .{});
        const reverse: ga.ConvGeometry = comptime convGeometry(d, .{ .reverse = true });
        const Channels = Tensor(f32, &.{c_out});
        const mean = try self.newTensor(Channels);
        const inv_std = try self.newTensor(Channels);
        const st: norm.Stats = .{ .mean = mean.data.ptr, .inv_std = inv_std.data.ptr, .running_mean = running.mean.data.ptr, .running_var = running.variance.data.ptr };
        const affine: norm.Affine = .{ .mean = st.mean, .inv_std = st.inv_std, .gamma = gamma.value.data.ptr, .beta = beta.value.data.ptr };
        const flag: u32 = switch (fuse) {
            .none => 0,
            .relu => norm.relu_flag,
            .add_relu => norm.relu_flag | norm.residual_flag,
        };
        const layout: norm.Layout = .{ .len = Out.len, .channels = c_out, .flag = flag };
        const conv_value = try self.newTensor(Out);
        {
            const scope = dev.scratchScope();
            defer scope.restore();
            const w_half = try dev.packMatrix(w.value.data, kc, c_out, .{ .cols_padded = gpu_mod.padTo(c_out, .columns) });
            const sums = try dev.scratchZeroed(f32, 2 * c_out);
            // NOTE: the sums are shifted by the running mean, so the finish kernel gets it as the f32 shift.
            try dev.convImplicitForwardStats(x.value.data, w_half, conv_value.data, geometry, running.mean.data, sums);
            try dev.batchNormFinishShifted(sums, running.mean.data, st, .{ .len = Out.len, .channels = c_out, .flag = 1 });
        }
        const value = try self.newTensor(Out);
        try dev.batchNormApply(conv_value.data, if (fuse == .add_relu) residual.value.data else conv_value.data, value.data, affine, layout);
        const residual_grad = fuse == .add_relu and residual.needs_grad;
        const out = try self.makeVar(value, x.needs_grad or w.needs_grad or gamma.needs_grad or beta.needs_grad or residual_grad);
        const norm_inputs = .{ gamma, beta, running.mean, running.variance };
        try self.note(.conv, .{ x, w }, conv_value, .{ .kernel = ks, .pad = opts.pad });
        switch (fuse) {
            .none => try self.note(.batch_norm, .{conv_value} ++ norm_inputs, out, .{}),
            .relu => try self.note(.batch_norm_relu, .{conv_value} ++ norm_inputs, out, .{}),
            .add_relu => try self.note(.batch_norm_add_relu, .{conv_value} ++ norm_inputs ++ .{residual}, out, .{}),
        }

        const Ctx = struct { x: @TypeOf(x), w: @TypeOf(w), residual: @TypeOf(residual), gamma: @TypeOf(gamma), beta: @TypeOf(beta), affine: norm.Affine, conv_value: Out, out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .w = w, .residual = residual, .gamma = gamma, .beta = beta, .affine = affine, .conv_value = conv_value, .out = out }, struct {
            fn run(t: *Tape, k: *const Ctx) Error!void {
                const g = t.par.gpu.?;
                const sums = try t.zeroedTensor(Tensor(f32, &.{2 * c_out}));
                const mask: []const u16 = if (fuse != .none) k.out.value.data else &.{};
                const grads: norm.GradOut = .{ .sums = sums.data, .gamma = k.gamma.grad.data, .beta = k.beta.grad.data };
                try g.batchNormGradSums(k.conv_value.data, k.out.grad.data, k.affine, mask, grads);
                const to_residual = fuse == .add_relu and k.residual.needs_grad;
                var grad_layout = layout;
                if (!to_residual) grad_layout.flag &= ~norm.residual_flag;
                grad_layout.flag |= norm.store_flag;
                if (k.x.needs_grad or k.w.needs_grad) {
                    const scope = g.scratchScope();
                    defer scope.restore();
                    // NOTE: the batch-norm input gradient is the conv output gradient; it is a fresh scratch
                    // buffer written once per element, so the kernel stores instead of accumulating.
                    const dconv: Out = .{ .data = try g.scratch(u16, Out.len) };
                    const io: norm.GradIo(u16) = .{
                        .x = k.conv_value.data.ptr,
                        .dy = k.out.grad.data.ptr,
                        .dx = dconv.data.ptr,
                        .sums = sums.data.ptr,
                        .mask = if (fuse != .none) k.out.value.data.ptr else k.out.grad.data.ptr,
                        .dr = if (to_residual) k.residual.grad.data.ptr else dconv.data.ptr,
                    };
                    try g.batchNormGradInput(io, k.affine, grad_layout);
                    if (k.x.needs_grad) {
                        const rotated = try g.packRotatedWeights(k.w.value.data, ks, c_in, c_out);
                        try g.convImplicitForward(dconv.data, rotated, k.x.grad.data, reverse, true);
                    }
                    if (k.w.needs_grad) try g.convImplicitWeightGrad(dconv.data, k.x.value.data, k.w.grad.data, geometry);
                } else if (to_residual) {
                    try g.addReluBackward(k.out.value.data, k.out.grad.data, k.residual.grad.data, k.residual.grad.data, true, false);
                }
            }
        }.run);
        return out;
    }

    /// Two convolutions of a residual block fused around a virtual middle map: the batch-norm-plus-ReLU output of
    /// the first convolution is never written to memory. The second convolution's GEMM rebuilds it on load
    /// (relu(scale * x + shift) per element, with scale and shift rounded to bf16 — one extra bf16 step next to
    /// the map's own bf16 storage), and in the backward pass the input-gradient GEMM of the second convolution
    /// accumulates the first norm's masked gradient sums in its epilogue. Both convolutions share
    /// opts. Off the bf16 GPU training path it falls back to the unfused pair.
    pub fn convBnChain(self: *Tape, x: anytype, w1: anytype, gamma1: anytype, beta1: anytype, running1: anytype, w2: anytype, residual: anytype, gamma2: anytype, beta2: anytype, running2: anytype, comptime opts: ConvOpts) Error!Var(ConvOut(ConvOut(@TypeOf(x.value), @TypeOf(w1.value), opts), @TypeOf(w2.value), opts)) {
        const In = @TypeOf(x.value);
        const Mid = ConvOut(In, @TypeOf(w1.value), opts);
        const Out = ConvOut(Mid, @TypeOf(w2.value), opts);
        comptime if (@TypeOf(residual.value) != Out) @compileError("convBnChain expects a residual of the chain's output shape");
        if (comptime In.Elem != u16) {
            const y1 = try self.convBatchNormRelu(x, w1, gamma1, beta1, running1, opts);
            return self.convBatchNormAddRelu(y1, w2, residual, gamma2, beta2, running2, opts);
        }
        // NOTE: only the bf16 implicit GEMM rebuilds the virtual map on load; evaluation uses the running
        // statistics and both convolutions take the unfused ops, like convBn.
        if (self.par.gpu == null or !self.training) {
            const y1 = try self.convBatchNormRelu(x, w1, gamma1, beta1, running1, opts);
            return self.convBatchNormAddRelu(y1, w2, residual, gamma2, beta2, running2, opts);
        }
        const dev = self.par.gpu.?;
        const ks1 = comptime kernelSize(In, @TypeOf(w1.value), 1);
        const ks2 = comptime kernelSize(Mid, @TypeOf(w2.value), 1);
        const c_in = In.shape[3];
        const c1 = Mid.shape[3];
        const c2 = Out.shape[3];
        comptime if (opts.stride != 1 or opts.groups != 1 or opts.dilation != 1 or
            c_in % 64 != 0 or c1 % 64 != 0 or c2 % 32 != 0)
            @compileError("bf16 activations need stride-1 convolutions with a multiple of 64 input and of 32 output channels; cast to f32 around other layers");
        const d1: conv.Dims = .{ .n = In.shape[0], .h = In.shape[1], .w = In.shape[2], .c = c_in, .k = ks1, .pad = opts.pad, .c_out = c1 };
        const d2: conv.Dims = .{ .n = In.shape[0], .h = Mid.shape[1], .w = Mid.shape[2], .c = c1, .k = ks2, .pad = opts.pad, .c_out = c2 };
        const kc1 = ks1 * ks1 * c_in;
        const kc2 = ks2 * ks2 * c1;
        const geometry1: ga.ConvGeometry = comptime convGeometry(d1, .{});
        const reverse1: ga.ConvGeometry = comptime convGeometry(d1, .{ .reverse = true });
        const geometry2: ga.ConvGeometry = comptime convGeometry(d2, .{});
        const reverse2: ga.ConvGeometry = comptime convGeometry(d2, .{ .reverse = true });
        const Channels1 = Tensor(f32, &.{c1});
        const Channels2 = Tensor(f32, &.{c2});
        const mean1 = try self.newTensor(Channels1);
        const inv_std1 = try self.newTensor(Channels1);
        const mean2 = try self.newTensor(Channels2);
        const inv_std2 = try self.newTensor(Channels2);
        const st1: norm.Stats = .{ .mean = mean1.data.ptr, .inv_std = inv_std1.data.ptr, .running_mean = running1.mean.data.ptr, .running_var = running1.variance.data.ptr };
        const st2: norm.Stats = .{ .mean = mean2.data.ptr, .inv_std = inv_std2.data.ptr, .running_mean = running2.mean.data.ptr, .running_var = running2.variance.data.ptr };
        const affine1: norm.Affine = .{ .mean = st1.mean, .inv_std = st1.inv_std, .gamma = gamma1.value.data.ptr, .beta = beta1.value.data.ptr };
        const affine2: norm.Affine = .{ .mean = st2.mean, .inv_std = st2.inv_std, .gamma = gamma2.value.data.ptr, .beta = beta2.value.data.ptr };
        const layout2: norm.Layout = .{ .len = Out.len, .channels = c2, .flag = norm.relu_flag | norm.residual_flag };
        const conv1_out = try self.newTensor(Mid);
        const conv2_out = try self.newTensor(Out);
        // NOTE: the virtual map is relu(scale * x + shift) per element with these expanded parameters; the same
        // expansion serves the second convolution's loads and its weight gradient in the backward pass, so the
        // buffer is a tape tensor rather than forward scratch.
        const aff_buf = try self.newTensor(Tensor(u32, &.{3 * gpu_mod.padTo(kc2, .rows)}));
        var aff: gpu_mod.Gpu.AffineExpanded = undefined;
        {
            const scope = dev.scratchScope();
            defer scope.restore();
            const w1_half = try dev.packMatrix(w1.value.data, kc1, c1, .{ .cols_padded = gpu_mod.padTo(c1, .columns) });
            const sums1 = try dev.scratchZeroed(f32, 2 * c1);
            try dev.convImplicitForwardStats(x.value.data, w1_half, conv1_out.data, geometry1, running1.mean.data, sums1);
            try dev.batchNormFinishShifted(sums1, running1.mean.data, st1, .{ .len = Mid.len, .channels = c1, .flag = 1 });
            aff = try dev.affineExpand(mean1.data, inv_std1.data, gamma1.value.data, beta1.value.data, aff_buf.data, c1, kc2);
            const w2_half = try dev.packMatrix(w2.value.data, kc2, c2, .{ .cols_padded = gpu_mod.padTo(c2, .columns) });
            const sums2 = try dev.scratchZeroed(f32, 2 * c2);
            try dev.convImplicitForwardAffineStats(conv1_out.data, w2_half, conv2_out.data, geometry2, aff, running2.mean.data, sums2);
            try dev.batchNormFinishShifted(sums2, running2.mean.data, st2, .{ .len = Out.len, .channels = c2, .flag = 1 });
        }
        const value = try self.newTensor(Out);
        try dev.batchNormApply(conv2_out.data, residual.value.data, value.data, affine2, layout2);
        const needs = x.needs_grad or w1.needs_grad or w2.needs_grad or gamma1.needs_grad or beta1.needs_grad or gamma2.needs_grad or beta2.needs_grad or residual.needs_grad;
        const out = try self.makeVar(value, needs);
        // NOTE: the first norm's output is virtual (never stored), so the trace records the chain's spine only.
        try self.note(.conv, .{ x, w1 }, conv1_out, .{ .kernel = ks1, .pad = opts.pad });
        try self.note(.conv, .{ conv1_out, w2 }, conv2_out, .{ .kernel = ks2, .pad = opts.pad });
        try self.note(.batch_norm_add_relu, .{ conv2_out, gamma2, beta2, running2.mean, running2.variance, residual }, out, .{});

        const Ctx = struct { x: @TypeOf(x), w1: @TypeOf(w1), gamma1: @TypeOf(gamma1), beta1: @TypeOf(beta1), affine1: norm.Affine, conv1_out: Mid, aff: gpu_mod.Gpu.AffineExpanded, w2: @TypeOf(w2), gamma2: @TypeOf(gamma2), beta2: @TypeOf(beta2), affine2: norm.Affine, conv2_out: Out, residual: @TypeOf(residual), out: @TypeOf(out) };
        if (out.needs_grad) try self.recordBackward(Ctx{ .x = x, .w1 = w1, .gamma1 = gamma1, .beta1 = beta1, .affine1 = affine1, .conv1_out = conv1_out, .aff = aff, .w2 = w2, .gamma2 = gamma2, .beta2 = beta2, .affine2 = affine2, .conv2_out = conv2_out, .residual = residual, .out = out }, struct {
            fn run(t: *Tape, k: *const Ctx) Error!void {
                const g = t.par.gpu.?;
                const sums2 = try t.zeroedTensor(Tensor(f32, &.{2 * c2}));
                const grads2: norm.GradOut = .{ .sums = sums2.data, .gamma = k.gamma2.grad.data, .beta = k.beta2.grad.data };
                try g.batchNormGradSums(k.conv2_out.data, k.out.grad.data, k.affine2, k.out.value.data, grads2);
                const to_residual = k.residual.needs_grad;
                const inner = k.x.needs_grad or k.w1.needs_grad or k.w2.needs_grad or k.gamma1.needs_grad or k.beta1.needs_grad;
                if (!inner) {
                    // NOTE: without an input gradient the residual still needs the masked output gradient.
                    if (to_residual) try g.addReluBackward(k.out.value.data, k.out.grad.data, k.residual.grad.data, k.residual.grad.data, true, false);
                    return;
                }
                var grad_layout2 = layout2;
                if (!to_residual) grad_layout2.flag &= ~norm.residual_flag;
                grad_layout2.flag |= norm.store_flag;
                const scope = g.scratchScope();
                defer scope.restore();
                // NOTE: the batch-norm input gradients are the conv output gradients; both are fresh scratch
                // buffers written once per element, so the kernels store instead of accumulating.
                const dconv2: Out = .{ .data = try g.scratch(u16, Out.len) };
                const io2: norm.GradIo(u16) = .{
                    .x = k.conv2_out.data.ptr,
                    .dy = k.out.grad.data.ptr,
                    .dx = dconv2.data.ptr,
                    .sums = sums2.data.ptr,
                    .mask = k.out.value.data.ptr,
                    .dr = if (to_residual) k.residual.grad.data.ptr else dconv2.data.ptr,
                };
                try g.batchNormGradInput(io2, k.affine2, grad_layout2);
                // NOTE: the input gradient of the second convolution is the output gradient of the virtual first
                // norm, so its GEMM's epilogue accumulates that norm's masked gradient sums straight from the
                // values it stores, saving the column reduction over the map.
                const dy1: Mid = .{ .data = try g.scratch(u16, Mid.len) };
                const sums1 = try t.zeroedTensor(Tensor(f32, &.{2 * c1}));
                {
                    const rotated2 = try g.packRotatedWeights(k.w2.value.data, ks2, c1, c2);
                    try g.convImplicitForwardDxStats(dconv2.data, rotated2, dy1.data, reverse2, k.conv1_out.data, k.affine1, sums1.data);
                }
                try g.batchNormGradFinish(sums1.data, k.gamma1.grad.data, k.beta1.grad.data);
                if (k.w2.needs_grad) try g.convImplicitWeightGradAffine(dconv2.data, k.conv1_out.data, k.w2.grad.data, geometry2, k.aff);
                const dconv1: Mid = .{ .data = try g.scratch(u16, Mid.len) };
                const io1: norm.GradIo(u16) = .{
                    .x = k.conv1_out.data.ptr,
                    .dy = dy1.data.ptr,
                    .dx = dconv1.data.ptr,
                    .sums = sums1.data.ptr,
                    .mask = dy1.data.ptr,
                    .dr = dconv1.data.ptr,
                };
                const layout1: norm.Layout = .{ .len = Mid.len, .channels = c1, .flag = norm.relu_flag | norm.store_flag | norm.mask_x_flag };
                try g.batchNormGradInput(io1, k.affine1, layout1);
                if (k.x.needs_grad) {
                    const rotated1 = try g.packRotatedWeights(k.w1.value.data, ks1, c_in, c1);
                    try g.convImplicitForward(dconv1.data, rotated1, k.x.grad.data, reverse1, true);
                }
                if (k.w1.needs_grad) try g.convImplicitWeightGrad(dconv1.data, k.x.value.data, k.w1.grad.data, geometry1);
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
