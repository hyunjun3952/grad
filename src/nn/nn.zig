const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const autograd = @import("../autograd/autograd.zig");
const Gpu = @import("../gpu/gpu.zig").Gpu;
const Var = autograd.Var;
const Tape = autograd.Tape;

// ***** layers *****
//
// A layer is a struct with init, deinit, toGpu, forward, params and state. params lists the
// trainable variables for the optimizer; state lists every float slice a checkpoint must store.

// Trainable variables share their life cycle: a value, a gradient that starts at zero, and a copy on the device.
fn newParam(comptime Tn: type, gpa: Allocator, value: Tn) Allocator.Error!Var(Tn) {
    errdefer value.deinit(gpa);
    return .{ .value = value, .grad = try Tn.zeros(gpa), .needs_grad = true };
}

fn freeParam(param: anytype, gpa: Allocator) void {
    param.value.deinit(gpa);
    param.grad.deinit(gpa);
}

fn paramToGpu(param: anytype, gpu: *Gpu) autograd.Error!@TypeOf(param) {
    const value = try gpu.alloc(f32, param.value.data.len);
    const grad = try gpu.alloc(f32, param.grad.data.len);
    try gpu.upload(value, param.value.data);
    try gpu.fill(grad, 0);
    return .{ .value = .{ .data = value }, .grad = .{ .data = grad }, .needs_grad = true };
}

// Statistics that are stored but not trained, such as the running mean of batch norm.
fn statisticToGpu(statistic: anytype, gpu: *Gpu) autograd.Error!@TypeOf(statistic) {
    const copy = try gpu.alloc(f32, statistic.data.len);
    try gpu.upload(copy, statistic.data);
    return .{ .data = copy };
}

fn deinitFields(self: anytype, gpa: Allocator) void {
    inline for (@typeInfo(@TypeOf(self)).@"struct".field_names) |field| {
        const value = @field(self, field);
        if (comptime @hasDecl(@TypeOf(value), "deinit")) value.deinit(gpa) else freeParam(value, gpa);
    }
}

fn toGpuFields(self: anytype, gpu: *Gpu) autograd.Error!@TypeOf(self) {
    const Self = @TypeOf(self);
    var out: Self = undefined;
    inline for (@typeInfo(Self).@"struct".field_names) |field| {
        const value = @field(self, field);
        const Field = @TypeOf(value);
        @field(out, field) = if (comptime @hasDecl(Field, "toGpu"))
            try value.toGpu(gpu)
        else if (comptime @hasField(Field, "grad"))
            try paramToGpu(value, gpu)
        else
            try statisticToGpu(value, gpu);
    }
    return out;
}

fn DirectParams(comptime Self: type) type {
    comptime var types: []const type = &.{};
    inline for (@typeInfo(Self).@"struct".field_types) |Field| {
        if (@hasField(Field, "grad")) types = types ++ &[1]type{Field};
    }
    return @Tuple(types);
}

fn directParams(self: anytype) DirectParams(@TypeOf(self)) {
    var out: DirectParams(@TypeOf(self)) = undefined;
    comptime var at = 0;
    inline for (@typeInfo(@TypeOf(self)).@"struct".field_names) |field| {
        const value = @field(self, field);
        if (comptime @hasField(@TypeOf(value), "grad")) {
            out[at] = value;
            at += 1;
        }
    }
    return out;
}

fn DirectState(comptime Self: type) type {
    comptime var types: []const type = &.{};
    inline for (@typeInfo(Self).@"struct".field_names) |_| types = types ++ &[1]type{[]f32};
    return @Tuple(types);
}

fn directState(self: anytype) DirectState(@TypeOf(self)) {
    var out: DirectState(@TypeOf(self)) = undefined;
    inline for (@typeInfo(@TypeOf(self)).@"struct".field_names, 0..) |field, i| {
        const value = @field(self, field);
        out[i] = if (comptime @hasField(@TypeOf(value), "grad")) value.value.data else value.data;
    }
    return out;
}

// He-normal weights for a layer whose output sums fan_in products.
fn heNormal(comptime W: type, gpa: Allocator, rng: std.Random, comptime fan_in: usize) Allocator.Error!W {
    const values = try W.empty(gpa);
    const std_dev = @sqrt(2.0 / @as(f32, fan_in));
    for (values.data) |*v| v.* = rng.floatNorm(f32) * std_dev;
    return values;
}

/// A fully connected layer, y = x * w + b, with He-normal weights and zero bias.
/// x has shape (rows, n_in) and the result has shape (rows, n_out).
pub fn Linear(comptime n_in: usize, comptime n_out: usize) type {
    const W = Tensor(f32, &.{ n_in, n_out });
    const B = Tensor(f32, &.{n_out});

    return struct {
        const Self = @This();

        w: Var(W),
        b: Var(B),

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            const w = try newParam(W, gpa, try heNormal(W, gpa, rng, n_in));
            errdefer freeParam(w, gpa);
            return .{ .w = w, .b = try newParam(B, gpa, try B.zeros(gpa)) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitFields(self, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return toGpuFields(self, gpu);
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(Tensor(@TypeOf(x.value).Elem, &.{ @TypeOf(x.value).shape[0], n_out })) {
            if (@TypeOf(x.value).Elem == f32) return tape.linear(x, self.w, self.b);
            // NOTE: for bf16 activations the f32 weights are rounded inside the product and the bias is rounded here.
            return tape.add(try tape.matmul(x, self.w), try tape.cast(self.b, @TypeOf(x.value).Elem));
        }

        pub fn params(self: Self) DirectParams(Self) {
            return directParams(self);
        }

        pub fn state(self: Self) DirectState(Self) {
            return directState(self);
        }
    };
}

/// Batch normalization over the last axis of c channels, with running statistics for evaluation.
/// forwardRelu fuses the following ReLU into the same kernel.
pub fn BatchNorm(comptime c: usize) type {
    const V = Tensor(f32, &.{c});

    return struct {
        const Self = @This();

        gamma: Var(V),
        beta: Var(V),
        mean: V,
        variance: V,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            _ = rng;
            const gamma = try newParam(V, gpa, try V.full(gpa, 1));
            errdefer freeParam(gamma, gpa);
            const beta = try newParam(V, gpa, try V.zeros(gpa));
            errdefer freeParam(beta, gpa);
            const mean = try V.zeros(gpa);
            errdefer mean.deinit(gpa);
            return .{ .gamma = gamma, .beta = beta, .mean = mean, .variance = try V.full(gpa, 1) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitFields(self, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return toGpuFields(self, gpu);
        }

        fn stats(self: Self) @TypeOf(.{ .mean = self.mean, .variance = self.variance }) {
            return .{ .mean = self.mean, .variance = self.variance };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            return tape.batchNorm(x, self.gamma, self.beta, self.stats());
        }

        pub fn forwardRelu(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            return tape.batchNormRelu(x, self.gamma, self.beta, self.stats());
        }

        /// Computes max(norm(x) + residual, 0), the end of a residual block, in one kernel.
        pub fn forwardAddRelu(self: Self, tape: *Tape, x: anytype, residual: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            return tape.batchNormAddRelu(x, residual, self.gamma, self.beta, self.stats());
        }

        pub fn params(self: Self) DirectParams(Self) {
            return directParams(self);
        }

        pub fn state(self: Self) DirectState(Self) {
            return directState(self);
        }
    };
}

/// A convolution of an NHWC input with a k x k kernel plus a per-channel bias; opts sets stride, padding, dilation and groups.
pub fn Conv2d(comptime c_in: usize, comptime c_out: usize, comptime k: usize, comptime opts: autograd.ConvOpts) type {
    const Inner = Linear(k * k * c_in / opts.groups, c_out);

    return struct {
        const Self = @This();
        pub const Weights = Tensor(f32, &.{ k * k * c_in / opts.groups, c_out });

        inner: Inner,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            return .{ .inner = try Inner.init(gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.inner.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return .{ .inner = try self.inner.toGpu(gpu) };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(autograd.ConvOut(@TypeOf(x.value), Weights, opts)) {
            return tape.add(try tape.conv2d(x, self.inner.w, opts), self.inner.b);
        }

        pub fn params(self: Self) @TypeOf(self.inner.params()) {
            return self.inner.params();
        }

        pub fn state(self: Self) @TypeOf(self.inner.state()) {
            return self.inner.state();
        }
    };
}

/// A bias-free convolution followed by batch normalization, the usual building block of the residual networks.
/// NOTE: the norm's shift replaces the bias, so this layer owns the weights only.
pub fn ConvBn(comptime c_in: usize, comptime c_out: usize, comptime k: usize, comptime opts: autograd.ConvOpts) type {
    const W = Tensor(f32, &.{ k * k * c_in / opts.groups, c_out });
    const Norm = BatchNorm(c_out);

    return struct {
        const Self = @This();

        w: Var(W),
        norm: Norm,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            const w = try newParam(W, gpa, try heNormal(W, gpa, rng, k * k * c_in / opts.groups));
            errdefer freeParam(w, gpa);
            return .{ .w = w, .norm = try Norm.init(gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitFields(self, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return toGpuFields(self, gpu);
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(autograd.ConvOut(@TypeOf(x.value), W, opts)) {
            return tape.convBatchNorm(x, self.w, self.norm.gamma, self.norm.beta, self.norm.stats(), opts);
        }

        pub fn forwardRelu(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(autograd.ConvOut(@TypeOf(x.value), W, opts)) {
            return tape.convBatchNormRelu(x, self.w, self.norm.gamma, self.norm.beta, self.norm.stats(), opts);
        }

        /// Convolution, batch norm, addition of residual and ReLU; the last three run as one kernel, and on the
        /// bf16 GPU path the batch statistics come from the convolution's epilogue.
        pub fn forwardAddRelu(self: Self, tape: *Tape, x: anytype, residual: anytype) autograd.Error!Var(autograd.ConvOut(@TypeOf(x.value), W, opts)) {
            return tape.convBatchNormAddRelu(x, self.w, residual, self.norm.gamma, self.norm.beta, self.norm.stats(), opts);
        }

        /// A residual block of this convolution fused with a preceding one around a virtual middle map: the
        /// first convolution's batch norm and ReLU are never written to memory. first must use the same opts.
        pub fn forwardChain(self: Self, tape: *Tape, first: anytype, x: anytype, residual: anytype) autograd.Error!Var(autograd.ConvOut(autograd.ConvOut(@TypeOf(x.value), @TypeOf(first.w.value), opts), W, opts)) {
            return tape.convBnChain(x, first.w, first.norm.gamma, first.norm.beta, first.norm.stats(), self.w, residual, self.norm.gamma, self.norm.beta, self.norm.stats(), opts);
        }

        pub fn params(self: Self) @TypeOf(.{self.w} ++ self.norm.params()) {
            return .{self.w} ++ self.norm.params();
        }

        pub fn state(self: Self) @TypeOf(.{self.w.value.data} ++ self.norm.state()) {
            return .{self.w.value.data} ++ self.norm.state();
        }
    };
}

/// A gated recurrent unit. forward takes the input and the previous hidden state and returns the next hidden state.
/// The projections are fused: one product maps the input to the update, reset and candidate parts (3 * hidden columns), one maps the
/// state to the update and reset parts, and one maps the reset state to the candidate; two kernels do the gate arithmetic.
pub fn Gru(comptime n_in: usize, comptime hidden: usize) type {
    return struct {
        const Self = @This();

        input: Linear(n_in, 3 * hidden),
        recurrent: Linear(hidden, 2 * hidden),
        candidate: Linear(hidden, hidden),

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            const input = try Linear(n_in, 3 * hidden).init(gpa, rng);
            errdefer input.deinit(gpa);
            const recurrent = try Linear(hidden, 2 * hidden).init(gpa, rng);
            errdefer recurrent.deinit(gpa);
            return .{ .input = input, .recurrent = recurrent, .candidate = try Linear(hidden, hidden).init(gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitFields(self, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return toGpuFields(self, gpu);
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype, h: anytype) autograd.Error!@TypeOf(h) {
            const gx = try self.input.forward(tape, x);
            const gh = try self.recurrent.forward(tape, h);
            const update, const reset_h = try tape.gruGates(gx, gh, h);
            return tape.gruNext(gx, try self.candidate.forward(tape, reset_h), update, h);
        }

        pub fn params(self: Self) @TypeOf(self.input.params() ++ self.recurrent.params() ++ self.candidate.params()) {
            return self.input.params() ++ self.recurrent.params() ++ self.candidate.params();
        }

        pub fn state(self: Self) @TypeOf(self.input.state() ++ self.recurrent.state() ++ self.candidate.state()) {
            return self.input.state() ++ self.recurrent.state() ++ self.candidate.state();
        }
    };
}
// ***** transformer *****

/// Layer normalization over the last axis of c features, with learned scale (starts at one) and shift (starts at zero).
pub fn LayerNorm(comptime c: usize) type {
    const V = Tensor(f32, &.{c});

    return struct {
        const Self = @This();

        gamma: Var(V),
        beta: Var(V),

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            _ = rng;
            const gamma = try newParam(V, gpa, try V.full(gpa, 1));
            errdefer freeParam(gamma, gpa);
            return .{ .gamma = gamma, .beta = try newParam(V, gpa, try V.zeros(gpa)) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitFields(self, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return toGpuFields(self, gpu);
        }
        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            return tape.layerNorm(x, self.gamma, self.beta);
        }

        pub fn params(self: Self) DirectParams(Self) {
            return directParams(self);
        }

        pub fn state(self: Self) DirectState(Self) {
            return directState(self);
        }
    };
}

/// A lookup table of vocab rows with width features each; forward gathers the rows named by ids.
pub fn Embedding(comptime vocab: usize, comptime width: usize) type {
    const W = Tensor(f32, &.{ vocab, width });

    return struct {
        const Self = @This();

        table: Var(W),

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            const values = try W.empty(gpa);
            for (values.data) |*v| v.* = rng.floatNorm(f32) * 0.1;
            return .{ .table = try newParam(W, gpa, values) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitFields(self, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return toGpuFields(self, gpu);
        }
        pub fn forward(self: Self, tape: *Tape, comptime n: usize, ids: []const u32, comptime Out: type) autograd.Error!Var(Tensor(Out, &.{ n, width })) {
            return tape.embedding(self.table, n, ids, Out);
        }

        pub fn params(self: Self) DirectParams(Self) {
            return directParams(self);
        }

        pub fn state(self: Self) DirectState(Self) {
            return directState(self);
        }
    };
}

// One causal attention head. Queries, keys and values are projected to depth features, the scores
// are softmaxed per sequence of steps rows, and the mixed values are projected back to width.
fn Head(comptime width: usize, comptime depth: usize, comptime steps: usize) type {
    const Layers = struct { Linear(width, depth), Linear(width, depth), Linear(width, depth), Linear(depth, width) };

    return struct {
        const Self = @This();

        layers: Layers,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            return .{ .layers = try initAll(Layers, gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitAll(self.layers, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return .{ .layers = try toGpuAll(self.layers, gpu) };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            const rows = @TypeOf(x.value).shape[0];
            const batch = rows / steps;
            const query = autograd.reshape(try self.layers[0].forward(tape, x), &.{ batch, steps, depth });
            const key = autograd.reshape(try self.layers[1].forward(tape, x), &.{ batch, steps, depth });
            const value = autograd.reshape(try self.layers[2].forward(tape, x), &.{ batch, steps, depth });
            const scores = autograd.reshape(try tape.batchedMatmul(query, key, true), &.{ rows, steps });
            const weights = autograd.reshape(try tape.softmax(scores, .{ .causal_period = steps, .scale = 1 / @sqrt(@as(f32, depth)) }), &.{ batch, steps, steps });
            const mixed = autograd.reshape(try tape.batchedMatmul(weights, value, false), &.{ rows, depth });
            return self.layers[3].forward(tape, mixed);
        }

        pub fn params(self: Self) Gathered(Layers, "params") {
            return params_(self.layers);
        }

        pub fn state(self: Self) Gathered(Layers, "state") {
            return state_(self.layers);
        }
    };
}

/// Causal multi-head self-attention over sequences of steps rows; the heads are separate layers whose outputs are summed.
/// x has shape (batch * steps, width) and heads must divide width.
pub fn Attention(comptime width: usize, comptime heads: usize, comptime steps: usize) type {
    const One = Head(width, width / heads, steps);
    const Group = blk: {
        var types: [heads]type = undefined;
        for (&types) |*slot| slot.* = One;
        break :blk @Tuple(&types);
    };

    return struct {
        const Self = @This();

        group: Group,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            return .{ .group = try initAll(Group, gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitAll(self.group, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return .{ .group = try toGpuAll(self.group, gpu) };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            var total = try self.group[0].forward(tape, x);
            inline for (1..heads) |i| total = try tape.add(total, try self.group[i].forward(tape, x));
            return total;
        }

        pub fn params(self: Self) Gathered(Group, "params") {
            return params_(self.group);
        }

        pub fn state(self: Self) Gathered(Group, "state") {
            return state_(self.group);
        }
    };
}

/// A pre-norm Transformer block: attention and a GELU feed-forward network (4 * width hidden), each with a residual connection.
pub fn TransformerBlock(comptime width: usize, comptime heads: usize, comptime steps: usize) type {
    const Layers = struct { LayerNorm(width), Attention(width, heads, steps), LayerNorm(width), Linear(width, 4 * width), Linear(4 * width, width) };

    return struct {
        const Self = @This();

        layers: Layers,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            return .{ .layers = try initAll(Layers, gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            deinitAll(self.layers, gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return .{ .layers = try toGpuAll(self.layers, gpu) };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            const norm_in, const attention, const norm_out, const expand, const contract = self.layers;
            const mixed = try tape.add(x, try attention.forward(tape, try norm_in.forward(tape, x)));
            const hidden = try tape.activate(try expand.forward(tape, try norm_out.forward(tape, mixed)), .gelu);
            return tape.add(mixed, try contract.forward(tape, hidden));
        }

        pub fn params(self: Self) Gathered(Layers, "params") {
            return params_(self.layers);
        }

        pub fn state(self: Self) Gathered(Layers, "state") {
            return state_(self.layers);
        }
    };
}
// ***** layer lists *****
//
// A model is a tuple of layers. The functions below walk the tuple at compile time and
// concatenate what each layer reports.

fn Gathered(comptime Layers: type, comptime method: []const u8) type {
    comptime var types: []const type = &.{};
    inline for (@typeInfo(Layers).@"struct".field_types) |Layer| {
        const Items = @TypeOf(@field(Layer, method)(@as(Layer, undefined)));
        types = types ++ @typeInfo(Items).@"struct".field_types;
    }
    return @Tuple(types);
}

fn gather(layers: anytype, comptime method: []const u8) Gathered(@TypeOf(layers), method) {
    var out: Gathered(@TypeOf(layers), method) = undefined;
    comptime var at: usize = 0;
    inline for (layers) |layer| {
        inline for (@field(@TypeOf(layer), method)(layer)) |item| {
            out[at] = item;
            at += 1;
        }
    }
    return out;
}

/// Builds every layer of a tuple type; layers already built are released if a later one fails.
pub fn initAll(comptime Layers: type, gpa: Allocator, rng: std.Random) Allocator.Error!Layers {
    var layers: Layers = undefined;
    inline for (@typeInfo(Layers).@"struct".field_types, 0..) |Layer, i| {
        layers[i] = Layer.init(gpa, rng) catch |err| {
            inline for (0..i) |built| layers[built].deinit(gpa);
            return err;
        };
    }
    return layers;
}

/// Releases every layer of a tuple.
pub fn deinitAll(layers: anytype, gpa: Allocator) void {
    inline for (layers) |layer| layer.deinit(gpa);
}

/// Copies every layer of a tuple to the device and returns the device copy.
pub fn toGpuAll(layers: anytype, gpu: *Gpu) autograd.Error!@TypeOf(layers) {
    var out: @TypeOf(layers) = undefined;
    inline for (layers, 0..) |layer, i| out[i] = try layer.toGpu(gpu);
    return out;
}

/// Returns the trainable variables of all layers as one flat tuple.
pub fn params(layers: anytype) Gathered(@TypeOf(layers), "params") {
    return gather(layers, "params");
}

const params_ = params;

/// Returns every float slice that a checkpoint must store (weights and batch norm statistics) as one flat tuple.
pub fn state(layers: anytype) Gathered(@TypeOf(layers), "state") {
    return gather(layers, "state");
}

const state_ = state;
