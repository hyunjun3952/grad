const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const autograd = @import("../autograd/autograd.zig");
const optim = @import("optim.zig");
const Gpu = @import("../gpu/gpu.zig").Gpu;
const Var = autograd.Var;
const Tape = autograd.Tape;

// ***** layers *****
//
// A layer is a struct with init, deinit, toGpu, forward, params and state. params lists the
// trainable variables for the optimizer; state lists every float slice a checkpoint must store.

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
            const w = try W.empty(gpa);
            errdefer w.deinit(gpa);
            const std_dev = @sqrt(2.0 / @as(f32, n_in));
            for (w.data) |*v| v.* = rng.floatNorm(f32) * std_dev;
            const grad_w = try W.zeros(gpa);
            errdefer grad_w.deinit(gpa);
            const b = try B.zeros(gpa);
            errdefer b.deinit(gpa);
            return .{
                .w = .{ .value = w, .grad = grad_w, .needs_grad = true },
                .b = .{ .value = b, .grad = try B.zeros(gpa), .needs_grad = true },
            };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.w.value.deinit(gpa);
            self.w.grad.deinit(gpa);
            self.b.value.deinit(gpa);
            self.b.grad.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            const w = try gpu.alloc(f32, W.len);
            const grad_w = try gpu.alloc(f32, W.len);
            const b = try gpu.alloc(f32, B.len);
            const grad_b = try gpu.alloc(f32, B.len);
            try gpu.upload(w, self.w.value.data);
            try gpu.upload(b, self.b.value.data);
            try gpu.fill(grad_w, 0);
            try gpu.fill(grad_b, 0);
            return .{
                .w = .{ .value = .{ .data = w }, .grad = .{ .data = grad_w }, .needs_grad = true },
                .b = .{ .value = .{ .data = b }, .grad = .{ .data = grad_b }, .needs_grad = true },
            };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(Tensor(@TypeOf(x.value).Elem, &.{ @TypeOf(x.value).shape[0], n_out })) {
            // NOTE: for bf16 activations the f32 weights are rounded inside the product and the bias is rounded here.
            return tape.add(try tape.matmul(x, self.w), try tape.cast(self.b, @TypeOf(x.value).Elem));
        }

        pub fn params(self: Self) struct { Var(W), Var(B) } {
            return .{ self.w, self.b };
        }

        pub fn state(self: Self) struct { []f32, []f32 } {
            return .{ self.w.value.data, self.b.value.data };
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
            const gamma = try V.full(gpa, 1);
            errdefer gamma.deinit(gpa);
            const grad_gamma = try V.zeros(gpa);
            errdefer grad_gamma.deinit(gpa);
            const beta = try V.zeros(gpa);
            errdefer beta.deinit(gpa);
            const grad_beta = try V.zeros(gpa);
            errdefer grad_beta.deinit(gpa);
            const mean = try V.zeros(gpa);
            errdefer mean.deinit(gpa);
            return .{
                .gamma = .{ .value = gamma, .grad = grad_gamma, .needs_grad = true },
                .beta = .{ .value = beta, .grad = grad_beta, .needs_grad = true },
                .mean = mean,
                .variance = try V.full(gpa, 1),
            };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.gamma.value.deinit(gpa);
            self.gamma.grad.deinit(gpa);
            self.beta.value.deinit(gpa);
            self.beta.grad.deinit(gpa);
            self.mean.deinit(gpa);
            self.variance.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            var out: Self = undefined;
            inline for (.{ "gamma", "beta" }) |name| {
                const value = try gpu.alloc(f32, c);
                const grad = try gpu.alloc(f32, c);
                try gpu.upload(value, @field(self, name).value.data);
                try gpu.fill(grad, 0);
                @field(out, name) = .{ .value = .{ .data = value }, .grad = .{ .data = grad }, .needs_grad = true };
            }
            inline for (.{ "mean", "variance" }) |name| {
                const statistic = try gpu.alloc(f32, c);
                try gpu.upload(statistic, @field(self, name).data);
                @field(out, name) = .{ .data = statistic };
            }
            return out;
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            return tape.batchNorm(x, self.gamma, self.beta, .{ .mean = self.mean, .variance = self.variance });
        }

        pub fn forwardRelu(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            return tape.batchNormRelu(x, self.gamma, self.beta, .{ .mean = self.mean, .variance = self.variance });
        }

        pub fn params(self: Self) struct { Var(V), Var(V) } {
            return .{ self.gamma, self.beta };
        }

        pub fn state(self: Self) struct { []f32, []f32, []f32, []f32 } {
            return .{ self.gamma.value.data, self.beta.value.data, self.mean.data, self.variance.data };
        }
    };
}

/// A bias-free convolution of an NHWC input with a k x k kernel; opts sets stride, padding, dilation and groups.
pub fn Conv2d(comptime c_in: usize, comptime c_out: usize, comptime k: usize, comptime opts: autograd.ConvOpts) type {
    const Inner = Linear(k * k * c_in / opts.groups, c_out);

    return struct {
        const Self = @This();

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

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(Tensor(@TypeOf(x.value).Elem, &.{ @TypeOf(x.value).shape[0], (@TypeOf(x.value).shape[1] + 2 * opts.pad - k) / opts.stride + 1, (@TypeOf(x.value).shape[2] + 2 * opts.pad - k) / opts.stride + 1, c_out })) {
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

/// A convolution followed by batch normalization, the usual building block of the residual networks.
pub fn ConvBn(comptime c_in: usize, comptime c_out: usize, comptime k: usize, comptime opts: autograd.ConvOpts) type {
    const Conv = Conv2d(c_in, c_out, k, opts);
    const Norm = BatchNorm(c_out);

    return struct {
        const Self = @This();

        conv: Conv,
        norm: Norm,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            const conv = try Conv.init(gpa, rng);
            errdefer conv.deinit(gpa);
            return .{ .conv = conv, .norm = try Norm.init(gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.conv.deinit(gpa);
            self.norm.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            return .{ .conv = try self.conv.toGpu(gpu), .norm = try self.norm.toGpu(gpu) };
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(Tensor(@TypeOf(x.value).Elem, &.{ @TypeOf(x.value).shape[0], (@TypeOf(x.value).shape[1] + 2 * opts.pad - k) / opts.stride + 1, (@TypeOf(x.value).shape[2] + 2 * opts.pad - k) / opts.stride + 1, c_out })) {
            return self.norm.forward(tape, try tape.conv2d(x, self.conv.inner.w, opts));
        }

        pub fn forwardRelu(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(Tensor(@TypeOf(x.value).Elem, &.{ @TypeOf(x.value).shape[0], (@TypeOf(x.value).shape[1] + 2 * opts.pad - k) / opts.stride + 1, (@TypeOf(x.value).shape[2] + 2 * opts.pad - k) / opts.stride + 1, c_out })) {
            return self.norm.forwardRelu(tape, try tape.conv2d(x, self.conv.inner.w, opts));
        }

        pub fn params(self: Self) @TypeOf(.{self.conv.inner.w} ++ self.norm.params()) {
            return .{self.conv.inner.w} ++ self.norm.params();
        }

        pub fn state(self: Self) @TypeOf(.{self.conv.inner.w.value.data} ++ self.norm.state()) {
            return .{self.conv.inner.w.value.data} ++ self.norm.state();
        }
    };
}

/// A gated recurrent unit. forward takes the input and the previous hidden state and returns the next hidden state.
pub fn Gru(comptime n_in: usize, comptime hidden: usize) type {
    const Input = Linear(n_in, hidden);
    const Recurrent = Linear(hidden, hidden);

    return struct {
        const Self = @This();

        update_x: Input,
        update_h: Recurrent,
        reset_x: Input,
        reset_h: Recurrent,
        candidate_x: Input,
        candidate_h: Recurrent,

        pub fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Self {
            const update_x = try Input.init(gpa, rng);
            errdefer update_x.deinit(gpa);
            const update_h = try Recurrent.init(gpa, rng);
            errdefer update_h.deinit(gpa);
            const reset_x = try Input.init(gpa, rng);
            errdefer reset_x.deinit(gpa);
            const reset_h = try Recurrent.init(gpa, rng);
            errdefer reset_h.deinit(gpa);
            const candidate_x = try Input.init(gpa, rng);
            errdefer candidate_x.deinit(gpa);
            return .{ .update_x = update_x, .update_h = update_h, .reset_x = reset_x, .reset_h = reset_h, .candidate_x = candidate_x, .candidate_h = try Recurrent.init(gpa, rng) };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            inline for (@typeInfo(Self).@"struct".field_names) |name| @field(self, name).deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            var out: Self = undefined;
            inline for (@typeInfo(Self).@"struct".field_names) |name| @field(out, name) = try @field(self, name).toGpu(gpu);
            return out;
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype, h: anytype) autograd.Error!@TypeOf(h) {
            const update = try tape.activate(try tape.add(try self.update_x.forward(tape, x), try self.update_h.forward(tape, h)), .sigmoid);
            const reset = try tape.activate(try tape.add(try self.reset_x.forward(tape, x), try self.reset_h.forward(tape, h)), .sigmoid);
            const candidate = try tape.activate(try tape.add(try self.candidate_x.forward(tape, x), try self.candidate_h.forward(tape, try tape.mul(reset, h))), .tanh);
            const minus_one = try tape.constant(try tape.filledTensor(Tensor(f32, &.{}), -1));
            return tape.add(candidate, try tape.mul(update, try tape.add(h, try tape.mul(candidate, minus_one))));
        }

        pub fn params(self: Self) @TypeOf(self.update_x.params() ++ self.update_h.params() ++ self.reset_x.params() ++ self.reset_h.params() ++ self.candidate_x.params() ++ self.candidate_h.params()) {
            return self.update_x.params() ++ self.update_h.params() ++ self.reset_x.params() ++ self.reset_h.params() ++ self.candidate_x.params() ++ self.candidate_h.params();
        }

        pub fn state(self: Self) @TypeOf(self.update_x.state() ++ self.update_h.state() ++ self.reset_x.state() ++ self.reset_h.state() ++ self.candidate_x.state() ++ self.candidate_h.state()) {
            return self.update_x.state() ++ self.update_h.state() ++ self.reset_x.state() ++ self.reset_h.state() ++ self.candidate_x.state() ++ self.candidate_h.state();
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
            const gamma = try V.full(gpa, 1);
            errdefer gamma.deinit(gpa);
            const grad_gamma = try V.zeros(gpa);
            errdefer grad_gamma.deinit(gpa);
            const beta = try V.zeros(gpa);
            errdefer beta.deinit(gpa);
            return .{
                .gamma = .{ .value = gamma, .grad = grad_gamma, .needs_grad = true },
                .beta = .{ .value = beta, .grad = try V.zeros(gpa), .needs_grad = true },
            };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.gamma.value.deinit(gpa);
            self.gamma.grad.deinit(gpa);
            self.beta.value.deinit(gpa);
            self.beta.grad.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            var out: Self = undefined;
            inline for (.{ "gamma", "beta" }) |name| {
                const value = try gpu.alloc(f32, c);
                const grad = try gpu.alloc(f32, c);
                try gpu.upload(value, @field(self, name).value.data);
                try gpu.fill(grad, 0);
                @field(out, name) = .{ .value = .{ .data = value }, .grad = .{ .data = grad }, .needs_grad = true };
            }
            return out;
        }

        pub fn forward(self: Self, tape: *Tape, x: anytype) autograd.Error!Var(@TypeOf(x.value)) {
            return tape.layerNorm(x, self.gamma, self.beta);
        }

        pub fn params(self: Self) struct { Var(V), Var(V) } {
            return .{ self.gamma, self.beta };
        }

        pub fn state(self: Self) struct { []f32, []f32 } {
            return .{ self.gamma.value.data, self.beta.value.data };
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
            const table = try W.empty(gpa);
            errdefer table.deinit(gpa);
            for (table.data) |*v| v.* = rng.floatNorm(f32) * 0.1;
            return .{ .table = .{ .value = table, .grad = try W.zeros(gpa), .needs_grad = true } };
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            self.table.value.deinit(gpa);
            self.table.grad.deinit(gpa);
        }

        pub fn toGpu(self: Self, gpu: *Gpu) autograd.Error!Self {
            const value = try gpu.alloc(f32, W.len);
            const grad = try gpu.alloc(f32, W.len);
            try gpu.upload(value, self.table.value.data);
            try gpu.fill(grad, 0);
            return .{ .table = .{ .value = .{ .data = value }, .grad = .{ .data = grad }, .needs_grad = true } };
        }

        pub fn forward(self: Self, tape: *Tape, comptime n: usize, ids: []const u32, comptime Out: type) autograd.Error!Var(Tensor(Out, &.{ n, width })) {
            return tape.embedding(self.table, n, ids, Out);
        }

        pub fn params(self: Self) struct { Var(W) } {
            return .{self.table};
        }

        pub fn state(self: Self) struct { []f32 } {
            return .{self.table.value.data};
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
            const scale = try tape.cast(try tape.constant(try tape.filledTensor(Tensor(f32, &.{}), 1 / @sqrt(@as(f32, depth)))), @TypeOf(x.value).Elem);
            const query = autograd.reshape(try self.layers[0].forward(tape, x), &.{ batch, steps, depth });
            const key = autograd.reshape(try self.layers[1].forward(tape, x), &.{ batch, steps, depth });
            const value = autograd.reshape(try self.layers[2].forward(tape, x), &.{ batch, steps, depth });
            const scores = try tape.mul(autograd.reshape(try tape.batchedMatmul(query, key, true), &.{ rows, steps }), scale);
            const weights = autograd.reshape(try tape.softmax(scores, steps), &.{ batch, steps, steps });
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
