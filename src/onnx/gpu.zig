const std = @import("std");
const Allocator = std.mem.Allocator;

const kernels = @import("../cpu/kernels.zig");
const ga = @import("../gpu/args.zig");
const gpu_mod = @import("../gpu/gpu.zig");
const cv = @import("../gpu/math/conv.zig");
const runtime = @import("runtime.zig");

const Gpu = gpu_mod.Gpu;
const Model = runtime.Model;
const Node = runtime.Node;
const Array = runtime.Array;

pub const Error = runtime.Error || gpu_mod.Error;

const max_rank = ga.max_rank;

// ***** values *****

// A tensor of the graph. It lives on the host, on the device or on both; the missing side is filled in when an op needs it.
const Value = struct {
    dims: []const usize,
    host: ?[]f32 = null,
    dev: ?[]f32 = null,
    ints: []const i64 = &.{},
    // The initializer behind this value, whose device copy is kept between runs.
    constant: ?[]const u8 = null,

    fn len(self: Value) usize {
        return product(self.dims);
    }
};

const Values = std.StringHashMap(Value);

const product = runtime.product;

// ***** shapes and strides *****

// Broadcasts two shapes by the numpy rule, or returns null when they do not fit or the rank is beyond what the kernels take.
fn broadcastDims(a: Allocator, x: []const usize, y: []const usize) Error!?[]usize {
    const rank = @max(x.len, y.len);
    if (rank > max_rank) return null;
    const out = try a.alloc(usize, rank);
    for (out, 0..) |*d, i| {
        const from_end = rank - i;
        const xd = if (from_end <= x.len) x[x.len - from_end] else 1;
        const yd = if (from_end <= y.len) y[y.len - from_end] else 1;
        if (xd != yd and xd != 1 and yd != 1) return null;
        d.* = if (xd == 1) yd else xd;
    }
    return out;
}

// Row-major strides of src placed on the right of a rank-6 array; an axis of size one gets stride zero so that it repeats.
fn stridesOf(src: []const usize) [max_rank]u32 {
    var out: [max_rank]u32 = @splat(0);
    var stride: u32 = 1;
    for (0..src.len) |i| {
        const axis = src.len - 1 - i;
        out[max_rank - 1 - i] = if (src[axis] == 1) 0 else stride;
        stride *= @intCast(src[axis]);
    }
    return out;
}

fn paddedDims(dims: []const usize) [max_rank]u32 {
    var out: [max_rank]u32 = @splat(1);
    for (dims, 0..) |d, i| out[max_rank - dims.len + i] = @intCast(d);
    return out;
}

fn broadcastPlan(out: []const usize, x: []const usize, y: []const usize) gpu_mod.BroadcastPlan {
    return .{ .dims = paddedDims(out), .a_strides = stridesOf(x), .b_strides = stridesOf(y) };
}

// Row-major strides without the zero for axes of size one, which a gather needs to step over them correctly.
fn stridesOfFull(dims: []const usize) [max_rank]u32 {
    var out: [max_rank]u32 = @splat(0);
    var stride: u32 = 1;
    for (0..dims.len) |i| {
        const axis = dims.len - 1 - i;
        out[max_rank - 1 - i] = stride;
        stride *= @intCast(dims[axis]);
    }
    return out;
}

fn hasInput(node: Node, index: usize) bool {
    return index < node.inputs.len and node.inputs[index].len != 0;
}

// The runtime reads only the integers of a Slice operand, so the data stays empty for tensors that may be on the device.
fn boundsOf(v: Value) Array {
    return .{ .dims = v.dims, .data = &.{}, .ints = v.ints };
}

// Most copies a Concat may take: it copies one block per input and outer index, because the elementwise kernel writes only
// contiguous output. TODO: a kernel with output strides would take any axis in one launch.
const max_concat_copies = 64;

// ***** window parameters *****

const ConvParams = struct { k: usize, stride: usize, pad: usize, dilation: usize, groups: usize };

// Reads a convolution node that the NHWC kernels can take: a square window with equal strides, pads and dilations in both directions.
fn convParams(node: Node, weights: []const usize) ?ConvParams {
    if (weights.len != 4 or weights[2] != weights[3]) return null;
    const strides = node.ints("strides");
    const dilations = node.ints("dilations");
    if (runtime.pair(strides, 0, 1) != runtime.pair(strides, 1, 1)) return null;
    if (runtime.pair(dilations, 0, 1) != runtime.pair(dilations, 1, 1)) return null;
    if (!runtime.padsAreUniform(node)) return null;
    const groups = std.math.cast(usize, node.int("group", 1)) orelse return null;
    if (groups == 0 or weights[0] % groups != 0) return null;
    const p: ConvParams = .{ .k = weights[2], .stride = runtime.pair(strides, 0, 1), .pad = runtime.pair(node.ints("pads"), 0, 0), .dilation = runtime.pair(dilations, 0, 1), .groups = groups };
    // A zero kernel, stride or dilation is a malformed model; the host path reports it.
    if (p.k == 0 or p.stride == 0 or p.dilation == 0) return null;
    return p;
}

const PoolParams = struct { k: usize, stride: usize, pad: usize };

fn poolParams(node: Node) ?PoolParams {
    const kernel = node.ints("kernel_shape");
    if (kernel.len != 2 or kernel[0] != kernel[1]) return null;
    const strides = node.ints("strides");
    if (runtime.pair(strides, 0, 1) != runtime.pair(strides, 1, 1)) return null;
    if (!runtime.padsAreUniform(node) or node.int("ceil_mode", 0) != 0) return null;
    const p: PoolParams = .{ .k = runtime.pair(kernel, 0, 1), .stride = runtime.pair(strides, 0, 1), .pad = runtime.pair(node.ints("pads"), 0, 0) };
    if (p.k == 0 or p.stride == 0) return null;
    return p;
}

// ***** executor *****

/// Runs an imported ONNX model on the GPU. Ops with a device kernel run there on run-time shapes; any other op downloads
/// its inputs, runs the CPU implementation of the runtime and uploads again when a device op needs the result.
pub const Executor = struct {
    gpa: Allocator,
    model: *const Model,
    gpu: *Gpu,
    // Worker pool of the host fallback; its gpu field stays null.
    par: kernels.Parallel,
    // Device copies of the initializers, by name, kept between runs.
    constants: std.StringHashMap([]f32),
    // Device arrays derived from initializers once: reordered convolution weights, batch-norm scale and shift, zero biases.
    derived: std.StringHashMap([]f32),
    one: []f32,
    minus_one: []f32,
    /// What the last run did: nodes on the device and on the host, and the copies between them.
    stats: Stats = .{},

    pub const Stats = struct { device_nodes: usize = 0, host_nodes: usize = 0, uploads: usize = 0, downloads: usize = 0 };

    /// Prepares a model for the device gpu; par is the CPU worker pool for ops without a device kernel.
    /// The device must have a step arena big enough for every activation of one run (the arena is reset by each run).
    pub fn init(gpa: Allocator, model: *const Model, gpu: *Gpu, par: kernels.Parallel) Error!Executor {
        const one = try gpu.alloc(f32, 1);
        const minus_one = try gpu.alloc(f32, 1);
        try gpu.upload(one, &[_]f32{1});
        try gpu.upload(minus_one, &[_]f32{-1});
        var host = par;
        host.gpu = null;
        return .{ .gpa = gpa, .model = model, .gpu = gpu, .par = host, .constants = .init(gpa), .derived = .init(gpa), .one = one, .minus_one = minus_one };
    }

    pub fn deinit(self: *Executor) void {
        self.constants.deinit();
        self.derived.deinit();
    }

    /// Runs the graph for the given feeds and returns the outputs on the host, like Model.run.
    pub fn run(self: *Executor, gpa: Allocator, feeds: []const Model.Feed) Error!Model.Result {
        var result: Model.Result = .{ .arena = .init(gpa), .outputs = &.{} };
        errdefer result.deinit();
        const a = result.arena.allocator();
        self.stats = .{};
        try self.gpu.resetStep();
        var values = Values.init(a);
        var it = self.model.initializers.iterator();
        while (it.next()) |entry| {
            const array = entry.value_ptr.*;
            try values.put(entry.key_ptr.*, .{ .dims = array.dims, .host = array.data, .ints = array.ints, .constant = entry.key_ptr.* });
        }
        try runtime.checkFeeds(feeds);
        for (feeds) |feed| try values.put(feed.name, .{ .dims = feed.dims, .host = @constCast(feed.data) });
        for (self.model.inputs) |name| {
            if (!values.contains(name)) return error.MissingInput;
        }
        var at: usize = 0;
        while (at < self.model.nodes.len) at += try self.step(a, &values, at);

        const outputs = try a.alloc(Array, self.model.outputs.len);
        for (self.model.outputs, outputs) |name, *slot| {
            const value = values.getPtr(name) orelse return error.BadModel;
            slot.* = .{ .dims = try a.dupe(usize, value.dims), .data = try self.hostData(a, value), .ints = value.ints };
        }
        result.outputs = outputs;
        return result;
    }

    // ***** transfers *****

    fn hostData(self: *Executor, a: Allocator, value: *Value) Error![]f32 {
        if (value.host) |host| return host;
        const host = try a.alloc(f32, value.len());
        try self.gpu.download(host, value.dev.?);
        self.stats.downloads += 1;
        value.host = host;
        return host;
    }

    fn deviceData(self: *Executor, value: *Value) Error![]f32 {
        if (value.dev) |dev| return dev;
        const host = value.host.?;
        if (value.constant) |name| {
            if (self.constants.get(name)) |dev| {
                value.dev = dev;
                return dev;
            }
            const dev = try self.gpu.alloc(f32, host.len);
            try self.gpu.upload(dev, host);
            try self.constants.put(name, dev);
            value.dev = dev;
            return dev;
        }
        const dev = try self.gpu.scratch(f32, host.len);
        try self.gpu.upload(dev, host);
        self.stats.uploads += 1;
        value.dev = dev;
        return dev;
    }

    // Returns input index of the node with a device copy.
    fn devInput(self: *Executor, values: *Values, node: Node, index: usize) Error!Value {
        if (index >= node.inputs.len) return error.MissingInput;
        const slot = values.getPtr(node.inputs[index]) orelse return error.MissingInput;
        _ = try self.deviceData(slot);
        return slot.*;
    }

    fn newValue(self: *Executor, a: Allocator, dims: []const usize) Error!Value {
        return .{ .dims = try a.dupe(usize, dims), .dev = try self.gpu.scratch(f32, product(dims)) };
    }

    // ***** scheduling *****

    // Runs the node at a (or the group of nodes starting there) and returns how many nodes it consumed.
    fn step(self: *Executor, a: Allocator, values: *Values, at: usize) Error!usize {
        if (try self.fusedWindow(a, values, at)) |used| return used;
        const node = self.model.nodes[at];
        if (try self.onDevice(a, values, node)) |out| {
            try values.put(node.outputs[0], out);
            self.stats.device_nodes += 1;
            return 1;
        }
        try self.onHost(a, values, node);
        return 1;
    }

    fn onHost(self: *Executor, a: Allocator, values: *Values, node: Node) Error!void {
        var inputs = std.StringHashMap(Array).init(a);
        for (node.inputs) |name| {
            if (name.len == 0) continue;
            const slot = values.getPtr(name) orelse return error.MissingInput;
            try inputs.put(name, .{ .dims = slot.dims, .data = try self.hostData(a, slot), .ints = slot.ints });
        }
        const outputs = try runtime.execute(a, self.par, node, &inputs);
        for (node.outputs, 0..) |name, i| {
            if (i < outputs.len) try values.put(name, .{ .dims = outputs[i].dims, .host = outputs[i].data, .ints = outputs[i].ints });
        }
        self.stats.host_nodes += 1;
    }

    // Runs the exporter's Transpose, window op, Transpose wrapper as one NHWC op, the way Model.run does on the CPU, so that the
    // layout transposes never execute. Returns null when the pattern or the window does not fit.
    fn fusedWindow(self: *Executor, a: Allocator, values: *Values, at: usize) Error!?usize {
        const nodes = self.model.nodes;
        if (at + 3 > nodes.len) return null;
        const lead = nodes[at];
        const mid = nodes[at + 1];
        const trail = nodes[at + 2];
        if (!runtime.isPermuted(lead, .{ 0, 3, 1, 2 }) or !runtime.isPermuted(trail, .{ 0, 2, 3, 1 })) return null;
        const Kind = enum { Conv, MaxPool, AveragePool };
        const kind = std.meta.stringToEnum(Kind, mid.op) orelse return null;
        if (lead.outputs.len != 1 or mid.outputs.len != 1 or trail.inputs.len != 1 or trail.outputs.len != 1) return null;
        if (!std.mem.eql(u8, trail.inputs[0], mid.outputs[0])) return null;
        // Fusing skips the wrapped tensors entirely, so a second user would read a missing value.
        if (self.model.uses(lead.outputs[0]) != 1 or self.model.uses(mid.outputs[0]) != 1) return null;
        const x = try self.devInput(values, lead, 0);
        if (x.dims.len != 4) return null;
        const out = switch (kind) {
            .Conv => try self.convNhwc(a, values, mid, x),
            .MaxPool => try self.poolNhwc(a, mid, x, true),
            .AveragePool => try self.poolNhwc(a, mid, x, false),
        } orelse return null;
        try values.put(trail.outputs[0], out);
        self.stats.device_nodes += 3;
        return 3;
    }

    // ***** ops *****

    // Runs one node on the device, or returns null when it has no device kernel for this node so that the host takes it.
    fn onDevice(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const Kind = enum { Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Gelu, Sqrt, Identity, Dropout, Flatten, Reshape, Transpose, Slice, Concat, MatMul, Gemm, Conv, MaxPool, AveragePool, GlobalAveragePool, BatchNormalization, Softmax, LayerNormalization, Gather };
        const kind = std.meta.stringToEnum(Kind, node.op) orelse return null;
        return switch (kind) {
            .Add => try self.binary(a, values, node, .add),
            .Sub => try self.binary(a, values, node, .sub),
            .Mul => try self.binary(a, values, node, .mul),
            .Div => try self.binary(a, values, node, .div),
            .Sqrt => try self.sqrt(a, values, node),
            .Slice => try self.slice(a, values, node),
            .Concat => try self.concat(a, values, node),
            .Relu => try self.unary(a, values, node, null),
            .Sigmoid => try self.unary(a, values, node, .sigmoid),
            .Tanh => try self.unary(a, values, node, .tanh),
            .Gelu => try self.unary(a, values, node, .gelu),
            .Identity, .Dropout => (values.get(node.inputs[0]) orelse return error.MissingInput),
            .Flatten => try self.flatten(a, values, node),
            .Reshape => try self.reshape(a, values, node),
            .Transpose => try self.transpose(a, values, node),
            .MatMul => try self.matmul(a, values, node),
            .Gemm => try self.gemm(a, values, node),
            .Conv, .MaxPool, .AveragePool, .GlobalAveragePool => try self.window(a, values, node, kind == .Conv),
            .BatchNormalization => try self.batchNorm(a, values, node),
            .Softmax => try self.softmax(a, values, node),
            .LayerNormalization => try self.layerNorm(a, values, node),
            .Gather => try self.gather(a, values, node),
        };
    }

    const BinaryKind = enum { add, sub, mul, div };

    // Add, Sub, Mul and Div with broadcasting. NOTE: the elementwise kernel only adds or multiplies, so a difference is
    // x + (-1) * y and a quotient is x * (1 / y).
    fn binary(self: *Executor, a: Allocator, values: *Values, node: Node, kind: BinaryKind) Error!?Value {
        const x = try self.devInput(values, node, 0);
        const y = try self.devInput(values, node, 1);
        const dims = (try broadcastDims(a, x.dims, y.dims)) orelse return null;
        const out = try self.newValue(a, dims);
        const plan = broadcastPlan(dims, x.dims, y.dims);
        if (kind == .add or kind == .mul) {
            try self.gpu.elementwise(if (kind == .add) .add else .mul, x.dev.?, y.dev.?, out.dev.?, plan);
            return out;
        }
        const operand = try self.gpu.scratch(f32, y.len());
        if (kind == .sub) {
            try self.gpu.elementwise(.mul, y.dev.?, self.minus_one, operand, broadcastPlan(y.dims, y.dims, &.{1}));
        } else {
            try self.gpu.pointwise(y.dev.?, operand, .reciprocal);
        }
        try self.gpu.elementwise(if (kind == .sub) .add else .mul, x.dev.?, operand, out.dev.?, plan);
        return out;
    }

    fn sqrt(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const x = try self.devInput(values, node, 0);
        const out = try self.newValue(a, x.dims);
        try self.gpu.pointwise(x.dev.?, out.dev.?, .sqrt);
        return out;
    }

    // Copies src into the start of dst through the elementwise kernel (a product with a one of the same value).
    fn copy(self: *Executor, src: []const f32, dst: []f32) Error!void {
        try self.gpu.elementwise(.mul, src, self.one, dst, broadcastPlan(&.{src.len}, &.{src.len}, &.{1}));
    }

    fn unary(self: *Executor, a: Allocator, values: *Values, node: Node, kind: ?gpu_mod.Activation) Error!?Value {
        const x = try self.devInput(values, node, 0);
        const out = try self.newValue(a, x.dims);
        if (kind) |k| try self.gpu.activation(x.dev.?, out.dev.?, k) else try self.gpu.relu(x.dev.?, out.dev.?);
        return out;
    }

    fn flatten(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        _ = self;
        var x = values.get(node.inputs[0]) orelse return error.MissingInput;
        x.dims = try a.dupe(usize, &(try runtime.flattenDims(x.dims, node.int("axis", 1))));
        return x;
    }

    fn reshape(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        _ = self;
        var x = values.get(node.inputs[0]) orelse return error.MissingInput;
        const shape = values.get(node.inputs[1]) orelse return error.MissingInput;
        if (shape.ints.len == 0) return null;
        // NOTE: the runtime only reads the dims of its operands here, so empty data stands in for tensors that may be on the device.
        const target = try runtime.reshape(a, .{ .dims = x.dims, .data = &.{} }, .{ .dims = shape.dims, .data = &.{}, .ints = shape.ints });
        x.dims = target.dims;
        return x;
    }

    // Copies x with its axes permuted. NOTE: no transpose kernel is needed, because the elementwise kernel reads its first
    // operand through arbitrary strides: a product with a one of the same value is a gather with the permuted strides.
    fn permuted(self: *Executor, a: Allocator, x: Value, perm: []const usize) Error!?Value {
        if (x.dims.len > max_rank or perm.len != x.dims.len) return null;
        const dims = try a.alloc(usize, perm.len);
        const source = stridesOfFull(x.dims);
        var strides: [max_rank]u32 = @splat(0);
        for (perm, dims, 0..) |p, *d, i| {
            d.* = x.dims[p];
            strides[max_rank - perm.len + i] = source[max_rank - perm.len + p];
        }
        const out = try self.newValue(a, dims);
        try self.gpu.elementwise(.mul, x.dev.?, self.one, out.dev.?, .{ .dims = paddedDims(dims), .a_strides = strides, .b_strides = @splat(0) });
        return out;
    }

    fn transpose(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const x = try self.devInput(values, node, 0);
        const perm = try a.alloc(usize, x.dims.len);
        try runtime.permutationOf(node, perm);
        return self.permuted(a, x, perm);
    }

    // Cuts a range out of some axes. The bounds must be known on the host (initializers or constants). NOTE: the elementwise
    // kernel reads x through the strides of the cut and starts at the offset of its first element, so no slice kernel is needed.
    fn slice(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const x = try self.devInput(values, node, 0);
        if (x.dims.len == 0 or x.dims.len > max_rank or node.inputs.len < 3) return null;
        const starts = values.get(node.inputs[1]) orelse return error.MissingInput;
        const ends = values.get(node.inputs[2]) orelse return error.MissingInput;
        const axes: ?Value = if (hasInput(node, 3)) (values.get(node.inputs[3]) orelse return error.MissingInput) else null;
        const steps: ?Value = if (hasInput(node, 4)) (values.get(node.inputs[4]) orelse return error.MissingInput) else null;
        if (starts.ints.len == 0 or starts.ints.len != ends.ints.len) return null;
        const cut = runtime.cutOf(x.dims, boundsOf(starts), boundsOf(ends), if (axes) |v| boundsOf(v) else null, if (steps) |v| boundsOf(v) else null) catch |err| return if (err == error.UnsupportedOp) null else err;
        const rank = x.dims.len;
        const dims = cut.dims[0..rank];
        if (product(dims) == 0) return null;
        const source = stridesOfFull(x.dims);
        var strides: [max_rank]u32 = @splat(0);
        var offset: usize = 0;
        for (0..rank) |i| {
            const slot = max_rank - rank + i;
            strides[slot] = source[slot] * @as(u32, @intCast(cut.step[i]));
            offset += cut.first[i] * source[slot];
        }
        const out = try self.newValue(a, dims);
        try self.gpu.elementwise(.mul, x.dev.?[offset..], self.one, out.dev.?, .{ .dims = paddedDims(dims), .a_strides = strides, .b_strides = @splat(0) });
        return out;
    }

    fn concat(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const first = try self.devInput(values, node, 0);
        if (first.dims.len == 0) return null;
        const axis = try runtime.normalizeAxis(node.int("axis", 0), first.dims.len);
        var total: usize = 0;
        for (0..node.inputs.len) |i| {
            const part = try self.devInput(values, node, i);
            if (part.dims.len != first.dims.len) return null;
            for (part.dims, first.dims, 0..) |have, want, d| {
                if (d != axis and have != want) return null;
            }
            total += part.dims[axis];
        }
        const outer = product(first.dims[0..axis]);
        const inner = product(first.dims[axis + 1 ..]);
        if (total * inner * outer == 0 or outer * node.inputs.len > max_concat_copies) return null;
        const dims = try a.dupe(usize, first.dims);
        dims[axis] = total;
        const out = try self.newValue(a, dims);
        var offset: usize = 0;
        for (0..node.inputs.len) |i| {
            const part = try self.devInput(values, node, i);
            const width = part.dims[axis] * inner;
            offset += width;
            if (width == 0) continue;
            for (0..outer) |o| try self.copy(part.dev.?[o * width ..][0..width], out.dev.?[o * total * inner + offset - width ..][0..width]);
        }
        return out;
    }

    fn matmul(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const x = try self.devInput(values, node, 0);
        const y = try self.devInput(values, node, 1);
        if (x.dims.len < 2 or y.dims.len < 2) return null;
        const m = x.dims[x.dims.len - 2];
        const k = x.dims[x.dims.len - 1];
        const n = y.dims[y.dims.len - 1];
        if (y.dims[y.dims.len - 2] != k) return null;
        const batch = product(x.dims[0 .. x.dims.len - 2]);
        const weight_batch = product(y.dims[0 .. y.dims.len - 2]);
        if (weight_batch != 1 and weight_batch != batch) return null;
        const dims = try a.dupe(usize, x.dims);
        dims[dims.len - 1] = n;
        const out = try self.newValue(a, dims);
        // NOTE: one matrix shared by every batch entry is a single taller product, which keeps the tiles full.
        if (weight_batch == 1) {
            try self.gpu.matmul(x.dev.?, y.dev.?, out.dev.?, .{ .m = batch * m, .n = n, .k = k }, .{});
        } else {
            try self.gpu.matmul(x.dev.?, y.dev.?, out.dev.?, .{ .m = m, .n = n, .k = k, .batch = batch }, .{});
        }
        return out;
    }

    fn gemm(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const trans_a = node.int("transA", 0) != 0;
        const trans_b = node.int("transB", 0) != 0;
        if ((trans_a and trans_b) or node.float("alpha", 1) != 1 or node.float("beta", 1) != 1) return null;
        const x = try self.devInput(values, node, 0);
        const y = try self.devInput(values, node, 1);
        if (x.dims.len != 2 or y.dims.len != 2) return null;
        const m = if (trans_a) x.dims[1] else x.dims[0];
        const k = if (trans_a) x.dims[0] else x.dims[1];
        const n = if (trans_b) y.dims[0] else y.dims[1];
        if ((if (trans_b) y.dims[1] else y.dims[0]) != k) return null;
        const bias: ?Value = if (hasInput(node, 2)) try self.devInput(values, node, 2) else null;
        const dims = [_]usize{ m, n };
        if (bias) |c| {
            const fit = (try broadcastDims(a, &dims, c.dims)) orelse return null;
            if (!std.mem.eql(usize, fit, &dims)) return null;
        }
        const out = try self.newValue(a, &dims);
        try self.gpu.matmul(x.dev.?, y.dev.?, out.dev.?, .{ .m = m, .n = n, .k = k }, .{ .trans_a = trans_a, .trans_b = trans_b });
        if (bias) |c| try self.gpu.elementwise(.add, out.dev.?, c.dev.?, out.dev.?, broadcastPlan(&dims, &dims, c.dims));
        return out;
    }

    // A convolution or pooling node in the graph's own NCHW layout: the device kernels are NHWC, so the activations are
    // permuted around them. Graphs from the exporter keep NHWC between the layout transposes and never come here.
    fn window(self: *Executor, a: Allocator, values: *Values, node: Node, is_conv: bool) Error!?Value {
        const x = try self.devInput(values, node, 0);
        if (x.dims.len != 4) return null;
        const to_nhwc = [_]usize{ 0, 2, 3, 1 };
        const to_nchw = [_]usize{ 0, 3, 1, 2 };
        if (std.mem.eql(u8, node.op, "GlobalAveragePool")) {
            if (x.dims[2] != x.dims[3]) return null;
            const nhwc = (try self.permuted(a, x, &to_nhwc)) orelse return null;
            const params: PoolParams = .{ .k = x.dims[2], .stride = x.dims[2], .pad = 0 };
            var pooled = (try self.poolWith(a, nhwc, params, false)) orelse return null;
            // NOTE: a one by one map has the same memory order in NHWC and NCHW, so only the dims change.
            pooled.dims = try a.dupe(usize, &.{ x.dims[0], x.dims[1], 1, 1 });
            return pooled;
        }
        const nhwc = (try self.permuted(a, x, &to_nhwc)) orelse return null;
        const result = (if (is_conv) try self.convNhwc(a, values, node, nhwc) else try self.poolNhwc(a, node, nhwc, std.mem.eql(u8, node.op, "MaxPool"))) orelse return null;
        return self.permuted(a, result, &to_nchw);
    }

    fn poolNhwc(self: *Executor, a: Allocator, node: Node, x: Value, is_max: bool) Error!?Value {
        const params = poolParams(node) orelse return null;
        return self.poolWith(a, x, params, is_max);
    }

    fn poolWith(self: *Executor, a: Allocator, x: Value, p: PoolParams, is_max: bool) Error!?Value {
        if (x.dims.len != 4 or p.k == 0 or p.stride == 0 or p.k > x.dims[1] + 2 * p.pad or p.k > x.dims[2] + 2 * p.pad) return null;
        const d: cv.Dims = .{ .n = @intCast(x.dims[0]), .h = @intCast(x.dims[1]), .w = @intCast(x.dims[2]), .c = @intCast(x.dims[3]), .k = @intCast(p.k), .stride = @intCast(p.stride), .pad = @intCast(p.pad) };
        const out = try self.newValue(a, &.{ x.dims[0], cv.outHeight(d), cv.outWidth(d), x.dims[3] });
        if (!is_max) {
            try self.gpu.convKernel(.avgPool, .{ x.dev.?, out.dev.? }, d);
            return out;
        }
        // The winners are only needed by the backward pass, so their buffer is released right after the launch.
        const mark = self.gpu.step_used;
        defer self.gpu.step_used = mark;
        const winners = try self.gpu.scratch(u32, out.len());
        try self.gpu.convKernel(.maxPool, .{ x.dev.?, out.dev.?, winners }, d);
        return out;
    }

    // Weights in the (taps, c_out) order of the NHWC kernels, rearranged once from the initializer's OIHW layout.
    fn reorderedWeights(self: *Executor, name: []const u8, w: Value) Error![]f32 {
        if (self.derived.get(name)) |dev| return dev;
        const host = try runtime.reorderOihw(self.gpa, self.par, .{ .dims = w.dims, .data = w.host.? });
        defer self.gpa.free(host);
        const dev = try self.gpu.alloc(f32, host.len);
        try self.gpu.upload(dev, host);
        try self.derived.put(name, dev);
        return dev;
    }

    fn convNhwc(self: *Executor, a: Allocator, values: *Values, node: Node, x: Value) Error!?Value {
        if (node.inputs.len < 2 or x.dims.len != 4) return null;
        const w = values.get(node.inputs[1]) orelse return error.MissingInput;
        const name = w.constant orelse return null;
        const p = convParams(node, w.dims) orelse return null;
        const n, const h, const width, const c_in = .{ x.dims[0], x.dims[1], x.dims[2], x.dims[3] };
        const c_out = w.dims[0];
        if (w.dims[1] * p.groups != c_in) return null;
        // The dilated window must fit the padded input, otherwise the unsigned output size would underflow.
        if (p.dilation * (p.k - 1) + 1 > h + 2 * p.pad or p.dilation * (p.k - 1) + 1 > width + 2 * p.pad) return null;
        const bias: ?Value = if (hasInput(node, 2)) try self.devInput(values, node, 2) else null;
        if (bias) |b| {
            if (b.len() != c_out) return null;
        }
        const d: cv.Dims = .{ .n = @intCast(n), .h = @intCast(h), .w = @intCast(width), .c = @intCast(c_in), .k = @intCast(p.k), .stride = @intCast(p.stride), .pad = @intCast(p.pad), .dilation = @intCast(p.dilation), .groups = @intCast(p.groups), .c_out = @intCast(c_out) };
        const out = try self.newValue(a, &.{ n, cv.outHeight(d), cv.outWidth(d), c_out });
        const weights = try self.reorderedWeights(name, w);
        const rows = n * cv.outHeight(d) * cv.outWidth(d);
        const taps = p.k * p.k * c_in;
        if (p.groups > 1) {
            try self.gpu.convKernel(.groupedForward, .{ x.dev.?, weights, out.dev.? }, d);
        } else if (p.stride == 1 and c_in % 64 == 0 and c_out % 16 == 0 and self.gpu.f32ImplicitWorthwhile(rows, c_out, taps)) {
            const geometry: ga.ConvGeometry = .{ .n = d.n, .h = d.h, .w = d.w, .c = d.c, .k = d.k, .stride = d.stride, .pad = d.pad, .out_h = cv.outHeight(d), .out_w = cv.outWidth(d), .positions = @intCast(rows), .c_out = d.c_out, .dilation = d.dilation };
            try self.gpu.convF32Forward(x.dev.?, weights, out.dev.?, geometry, .{});
        } else {
            // The column matrix is taps times the input and is not needed after the product, so it is released at once.
            const mark = self.gpu.step_used;
            defer self.gpu.step_used = mark;
            const columns = try self.gpu.scratch(f32, rows * taps);
            try self.gpu.convKernel(.im2col, .{ x.dev.?, columns }, d);
            try self.gpu.matmul(columns, weights, out.dev.?, .{ .m = rows, .n = c_out, .k = taps }, .{});
        }
        if (bias) |b| try self.gpu.elementwise(.add, out.dev.?, b.dev.?, out.dev.?, broadcastPlan(out.dims, out.dims, b.dims));
        return out;
    }

    // Inference-mode batch norm with the statistics of the initializers: y = x * scale + shift per channel (axis 1).
    fn batchNorm(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        if (node.inputs.len < 5) return null;
        const scale = values.get(node.inputs[1]) orelse return error.MissingInput;
        const shift = values.get(node.inputs[2]) orelse return error.MissingInput;
        const mean = values.get(node.inputs[3]) orelse return error.MissingInput;
        const variance = values.get(node.inputs[4]) orelse return error.MissingInput;
        const x = try self.devInput(values, node, 0);
        if (x.dims.len < 2 or x.dims.len > max_rank) return null;
        const c = x.dims[1];
        const host = [_]?[]f32{ scale.host, shift.host, mean.host, variance.host };
        for (host) |h| {
            if (h == null or h.?.len != c) return null;
        }
        const key = node.outputs[0];
        const folded = if (self.derived.get(key)) |dev| dev else blk: {
            const eps = node.float("epsilon", 1e-5);
            const both = try self.gpa.alloc(f32, 2 * c);
            defer self.gpa.free(both);
            for (0..c) |ch| {
                const s = scale.host.?[ch] / @sqrt(variance.host.?[ch] + eps);
                both[ch] = s;
                both[c + ch] = shift.host.?[ch] - mean.host.?[ch] * s;
            }
            const dev = try self.gpu.alloc(f32, 2 * c);
            try self.gpu.upload(dev, both);
            try self.derived.put(key, dev);
            break :blk dev;
        };
        var channel_dims: [max_rank]usize = @splat(1);
        channel_dims[1] = c;
        const per_channel = channel_dims[0..x.dims.len];
        const out = try self.newValue(a, x.dims);
        try self.gpu.elementwise(.mul, x.dev.?, folded[0..c], out.dev.?, broadcastPlan(x.dims, x.dims, per_channel));
        try self.gpu.elementwise(.add, out.dev.?, folded[c..], out.dev.?, broadcastPlan(x.dims, x.dims, per_channel));
        return out;
    }

    fn softmax(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const x = try self.devInput(values, node, 0);
        if (x.dims.len == 0 or (runtime.normalizeAxis(node.int("axis", -1), x.dims.len) catch return null) != x.dims.len - 1) return null;
        const out = try self.newValue(a, x.dims);
        try self.gpu.softmax(x.dev.?, out.dev.?, x.dims[x.dims.len - 1], 0, 1);
        return out;
    }

    fn layerNorm(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        const x = try self.devInput(values, node, 0);
        if (x.dims.len == 0 or (runtime.normalizeAxis(node.int("axis", -1), x.dims.len) catch return null) != x.dims.len - 1) return null;
        // NOTE: the device kernel has the engine's epsilon built in.
        if (@abs(node.float("epsilon", 1e-5) - 1e-5) > 1e-9) return null;
        const cols = x.dims[x.dims.len - 1];
        if (cols == 0) return null;
        const scale = try self.devInput(values, node, 1);
        if (scale.len() != cols) return null;
        const beta = if (hasInput(node, 2)) (try self.devInput(values, node, 2)).dev.? else try self.zeros(node.outputs[0], cols);
        if (beta.len != cols) return null;
        const rows = x.len() / cols;
        const out = try self.newValue(a, x.dims);
        const stats = Gpu.LayerNorm(f32){ .x = x.dev.?, .gamma = scale.dev.?, .mean = try self.gpu.scratch(f32, rows), .inv_std = try self.gpu.scratch(f32, rows) };
        try self.gpu.layerNorm(stats, beta, out.dev.?);
        return out;
    }

    fn zeros(self: *Executor, key: []const u8, n: usize) Error![]f32 {
        if (self.derived.get(key)) |dev| return dev;
        const dev = try self.gpu.alloc(f32, n);
        try self.gpu.fill(dev, 0);
        try self.derived.put(key, dev);
        return dev;
    }

    // Picks rows of a table along axis 0; the indices must be known on the host, which holds for graph inputs and constants.
    fn gather(self: *Executor, a: Allocator, values: *Values, node: Node) Error!?Value {
        if (node.int("axis", 0) != 0 or node.inputs.len < 2) return null;
        const table = try self.devInput(values, node, 0);
        const indices = values.get(node.inputs[1]) orelse return error.MissingInput;
        const host = indices.host orelse return null;
        if (table.dims.len == 0) return null;
        const ids = try a.alloc(u32, host.len);
        for (ids, host) |*id, raw| {
            if (raw < 0 or raw >= @as(f32, @floatFromInt(table.dims[0]))) return null;
            id.* = @intFromFloat(raw);
        }
        const dims = try a.alloc(usize, indices.dims.len + table.dims.len - 1);
        @memcpy(dims[0..indices.dims.len], indices.dims);
        @memcpy(dims[indices.dims.len..], table.dims[1..]);
        const out = try self.newValue(a, dims);
        const device_ids = try self.gpu.scratch(u32, ids.len);
        try self.gpu.upload(device_ids, ids);
        try self.gpu.embeddingGather(table.dev.?, device_ids, out.dev.?);
        return out;
    }
};
