const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const layernorm = @import("../gpu/math/layernorm.zig");
const kernels = @import("../cpu/kernels.zig");
const norm = @import("../gpu/math/norm.zig");
const trace_mod = @import("../autograd/trace.zig");

const Entry = trace_mod.Entry;
pub const View = trace_mod.View;

pub const Trace = trace_mod.Trace;
pub const runtime = @import("runtime.zig");
pub const Model = runtime.Model;

pub const Weight = struct { id: usize, data: []const f32 };

/// Host copies of the float slices of a model (from nn.state), named by the address of each slice, ready to be used as Graph.weights.
/// The slices may live on the device of par; deinit frees the copies.
pub fn WeightSet(comptime count: usize) type {
    return struct {
        const Self = @This();

        items: [count]Weight = undefined,
        copied: usize = 0,

        pub fn collect(gpa: Allocator, par: kernels.Parallel, slices: anytype) !Self {
            var set: Self = .{};
            errdefer set.deinit(gpa);
            inline for (slices, 0..) |slice, i| {
                const host = try gpa.alloc(f32, slice.len);
                set.items[i] = .{ .id = @intFromPtr(slice.ptr), .data = host };
                set.copied += 1;
                try par.download(host, slice);
            }
            return set;
        }

        pub fn deinit(self: Self, gpa: Allocator) void {
            for (self.items[0..self.copied]) |w| gpa.free(w.data);
        }
    };
}

pub const Error = error{ UnknownTensor, UnsupportedGraph } || Allocator.Error;

const float_type = 1;
const int32_type = 6;
const int64_type = 7;
const opset = 20;

const Proto = struct {
    bytes: std.ArrayList(u8) = .empty,

    fn varint(self: *Proto, gpa: Allocator, value: u64) !void {
        var v = value;
        while (v >= 0x80) : (v >>= 7) try self.bytes.append(gpa, @as(u8, @truncate(v)) | 0x80);
        try self.bytes.append(gpa, @truncate(v));
    }

    fn tag(self: *Proto, gpa: Allocator, field: u32, wire: u32) !void {
        try self.varint(gpa, @as(u64, field) << 3 | wire);
    }

    fn int(self: *Proto, gpa: Allocator, field: u32, value: i64) !void {
        try self.tag(gpa, field, 0);
        try self.varint(gpa, @bitCast(value));
    }

    fn fixed32(self: *Proto, gpa: Allocator, field: u32, value: f32) !void {
        try self.tag(gpa, field, 5);
        try self.bytes.appendSlice(gpa, &std.mem.toBytes(@as(u32, @bitCast(value))));
    }

    fn bytesField(self: *Proto, gpa: Allocator, field: u32, payload: []const u8) !void {
        try self.tag(gpa, field, 2);
        try self.varint(gpa, payload.len);
        try self.bytes.appendSlice(gpa, payload);
    }

    fn message(self: *Proto, gpa: Allocator, field: u32, inner: *const Proto) !void {
        try self.bytesField(gpa, field, inner.bytes.items);
    }
};

const Known = struct { name: []const u8, view: View };

const Builder = struct {
    gpa: Allocator,
    names: std.heap.ArenaAllocator,
    graph: Proto = .{},
    weights: []const Weight,
    constants: []const trace_mod.Constant,
    values: std.AutoHashMap(usize, Known),
    emitted: std.AutoHashMap(usize, []const u8),
    outputs: []const View,
    // The output id of a batch norm whose scale and shift went into the convolution before it (zero for none).
    folded_norm: usize = 0,
    counter: usize = 0,

    fn fresh(self: *Builder, prefix: []const u8) ![]const u8 {
        self.counter += 1;
        return std.fmt.allocPrint(self.names.allocator(), "{s}{d}", .{ prefix, self.counter });
    }

    fn weightOf(self: *const Builder, id: usize) ?[]const f32 {
        for (self.weights) |w| {
            if (w.id == id) return w.data;
        }
        return null;
    }

    fn tensor(self: *Builder, name: []const u8, dims: []const usize, data_type: u32, raw: []const u8) !void {
        var t: Proto = .{};
        defer t.bytes.deinit(self.gpa);
        for (dims) |d| try t.int(self.gpa, 1, @intCast(d));
        try t.int(self.gpa, 2, data_type);
        try t.bytesField(self.gpa, 8, name);
        try t.bytesField(self.gpa, 9, raw);
        try self.graph.message(self.gpa, 5, &t);
    }

    fn floats(self: *Builder, prefix: []const u8, dims: []const usize, data: []const f32) ![]const u8 {
        const name = try self.fresh(prefix);
        try self.tensor(name, dims, float_type, std.mem.sliceAsBytes(data));
        return name;
    }

    fn shapeConstant(self: *Builder, dims: []const usize) ![]const u8 {
        const name = try self.fresh("shape");
        var values: [trace_mod.max_rank]i64 = undefined;
        for (dims, 0..) |d, i| values[i] = @intCast(d);
        try self.tensor(name, &.{dims.len}, int64_type, std.mem.sliceAsBytes(values[0..dims.len]));
        return name;
    }

    const Attribute = union(enum) { ints: struct { []const u8, []const i64 }, int: struct { []const u8, i64 }, text: struct { []const u8, []const u8 }, float: struct { []const u8, f32 } };

    fn node(self: *Builder, op_type: []const u8, inputs: []const []const u8, output: []const u8, attributes: []const Attribute) !void {
        var n: Proto = .{};
        defer n.bytes.deinit(self.gpa);
        for (inputs) |input| try n.bytesField(self.gpa, 1, input);
        try n.bytesField(self.gpa, 2, output);
        try n.bytesField(self.gpa, 4, op_type);
        for (attributes) |attribute| {
            var a: Proto = .{};
            defer a.bytes.deinit(self.gpa);
            switch (attribute) {
                .ints => |pair| {
                    try a.bytesField(self.gpa, 1, pair[0]);
                    for (pair[1]) |v| try a.int(self.gpa, 8, v);
                    try a.int(self.gpa, 20, 7);
                },
                .int => |pair| {
                    try a.bytesField(self.gpa, 1, pair[0]);
                    try a.int(self.gpa, 3, pair[1]);
                    try a.int(self.gpa, 20, 2);
                },
                .float => |pair| {
                    try a.bytesField(self.gpa, 1, pair[0]);
                    try a.fixed32(self.gpa, 2, pair[1]);
                    try a.int(self.gpa, 20, 1);
                },
                .text => |pair| {
                    try a.bytesField(self.gpa, 1, pair[0]);
                    try a.bytesField(self.gpa, 4, pair[1]);
                    try a.int(self.gpa, 20, 3);
                },
            }
            try n.message(self.gpa, 5, &a);
        }
        try self.graph.message(self.gpa, 1, &n);
    }

    fn use(self: *Builder, view: View) Error![]const u8 {
        if (self.values.get(view.id)) |known| {
            if (std.mem.eql(usize, known.view.shape(), view.shape())) return known.name;
            const name = try self.fresh("reshaped");
            try self.node("Reshape", &.{ known.name, try self.shapeConstant(view.shape()) }, name, &.{});
            try self.values.put(view.id, .{ .name = name, .view = view });
            return name;
        }
        if (self.emitted.get(view.id)) |name| return name;
        for (self.constants) |constant| {
            if (constant.view.id != view.id) continue;
            const data = try self.gpa.alloc(f32, view.len());
            defer self.gpa.free(data);
            @memset(data, constant.value);
            const name = try self.floats("constant", view.shape(), data);
            try self.emitted.put(view.id, name);
            return name;
        }
        const data = self.weightOf(view.id) orelse return error.UnknownTensor;
        const name = try self.floats("weight", view.shape(), data);
        try self.emitted.put(view.id, name);
        return name;
    }

    // Takes the columns start..end of a matrix, the way a fused GRU projection is cut into its gates.
    fn columns(self: *Builder, source: []const u8, start: usize, end: usize) ![]const u8 {
        const name = try self.fresh("part");
        const bounds = [_]i64{ @intCast(start), @intCast(end), 1 };
        const starts = try self.fresh("starts");
        try self.tensor(starts, &.{1}, int64_type, std.mem.sliceAsBytes(bounds[0..1]));
        const ends = try self.fresh("ends");
        try self.tensor(ends, &.{1}, int64_type, std.mem.sliceAsBytes(bounds[1..2]));
        const axes = try self.fresh("axes");
        try self.tensor(axes, &.{1}, int64_type, std.mem.sliceAsBytes(bounds[2..3]));
        try self.node("Slice", &.{ source, starts, ends, axes }, name, &.{});
        return name;
    }

    fn boundary(self: *Builder, comptime prefix: []const u8, index: usize, count: usize) ![]const u8 {
        if (count == 1) return prefix;
        return std.fmt.allocPrint(self.names.allocator(), prefix ++ "{d}", .{index});
    }

    fn produce(self: *Builder, view: View, name: []const u8) !void {
        try self.values.put(view.id, .{ .name = name, .view = view });
    }

    fn channelsFirst(self: *Builder, source: []const u8) ![]const u8 {
        const name = try self.fresh("nchw");
        try self.node("Transpose", &.{source}, name, &.{.{ .ints = .{ "perm", &.{ 0, 3, 1, 2 } } }});
        return name;
    }

    fn channelsLast(self: *Builder, source: []const u8, output: []const u8) !void {
        try self.node("Transpose", &.{source}, output, &.{.{ .ints = .{ "perm", &.{ 0, 2, 3, 1 } } }});
    }

    const Affine = struct { scale: []const f32, shift: []const f32 };

    // The per-channel scale and shift that a batch norm entry applies with its running statistics.
    fn normAffine(self: *Builder, e: Entry) Error!Affine {
        const c = e.inputs[1].dims[0];
        const gamma = self.weightOf(e.inputs[1].id) orelse return error.UnknownTensor;
        const beta = self.weightOf(e.inputs[2].id) orelse return error.UnknownTensor;
        const mean = self.weightOf(e.inputs[3].id) orelse return error.UnknownTensor;
        const variance = self.weightOf(e.inputs[4].id) orelse return error.UnknownTensor;
        const scale = try self.names.allocator().alloc(f32, c);
        const shift = try self.names.allocator().alloc(f32, c);
        for (0..c) |i| {
            scale[i] = gamma[i] / @sqrt(variance[i] + norm.eps);
            shift[i] = beta[i] - mean[i] * scale[i];
        }
        return .{ .scale = scale, .shift = shift };
    }

    // Returns the batch norm that directly follows a convolution when it is the only user of the convolution's output;
    // its scale then goes into the filters and its shift becomes the convolution bias, so no Mul and Add are written.
    fn foldableNorm(self: *const Builder, conv: Entry, rest: []const Entry) ?Entry {
        if (rest.len == 0) return null;
        const next = rest[0];
        if (next.op != .batch_norm and next.op != .batch_norm_relu and next.op != .batch_norm_add_relu) return null;
        if (next.inputs[0].id != conv.output.id) return null;
        if (next.op == .batch_norm_add_relu and next.inputs[5].id == conv.output.id) return null;
        for (rest[1..]) |later| {
            for (later.inputs[0..later.count]) |input| {
                if (input.id == conv.output.id) return null;
            }
        }
        for (self.outputs) |output| {
            if (output.id == conv.output.id) return null;
        }
        return next;
    }

    // Writes the nodes of one traced op; rest holds the entries after it, which a convolution looks at to fold a batch norm.
    fn convert(self: *Builder, e: Entry, rest: []const Entry) Error!void {
        // NOTE: a cast only changes how an activation is stored, so the exported graph stays in f32 and the output aliases the input.
        if (e.op == .cast) return self.produce(e.output, try self.use(e.inputs[0]));
        const out = try self.fresh("value");
        switch (e.op) {
            .cast => unreachable,
            .add => try self.node("Add", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, out, &.{}),
            .mul => try self.node("Mul", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, out, &.{}),
            .matmul => try self.node("MatMul", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, out, &.{}),
            .linear => {
                const product = try self.fresh("product");
                try self.node("MatMul", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, product, &.{});
                try self.node("Add", &.{ product, try self.use(e.inputs[2]) }, out, &.{});
            },
            .gru_update, .gru_reset_h => {
                // NOTE: the fused kernel is written out as the gate arithmetic it replaces, so any ONNX runtime can read it.
                const hidden = e.output.dims[1];
                const first: usize = if (e.op == .gru_update) 0 else hidden;
                const pre = try self.fresh("pre");
                try self.node("Add", &.{ try self.columns(try self.use(e.inputs[0]), first, first + hidden), try self.columns(try self.use(e.inputs[1]), first, first + hidden) }, pre, &.{});
                if (e.op == .gru_update) {
                    try self.node("Sigmoid", &.{pre}, out, &.{});
                } else {
                    const gate = try self.fresh("gate");
                    try self.node("Sigmoid", &.{pre}, gate, &.{});
                    try self.node("Mul", &.{ gate, try self.use(e.inputs[2]) }, out, &.{});
                }
            },
            .gru_next => {
                const hidden = e.output.dims[1];
                const state = try self.use(e.inputs[3]);
                const pre = try self.fresh("pre");
                try self.node("Add", &.{ try self.columns(try self.use(e.inputs[0]), 2 * hidden, 3 * hidden), try self.use(e.inputs[1]) }, pre, &.{});
                const candidate = try self.fresh("candidate");
                try self.node("Tanh", &.{pre}, candidate, &.{});
                const difference = try self.fresh("difference");
                try self.node("Sub", &.{ state, candidate }, difference, &.{});
                const kept = try self.fresh("kept");
                try self.node("Mul", &.{ try self.use(e.inputs[2]), difference }, kept, &.{});
                try self.node("Add", &.{ candidate, kept }, out, &.{});
            },
            .batched_matmul => {
                var right = try self.use(e.inputs[1]);
                if (e.trans_b) {
                    const flipped = try self.fresh("flipped");
                    try self.node("Transpose", &.{right}, flipped, &.{.{ .ints = .{ "perm", &.{ 0, 2, 1 } } }});
                    right = flipped;
                }
                try self.node("MatMul", &.{ try self.use(e.inputs[0]), right }, out, &.{});
            },
            .embedding => try self.node("Gather", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, out, &.{.{ .int = .{ "axis", 0 } }}),
            .layer_norm => try self.node("LayerNormalization", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]), try self.use(e.inputs[2]) }, out, &.{
                .{ .int = .{ "axis", -1 } },
                .{ .float = .{ "epsilon", layernorm.eps } },
            }),
            .softmax => {
                var x = try self.use(e.inputs[0]);
                if (e.scale != 1) {
                    const scaled = try self.fresh("scaled");
                    try self.node("Mul", &.{ x, try self.floats("scale", &.{1}, &.{e.scale}) }, scaled, &.{});
                    x = scaled;
                }
                if (e.period == 0) {
                    try self.node("Softmax", &.{x}, out, &.{.{ .int = .{ "axis", -1 } }});
                } else {
                    // NOTE: the engine lets row r see the first r % period + 1 columns, which is a causal mask once rows are grouped by sequence.
                    const rows = e.output.dims[0];
                    const cols = e.output.dims[1];
                    const mask = try self.gpa.alloc(f32, e.period * cols);
                    defer self.gpa.free(mask);
                    for (0..e.period) |i| {
                        for (0..cols) |j| mask[i * cols + j] = if (j <= i) 0 else -1e9;
                    }
                    const grouped = try self.fresh("grouped");
                    try self.node("Reshape", &.{ x, try self.shapeConstant(&.{ rows / e.period, e.period, cols }) }, grouped, &.{});
                    const masked = try self.fresh("masked");
                    try self.node("Add", &.{ grouped, try self.floats("mask", &.{ e.period, cols }, mask) }, masked, &.{});
                    const weights = try self.fresh("weights");
                    try self.node("Softmax", &.{masked}, weights, &.{.{ .int = .{ "axis", -1 } }});
                    try self.node("Reshape", &.{ weights, try self.shapeConstant(&.{ rows, cols }) }, out, &.{});
                }
            },
            .relu => try self.node("Relu", &.{try self.use(e.inputs[0])}, out, &.{}),
            .sigmoid => try self.node("Sigmoid", &.{try self.use(e.inputs[0])}, out, &.{}),
            .tanh => try self.node("Tanh", &.{try self.use(e.inputs[0])}, out, &.{}),
            .gelu => try self.node("Gelu", &.{try self.use(e.inputs[0])}, out, &.{.{ .text = .{ "approximate", "tanh" } }}),
            .add_relu => {
                const sum = try self.fresh("sum");
                try self.node("Add", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, sum, &.{});
                try self.node("Relu", &.{sum}, out, &.{});
            },
            .conv => {
                const k = e.kernel;
                const c_out = e.inputs[1].dims[1];
                const c_in = e.inputs[1].dims[0] / (k * k);
                const w = self.weightOf(e.inputs[1].id) orelse return error.UnknownTensor;
                const oihw = try self.gpa.alloc(f32, w.len);
                defer self.gpa.free(oihw);
                for (0..c_out) |co| {
                    for (0..c_in) |ci| {
                        for (0..k) |ky| {
                            for (0..k) |kx| oihw[((co * c_in + ci) * k + ky) * k + kx] = w[((ky * k + kx) * c_in + ci) * c_out + co];
                        }
                    }
                }
                var operands: [3][]const u8 = .{ try self.channelsFirst(try self.use(e.inputs[0])), "", "" };
                var count: usize = 2;
                if (self.foldableNorm(e, rest)) |next| {
                    const affine = try self.normAffine(next);
                    const per_filter = oihw.len / c_out;
                    for (0..c_out) |co| {
                        for (oihw[co * per_filter ..][0..per_filter]) |*v| v.* *= affine.scale[co];
                    }
                    operands[2] = try self.floats("bias", &.{c_out}, affine.shift);
                    count = 3;
                    self.folded_norm = next.output.id;
                }
                operands[1] = try self.floats("filters", &.{ c_out, c_in, k, k }, oihw);
                const convolved = try self.fresh("convolved");
                const kernel: i64 = @intCast(k);
                const stride: i64 = @intCast(e.stride);
                const pad: i64 = @intCast(e.pad);
                const dilation: i64 = @intCast(e.dilation);
                const groups: i64 = @intCast(e.groups);
                try self.node("Conv", operands[0..count], convolved, &.{
                    .{ .ints = .{ "kernel_shape", &.{ kernel, kernel } } },
                    .{ .ints = .{ "strides", &.{ stride, stride } } },
                    .{ .ints = .{ "pads", &.{ pad, pad, pad, pad } } },
                    .{ .ints = .{ "dilations", &.{ dilation, dilation } } },
                    .{ .int = .{ "group", groups } },
                });
                try self.channelsLast(convolved, out);
            },
            .max_pool, .avg_pool => {
                const planar = try self.channelsFirst(try self.use(e.inputs[0]));
                const pooled = try self.fresh("pooled");
                const kernel: i64 = @intCast(e.kernel);
                const stride: i64 = @intCast(e.stride);
                const pad: i64 = @intCast(e.pad);
                try self.node(if (e.op == .max_pool) "MaxPool" else "AveragePool", &.{planar}, pooled, &.{
                    .{ .ints = .{ "kernel_shape", &.{ kernel, kernel } } },
                    .{ .ints = .{ "strides", &.{ stride, stride } } },
                    .{ .ints = .{ "pads", &.{ pad, pad, pad, pad } } },
                });
                try self.channelsLast(pooled, out);
            },
            .batch_norm, .batch_norm_relu, .batch_norm_add_relu => {
                var shifted = try self.use(e.inputs[0]);
                // NOTE: a folded batch norm is already in the filters and the bias of the convolution before it.
                if (self.folded_norm != e.output.id) {
                    const c = e.inputs[1].dims[0];
                    const affine = try self.normAffine(e);
                    const scaled = try self.fresh("scaled");
                    try self.node("Mul", &.{ shifted, try self.floats("scale", &.{c}, affine.scale) }, scaled, &.{});
                    shifted = if (e.op == .batch_norm) out else try self.fresh("shifted");
                    try self.node("Add", &.{ scaled, try self.floats("shift", &.{c}, affine.shift) }, shifted, &.{});
                }
                if (e.op == .batch_norm) return self.produce(e.output, shifted);
                var activated = shifted;
                if (e.op == .batch_norm_add_relu) {
                    activated = try self.fresh("summed");
                    try self.node("Add", &.{ shifted, try self.use(e.inputs[5]) }, activated, &.{});
                }
                try self.node("Relu", &.{activated}, out, &.{});
            },
        }
        try self.produce(e.output, out);
    }

    fn valueInfo(self: *Builder, field: u32, name: []const u8, view: View) !void {
        var shape: Proto = .{};
        defer shape.bytes.deinit(self.gpa);
        for (view.shape()) |d| {
            var dim: Proto = .{};
            defer dim.bytes.deinit(self.gpa);
            try dim.int(self.gpa, 1, @intCast(d));
            try shape.message(self.gpa, 1, &dim);
        }
        var tensor_type: Proto = .{};
        defer tensor_type.bytes.deinit(self.gpa);
        try tensor_type.int(self.gpa, 1, if (view.integer) int32_type else float_type);
        try tensor_type.message(self.gpa, 2, &shape);
        var type_proto: Proto = .{};
        defer type_proto.bytes.deinit(self.gpa);
        try type_proto.message(self.gpa, 1, &tensor_type);
        var info: Proto = .{};
        defer info.bytes.deinit(self.gpa);
        try info.bytesField(self.gpa, 1, name);
        try info.message(self.gpa, 2, &type_proto);
        try self.graph.message(self.gpa, field, &info);
    }
};

pub const Graph = struct { trace: *const Trace, weights: []const Weight, inputs: []const View, outputs: []const View };

pub fn build(gpa: Allocator, graph: Graph) Error![]u8 {
    const trace, const weights, const inputs, const outputs = .{ graph.trace, graph.weights, graph.inputs, graph.outputs };
    var b: Builder = .{
        .gpa = gpa,
        .names = .init(gpa),
        .weights = weights,
        .constants = trace.constants.items,
        .values = .init(gpa),
        .emitted = .init(gpa),
        .outputs = outputs,
    };
    defer {
        b.names.deinit();
        b.values.deinit();
        b.emitted.deinit();
        b.graph.bytes.deinit(gpa);
    }
    for (inputs, 0..) |input, i| try b.produce(input, try b.boundary("input", i, inputs.len));
    for (trace.entries.items, 0..) |e, i| try b.convert(e, trace.entries.items[i + 1 ..]);
    for (outputs, 0..) |output, i| try b.node("Identity", &.{try b.use(output)}, try b.boundary("output", i, outputs.len), &.{});
    try b.graph.bytesField(gpa, 2, "grad");
    for (inputs, 0..) |input, i| try b.valueInfo(11, try b.boundary("input", i, inputs.len), input);
    for (outputs, 0..) |output, i| try b.valueInfo(12, try b.boundary("output", i, outputs.len), output);

    var operator_set: Proto = .{};
    defer operator_set.bytes.deinit(gpa);
    try operator_set.bytesField(gpa, 1, "");
    try operator_set.int(gpa, 2, opset);

    var model: Proto = .{};
    errdefer model.bytes.deinit(gpa);
    try model.int(gpa, 1, 9);
    try model.bytesField(gpa, 2, "grad");
    try model.message(gpa, 7, &b.graph);
    try model.message(gpa, 8, &operator_set);
    return model.bytes.toOwnedSlice(gpa);
}

pub fn save(gpa: Allocator, io: Io, path: []const u8, graph: Graph) !void {
    const bytes = try build(gpa, graph);
    defer gpa.free(bytes);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}
