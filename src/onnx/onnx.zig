const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const norm = @import("../gpu/math/norm.zig");
const trace_mod = @import("../autograd/trace.zig");

const Entry = trace_mod.Entry;
pub const View = trace_mod.View;

pub const Trace = trace_mod.Trace;
pub const runtime = @import("runtime.zig");
pub const Model = runtime.Model;

pub const Weight = struct { id: usize, data: []const f32 };

pub const Error = error{ UnknownTensor, UnsupportedGraph } || Allocator.Error;

const float_type = 1;
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

    const Attribute = union(enum) { ints: struct { []const u8, []const i64 }, int: struct { []const u8, i64 }, text: struct { []const u8, []const u8 } };

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

    fn convert(self: *Builder, e: Entry) Error!void {
        const out = try self.fresh("value");
        switch (e.op) {
            .add => try self.node("Add", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, out, &.{}),
            .mul => try self.node("Mul", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, out, &.{}),
            .matmul => try self.node("MatMul", &.{ try self.use(e.inputs[0]), try self.use(e.inputs[1]) }, out, &.{}),
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
                const filters = try self.floats("filters", &.{ c_out, c_in, k, k }, oihw);
                const planar = try self.channelsFirst(try self.use(e.inputs[0]));
                const convolved = try self.fresh("convolved");
                const kernel: i64 = @intCast(k);
                const stride: i64 = @intCast(e.stride);
                const pad: i64 = @intCast(e.pad);
                const dilation: i64 = @intCast(e.dilation);
                const groups: i64 = @intCast(e.groups);
                try self.node("Conv", &.{ planar, filters }, convolved, &.{
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
            .batch_norm, .batch_norm_relu => {
                const c = e.inputs[1].dims[0];
                const gamma = self.weightOf(e.inputs[1].id) orelse return error.UnknownTensor;
                const beta = self.weightOf(e.inputs[2].id) orelse return error.UnknownTensor;
                const mean = self.weightOf(e.inputs[3].id) orelse return error.UnknownTensor;
                const variance = self.weightOf(e.inputs[4].id) orelse return error.UnknownTensor;
                const scale = try self.gpa.alloc(f32, c);
                defer self.gpa.free(scale);
                const shift = try self.gpa.alloc(f32, c);
                defer self.gpa.free(shift);
                for (0..c) |i| {
                    scale[i] = gamma[i] / @sqrt(variance[i] + norm.eps);
                    shift[i] = beta[i] - mean[i] * scale[i];
                }
                const scaled = try self.fresh("scaled");
                try self.node("Mul", &.{ try self.use(e.inputs[0]), try self.floats("scale", &.{c}, scale) }, scaled, &.{});
                const shifted = if (e.op == .batch_norm_relu) try self.fresh("shifted") else out;
                try self.node("Add", &.{ scaled, try self.floats("shift", &.{c}, shift) }, shifted, &.{});
                if (e.op == .batch_norm_relu) try self.node("Relu", &.{shifted}, out, &.{});
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
        try tensor_type.int(self.gpa, 1, float_type);
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

pub fn build(gpa: Allocator, trace: *const Trace, weights: []const Weight, inputs: []const View, outputs: []const View) Error![]u8 {
    var b: Builder = .{
        .gpa = gpa,
        .names = .init(gpa),
        .weights = weights,
        .constants = trace.constants.items,
        .values = .init(gpa),
        .emitted = .init(gpa),
    };
    defer {
        b.names.deinit();
        b.values.deinit();
        b.emitted.deinit();
        b.graph.bytes.deinit(gpa);
    }
    for (inputs, 0..) |input, i| try b.produce(input, try b.boundary("input", i, inputs.len));
    for (trace.entries.items) |e| try b.convert(e);
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

pub fn save(gpa: Allocator, io: Io, path: []const u8, trace: *const Trace, weights: []const Weight, inputs: []const View, outputs: []const View) !void {
    const bytes = try build(gpa, trace, weights, inputs, outputs);
    defer gpa.free(bytes);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

const testing = std.testing;
const autograd = @import("../autograd/autograd.zig");
const Tensor = @import("../tensor/tensor.zig").Tensor;

fn countOf(haystack: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, haystack, needle);
}

test "a traced network becomes an onnx model with its weights, layout changes and reshapes" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var trace: Trace = .{ .gpa = gpa };
    defer trace.deinit();
    var tape: autograd.Tape = .{ .gpa = arena.allocator(), .trace = &trace };

    const image = try tape.constant(try tape.filledTensor(Tensor(f32, &.{ 1, 36 }), 1));
    const filters = try tape.constant(try tape.filledTensor(Tensor(f32, &.{ 9, 4 }), 0.5));
    const dense = try tape.constant(try tape.filledTensor(Tensor(f32, &.{ 16, 2 }), 1));
    const bias = try tape.constant(try tape.filledTensor(Tensor(f32, &.{2}), 0.25));
    const convolved = try tape.conv2d(autograd.reshape(image, &.{ 1, 6, 6, 1 }), filters, .{});
    const pooled = try tape.maxPool(try tape.activate(convolved, .gelu), .{});
    const logits = try tape.add(try tape.matmul(autograd.reshape(pooled, &.{ 1, 16 }), dense), bias);

    const weights = [_]Weight{
        .{ .id = @intFromPtr(filters.value.data.ptr), .data = filters.value.data },
        .{ .id = @intFromPtr(dense.value.data.ptr), .data = dense.value.data },
        .{ .id = @intFromPtr(bias.value.data.ptr), .data = bias.value.data },
    };
    const bytes = try build(gpa, &trace, &weights, &.{View.of(image)}, &.{View.of(logits)});
    defer gpa.free(bytes);

    try testing.expectEqualSlices(u8, &.{ 0x08, 0x09 }, bytes[0..2]);
    for ([_][]const u8{ "Conv", "Gelu", "MaxPool", "MatMul", "Add", "Identity", "output", "input" }) |name| try testing.expect(countOf(bytes, name) >= 1);
    try testing.expectEqual(@as(usize, 2), countOf(bytes, "Reshape"));
    try testing.expectEqual(@as(usize, 4), countOf(bytes, "Transpose"));
    try testing.expectError(error.UnknownTensor, build(gpa, &trace, weights[1..], &.{View.of(image)}, &.{View.of(logits)}));
}
