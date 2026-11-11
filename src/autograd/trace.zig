const std = @import("std");
const Allocator = std.mem.Allocator;

pub const max_inputs = 5;
pub const max_rank = 6;

pub const Op = enum { gru_update, gru_reset_h, gru_next, linear, add, mul, matmul, batched_matmul, embedding, layer_norm, softmax, relu, sigmoid, tanh, gelu, conv, max_pool, avg_pool, batch_norm, batch_norm_relu, add_relu, cast };

pub const View = struct {
    id: usize,
    rank: usize,
    dims: [max_rank]usize,
    // Integer tensors (token ids) are graph inputs of type int64; every other tensor is f32.
    integer: bool = false,

    /// Describes a slice of ids, which an embedding takes as a runtime input of the exported graph.
    pub fn ofIds(ids: []const u32) View {
        var dims: [max_rank]usize = @splat(1);
        dims[0] = ids.len;
        return .{ .id = @intFromPtr(ids.ptr), .rank = 1, .dims = dims, .integer = true };
    }

    pub fn of(value: anytype) View {
        const Tn = @TypeOf(if (@hasField(@TypeOf(value), "value")) value.value else value);
        const data = if (@hasField(@TypeOf(value), "value")) value.value.data else value.data;
        var dims: [max_rank]usize = @splat(1);
        inline for (Tn.shape, 0..) |d, i| dims[i] = d;
        return .{ .id = @intFromPtr(data.ptr), .rank = Tn.rank, .dims = dims };
    }

    pub fn shape(self: *const View) []const usize {
        return self.dims[0..self.rank];
    }

    pub fn len(self: View) usize {
        var n: usize = 1;
        for (self.shape()) |d| n *= d;
        return n;
    }
};

/// Extra properties of a traced op: window geometry for convolutions and pooling, the causal period of a softmax (zero for none), and the transposed second operand of a batched matmul.
pub const Attrs = struct { kernel: usize = 1, stride: usize = 1, pad: usize = 0, dilation: usize = 1, groups: usize = 1, period: usize = 0, trans_b: bool = false };

pub const Entry = struct {
    op: Op,
    inputs: [max_inputs]View,
    count: usize,
    output: View,
    kernel: usize = 1,
    stride: usize = 1,
    pad: usize = 0,
    dilation: usize = 1,
    groups: usize = 1,
    period: usize = 0,
    trans_b: bool = false,
};

pub const Constant = struct { view: View, value: f32 };

pub const Trace = struct {
    gpa: Allocator,
    entries: std.ArrayList(Entry) = .empty,
    constants: std.ArrayList(Constant) = .empty,

    pub fn deinit(self: *Trace) void {
        self.entries.deinit(self.gpa);
        self.constants.deinit(self.gpa);
    }

    pub fn addConstant(self: *Trace, tensor: anytype, value: f32) Allocator.Error!void {
        try self.constants.append(self.gpa, .{ .view = View.of(tensor), .value = value });
    }

    pub fn add(self: *Trace, op: Op, inputs: anytype, output: anytype, attrs: Attrs) Allocator.Error!void {
        var entry: Entry = .{ .op = op, .inputs = undefined, .count = inputs.len, .output = View.of(output), .kernel = attrs.kernel, .stride = attrs.stride, .pad = attrs.pad, .dilation = attrs.dilation, .groups = attrs.groups, .period = attrs.period, .trans_b = attrs.trans_b };
        inline for (inputs, 0..) |input, i| entry.inputs[i] = if (@TypeOf(input) == View) input else View.of(input);
        try self.entries.append(self.gpa, entry);
    }
};
