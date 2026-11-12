const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const kernels = @import("../cpu/kernels.zig");
const cpu_ops = @import("../cpu/ops.zig");
const cv = @import("../gpu/math/conv.zig");

pub const Error = error{ BadModel, UnsupportedOp, MissingInput, ShapeMismatch } || Allocator.Error;

pub const Array = struct {
    dims: []const usize,
    data: []f32,
    ints: []const i64 = &.{},

    pub fn len(self: Array) usize {
        return product(self.dims);
    }
};

fn product(dims: []const usize) usize {
    var n: usize = 1;
    for (dims) |d| n *= d;
    return n;
}

const Field = struct { number: u32, wire: u3, varint: u64 = 0, bytes: []const u8 = &.{} };

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,

    fn varint(self: *Reader) Error!u64 {
        var value: u64 = 0;
        var shift: u6 = 0;
        while (true) {
            if (self.at >= self.bytes.len) return error.BadModel;
            const b = self.bytes[self.at];
            self.at += 1;
            value |= @as(u64, b & 0x7f) << shift;
            if (b & 0x80 == 0) return value;
            shift += 7;
        }
    }

    fn next(self: *Reader) Error!?Field {
        if (self.at >= self.bytes.len) return null;
        const key = try self.varint();
        var field: Field = .{ .number = @intCast(key >> 3), .wire = @intCast(key & 7) };
        switch (field.wire) {
            0 => field.varint = try self.varint(),
            1 => {
                field.bytes = try self.take(8);
            },
            2 => field.bytes = try self.take(try self.varint()),
            5 => field.bytes = try self.take(4),
            else => return error.BadModel,
        }
        return field;
    }

    fn take(self: *Reader, n: u64) Error![]const u8 {
        if (self.at + n > self.bytes.len) return error.BadModel;
        defer self.at += n;
        return self.bytes[self.at..][0..n];
    }
};

fn appendVarints(gpa: Allocator, list: *std.ArrayList(i64), field: Field) Error!void {
    if (field.wire == 0) return list.append(gpa, @bitCast(field.varint));
    var r: Reader = .{ .bytes = field.bytes };
    while (r.at < r.bytes.len) try list.append(gpa, @bitCast(try r.varint()));
}

const Attribute = struct {
    name: []const u8 = "",
    int: i64 = 0,
    float: f32 = 0,
    text: []const u8 = "",
    ints: []const i64 = &.{},
};

pub const Node = struct {
    op: []const u8 = "",
    inputs: []const []const u8 = &.{},
    outputs: []const []const u8 = &.{},
    attributes: []const Attribute = &.{},

    pub fn int(self: Node, name: []const u8, default: i64) i64 {
        for (self.attributes) |a| {
            if (std.mem.eql(u8, a.name, name)) return a.int;
        }
        return default;
    }

    pub fn float(self: Node, name: []const u8, default: f32) f32 {
        for (self.attributes) |a| {
            if (std.mem.eql(u8, a.name, name)) return a.float;
        }
        return default;
    }

    pub fn text(self: Node, name: []const u8, default: []const u8) []const u8 {
        for (self.attributes) |a| {
            if (std.mem.eql(u8, a.name, name)) return a.text;
        }
        return default;
    }

    pub fn ints(self: Node, name: []const u8) []const i64 {
        for (self.attributes) |a| {
            if (std.mem.eql(u8, a.name, name)) return a.ints;
        }
        return &.{};
    }
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    nodes: []const Node,
    initializers: std.StringHashMap(Array),
    inputs: []const []const u8,
    outputs: []const []const u8,

    pub fn deinit(self: *Model) void {
        self.initializers.deinit();
        self.arena.deinit();
    }

    pub fn load(gpa: Allocator, io: Io, path: []const u8) !Model {
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn parse(gpa: Allocator, bytes: []const u8) Error!Model {
        var model: Model = .{ .arena = .init(gpa), .nodes = &.{}, .initializers = .init(gpa), .inputs = &.{}, .outputs = &.{} };
        errdefer model.deinit();
        const a = model.arena.allocator();
        var top: Reader = .{ .bytes = try a.dupe(u8, bytes) };
        while (try top.next()) |field| {
            if (field.number != 7) continue;
            var nodes: std.ArrayList(Node) = .empty;
            var inputs: std.ArrayList([]const u8) = .empty;
            var outputs: std.ArrayList([]const u8) = .empty;
            var graph: Reader = .{ .bytes = field.bytes };
            while (try graph.next()) |g| switch (g.number) {
                1 => try nodes.append(a, try parseNode(a, g.bytes)),
                5 => {
                    const named = try parseTensor(a, g.bytes);
                    try model.initializers.put(named.name, named.array);
                },
                11, 12 => {
                    var info: Reader = .{ .bytes = g.bytes };
                    while (try info.next()) |i| {
                        if (i.number == 1) try (if (g.number == 11) inputs else outputs).append(a, i.bytes);
                    }
                },
                else => {},
            };
            model.nodes = try nodes.toOwnedSlice(a);
            model.inputs = try inputs.toOwnedSlice(a);
            model.outputs = try outputs.toOwnedSlice(a);
        }
        if (model.nodes.len == 0) return error.BadModel;
        return model;
    }

    pub const Feed = struct { name: []const u8, dims: []const usize, data: []const f32 };

    pub const Result = struct {
        arena: std.heap.ArenaAllocator,
        outputs: []Array,

        pub fn deinit(self: *Result) void {
            self.arena.deinit();
        }
    };

    pub fn run(self: *const Model, gpa: Allocator, par: kernels.Parallel, feeds: []const Feed) Error!Result {
        var result: Result = .{ .arena = .init(gpa), .outputs = &.{} };
        errdefer result.deinit();
        const a = result.arena.allocator();
        var values = std.StringHashMap(Array).init(a);
        var it = self.initializers.iterator();
        while (it.next()) |entry| try values.put(entry.key_ptr.*, entry.value_ptr.*);
        for (feeds) |feed| try values.put(feed.name, .{ .dims = feed.dims, .data = @constCast(feed.data) });
        for (self.inputs) |name| {
            if (!values.contains(name)) return error.MissingInput;
        }
        var at: usize = 0;
        while (at < self.nodes.len) {
            if (try self.fusedConv(a, par, &values, at)) |used| {
                at += used;
                continue;
            }
            const node = self.nodes[at];
            const outputs = try execute(a, par, node, &values);
            for (node.outputs, 0..) |name, i| {
                if (i < outputs.len) try values.put(name, outputs[i]);
            }
            at += 1;
        }
        const outputs = try a.alloc(Array, self.outputs.len);
        for (self.outputs, outputs) |name, *slot| slot.* = values.get(name) orelse return error.BadModel;
        result.outputs = outputs;
        return result;
    }

    // Counts the users of a tensor name: every node input plus the graph outputs.
    pub fn uses(self: *const Model, name: []const u8) usize {
        var count: usize = 0;
        for (self.nodes) |node| {
            for (node.inputs) |arg| {
                if (std.mem.eql(u8, arg, name)) count += 1;
            }
        }
        for (self.outputs) |output| {
            if (std.mem.eql(u8, output, name)) count += 1;
        }
        return count;
    }

    // Runs the exporter's Transpose-Conv-Transpose wrapper as one NHWC convolution when the pattern is exact and the
    // wrapped tensors have no other users; returns the number of nodes consumed, or null when the pattern does not
    // match so the caller falls back to executing the nodes one by one.
    fn fusedConv(self: *const Model, a: Allocator, par: kernels.Parallel, values: *std.StringHashMap(Array), at: usize) Error!?usize {
        if (at + 3 > self.nodes.len) return null;
        const lead = self.nodes[at];
        const conv = self.nodes[at + 1];
        const trail = self.nodes[at + 2];
        if (!isPermuted(lead, .{ 0, 3, 1, 2 }) or !isPermuted(trail, .{ 0, 2, 3, 1 })) return null;
        if (!std.mem.eql(u8, conv.op, "Conv") or conv.int("group", 1) != 1) return null;
        if (lead.outputs.len != 1 or conv.outputs.len != 1 or trail.inputs.len != 1) return null;
        if (!std.mem.eql(u8, trail.inputs[0], conv.outputs[0])) return null;
        // Fusing skips the wrapped tensors entirely, so a second user would silently read a missing value.
        if (self.uses(lead.outputs[0]) != 1 or self.uses(conv.outputs[0]) != 1) return null;
        const weights = try input(values, conv, 1);
        // The NHWC convolution handles square windows only; anything else keeps the transpose path.
        if (weights.dims.len != 4 or weights.dims[2] != weights.dims[3]) return null;
        if (pair(conv.ints("strides"), 0, 1) != pair(conv.ints("strides"), 1, 1)) return null;
        if (!padsAreUniform(conv)) return null;
        if (pair(conv.ints("dilations"), 0, 1) != pair(conv.ints("dilations"), 1, 1)) return null;
        const x = try input(values, lead, 0);
        const out = try convolveNhwc(a, par, conv, x, weights, maybeInput(values, conv, 2));
        try values.put(trail.outputs[0], out);
        return 3;
    }
};

/// Returns true when a convolution node pads all four sides by the same amount and has no auto_pad rule.
/// NOTE: the NHWC kernel pads every side alike, so asymmetric pads and any auto_pad other than NOTSET must keep the transpose path.
pub fn padsAreUniform(conv: Node) bool {
    const auto_pad = conv.text("auto_pad", "NOTSET");
    if (auto_pad.len != 0 and !std.mem.eql(u8, auto_pad, "NOTSET")) return false;
    const pads = conv.ints("pads");
    if (pads.len == 0) return true;
    if (pads.len != 4) return false;
    return std.mem.allEqual(i64, pads, pads[0]);
}

pub fn isPermuted(node: Node, comptime perm: [4]i64) bool {
    if (!std.mem.eql(u8, node.op, "Transpose")) return false;
    return std.mem.eql(i64, node.ints("perm"), &perm);
}

fn parseNode(a: Allocator, bytes: []const u8) Error!Node {
    var node: Node = .{};
    var inputs: std.ArrayList([]const u8) = .empty;
    var outputs: std.ArrayList([]const u8) = .empty;
    var attributes: std.ArrayList(Attribute) = .empty;
    var r: Reader = .{ .bytes = bytes };
    while (try r.next()) |f| switch (f.number) {
        1 => try inputs.append(a, f.bytes),
        2 => try outputs.append(a, f.bytes),
        4 => node.op = f.bytes,
        5 => try attributes.append(a, try parseAttribute(a, f.bytes)),
        else => {},
    };
    node.inputs = try inputs.toOwnedSlice(a);
    node.outputs = try outputs.toOwnedSlice(a);
    node.attributes = try attributes.toOwnedSlice(a);
    return node;
}

fn parseAttribute(a: Allocator, bytes: []const u8) Error!Attribute {
    var attribute: Attribute = .{};
    var ints: std.ArrayList(i64) = .empty;
    var r: Reader = .{ .bytes = bytes };
    while (try r.next()) |f| switch (f.number) {
        1 => attribute.name = f.bytes,
        2 => attribute.float = @bitCast(std.mem.readInt(u32, f.bytes[0..4], .little)),
        3 => attribute.int = @bitCast(f.varint),
        4 => attribute.text = f.bytes,
        8 => try appendVarints(a, &ints, f),
        else => {},
    };
    attribute.ints = try ints.toOwnedSlice(a);
    return attribute;
}

fn parseTensor(a: Allocator, bytes: []const u8) Error!struct { name: []const u8, array: Array } {
    var dims: std.ArrayList(i64) = .empty;
    var data_type: u64 = 1;
    var name: []const u8 = "";
    var raw: []const u8 = &.{};
    var float_data: std.ArrayList(f32) = .empty;
    var int_data: std.ArrayList(i64) = .empty;
    var r: Reader = .{ .bytes = bytes };
    while (try r.next()) |f| switch (f.number) {
        1 => try appendVarints(a, &dims, f),
        2 => data_type = f.varint,
        4 => if (f.wire == 5) {
            try float_data.append(a, @bitCast(std.mem.readInt(u32, f.bytes[0..4], .little)));
        } else {
            for (0..f.bytes.len / 4) |i| try float_data.append(a, @bitCast(std.mem.readInt(u32, f.bytes[i * 4 ..][0..4], .little)));
        },
        5, 7 => try appendVarints(a, &int_data, f),
        8 => name = f.bytes,
        9 => raw = f.bytes,
        else => {},
    };
    const shape = try a.alloc(usize, dims.items.len);
    for (dims.items, shape) |d, *s| s.* = @intCast(d);
    const count = product(shape);
    var array: Array = .{ .dims = shape, .data = try a.alloc(f32, count) };
    switch (data_type) {
        1 => if (raw.len == 4 * count) {
            for (array.data, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        } else if (float_data.items.len == count) {
            @memcpy(array.data, float_data.items);
        } else return error.BadModel,
        6, 7 => {
            const ints = try a.alloc(i64, count);
            const width: usize = if (data_type == 7) 8 else 4;
            if (raw.len == width * count) {
                for (ints, 0..) |*v, i| v.* = if (data_type == 7) std.mem.readInt(i64, raw[i * 8 ..][0..8], .little) else std.mem.readInt(i32, raw[i * 4 ..][0..4], .little);
            } else if (int_data.items.len == count) {
                @memcpy(ints, int_data.items);
            } else return error.BadModel;
            for (ints, array.data) |v, *f| f.* = @floatFromInt(v);
            array.ints = ints;
        },
        else => return error.UnsupportedOp,
    }
    return .{ .name = name, .array = array };
}

fn make(a: Allocator, dims: []const usize) Error!Array {
    const owned = try a.dupe(usize, dims);
    return .{ .dims = owned, .data = try a.alloc(f32, product(dims)) };
}

pub fn input(values: *std.StringHashMap(Array), node: Node, index: usize) Error!Array {
    if (index >= node.inputs.len) return error.MissingInput;
    return values.get(node.inputs[index]) orelse error.MissingInput;
}

fn maybeInput(values: *std.StringHashMap(Array), node: Node, index: usize) ?Array {
    if (index >= node.inputs.len or node.inputs[index].len == 0) return null;
    return values.get(node.inputs[index]);
}

const max_rank = 8;

fn stridesFor(dims: []const usize, rank: usize, out: *[max_rank]usize) void {
    @memset(out, 0);
    var stride: usize = 1;
    var i = dims.len;
    while (i > 0) {
        i -= 1;
        out[rank - dims.len + i] = if (dims[i] == 1) 0 else stride;
        stride *= dims[i];
    }
}

// Returns true when small, without its leading ones, equals the trailing axes of big: small then repeats along the leading
// axes of big (a per-channel scale, a bias, a scalar or a tensor of the same shape) and the result has the shape of big.
fn repeatsAlong(small: []const usize, big: []const usize) bool {
    var rest = small;
    while (rest.len > 0 and rest[0] == 1) rest = rest[1..];
    if (small.len > big.len) return false;
    return std.mem.eql(usize, rest, big[big.len - rest.len ..]);
}

fn Repeated(comptime f: fn (f32, f32) f32) type {
    return struct {
        const Job = struct { big: []const f32, small: []const f32, out: []f32, swapped: bool };

        fn run(job: Job, lo: usize, hi: usize) void {
            var j = lo % job.small.len;
            for (job.out[lo..hi], job.big[lo..hi]) |*o, v| {
                o.* = if (job.swapped) f(job.small[j], v) else f(v, job.small[j]);
                j += 1;
                if (j == job.small.len) j = 0;
            }
        }
    };
}

/// Combines x and y element by element with ONNX (numpy) broadcasting. The common cases, equal shapes or one operand
/// repeating along the leading axes of the other, run as a flat loop on the worker pool; other shapes take an index walk.
pub fn broadcast(a: Allocator, par: kernels.Parallel, x: Array, y: Array, comptime f: fn (f32, f32) f32) Error!Array {
    const x_big = repeatsAlong(y.dims, x.dims);
    if (x_big or repeatsAlong(x.dims, y.dims)) {
        const big = if (x_big) x else y;
        const out = try make(a, big.dims);
        if (out.data.len == 0) return out;
        const job: Repeated(f).Job = .{ .big = big.data, .small = if (x_big) y.data else x.data, .out = out.data, .swapped = !x_big };
        kernels.parallelFor(par, out.data.len, 1 << 14, job, Repeated(f).run);
        return out;
    }
    const rank = @max(x.dims.len, y.dims.len);
    var dims: [max_rank]usize = undefined;
    for (0..rank) |i| {
        const dx = if (i + x.dims.len >= rank) x.dims[i + x.dims.len - rank] else 1;
        const dy = if (i + y.dims.len >= rank) y.dims[i + y.dims.len - rank] else 1;
        if (dx != dy and dx != 1 and dy != 1) return error.ShapeMismatch;
        dims[i] = @max(dx, dy);
    }
    const out = try make(a, dims[0..rank]);
    var sx: [max_rank]usize = undefined;
    var sy: [max_rank]usize = undefined;
    stridesFor(x.dims, rank, &sx);
    stridesFor(y.dims, rank, &sy);
    var index: [max_rank]usize = @splat(0);
    for (out.data) |*v| {
        var ox: usize = 0;
        var oy: usize = 0;
        for (0..rank) |d| {
            ox += index[d] * sx[d];
            oy += index[d] * sy[d];
        }
        v.* = f(x.data[ox], y.data[oy]);
        var d = rank;
        while (d > 0) {
            d -= 1;
            index[d] += 1;
            if (index[d] < dims[d]) break;
            index[d] = 0;
        }
    }
    return out;
}

pub fn plus(x: f32, y: f32) f32 {
    return x + y;
}
pub fn minus(x: f32, y: f32) f32 {
    return x - y;
}
fn times(x: f32, y: f32) f32 {
    return x * y;
}
fn over(x: f32, y: f32) f32 {
    return x / y;
}

fn map(a: Allocator, par: kernels.Parallel, x: Array, comptime f: fn (f32) f32) Error!Array {
    const out = try make(a, x.dims);
    const Job = struct { x: []const f32, out: []f32 };
    kernels.parallelFor(par, out.data.len, 1 << 14, Job{ .x = x.data, .out = out.data }, struct {
        fn run(job: Job, lo: usize, hi: usize) void {
            for (job.out[lo..hi], job.x[lo..hi]) |*o, v| o.* = f(v);
        }
    }.run);
    return out;
}

fn relu(x: f32) f32 {
    return @max(x, 0);
}
fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
fn tanhf(x: f32) f32 {
    return std.math.tanh(x);
}
fn geluTanh(x: f32) f32 {
    return 0.5 * x * (1 + std.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}
fn sqrtf(x: f32) f32 {
    return @sqrt(x);
}

fn normalizeAxis(axis: i64, rank: usize) usize {
    return @intCast(if (axis < 0) axis + @as(i64, @intCast(rank)) else axis);
}

pub fn execute(a: Allocator, par: kernels.Parallel, node: Node, values: *std.StringHashMap(Array)) Error![]const Array {
    const Kind = enum {
        Add,
        Sub,
        Mul,
        Div,
        Relu,
        Sigmoid,
        Tanh,
        Gelu,
        Sqrt,
        MatMul,
        Gemm,
        Conv,
        MaxPool,
        AveragePool,
        GlobalAveragePool,
        BatchNormalization,
        Flatten,
        Reshape,
        Transpose,
        Concat,
        Softmax,
        Identity,
        Dropout,
        Gather,
        LayerNormalization,
        Slice,
    };
    const kind = std.meta.stringToEnum(Kind, node.op) orelse return error.UnsupportedOp;
    const out = try a.alloc(Array, 1);
    out[0] = switch (kind) {
        .Add => try broadcast(a, par, try input(values, node, 0), try input(values, node, 1), plus),
        .Sub => try broadcast(a, par, try input(values, node, 0), try input(values, node, 1), minus),
        .Mul => try broadcast(a, par, try input(values, node, 0), try input(values, node, 1), times),
        .Div => try broadcast(a, par, try input(values, node, 0), try input(values, node, 1), over),
        .Relu => try map(a, par, try input(values, node, 0), relu),
        .Sigmoid => try map(a, par, try input(values, node, 0), sigmoid),
        .Tanh => try map(a, par, try input(values, node, 0), tanhf),
        .Gelu => try map(a, par, try input(values, node, 0), geluTanh),
        .Sqrt => try map(a, par, try input(values, node, 0), sqrtf),
        .Identity, .Dropout => try input(values, node, 0),
        .MatMul => try matmul(a, try input(values, node, 0), try input(values, node, 1)),
        .Gemm => try gemm(a, par, node, try input(values, node, 0), try input(values, node, 1), maybeInput(values, node, 2)),
        .Conv => try convolve(a, par, node, try input(values, node, 0), try input(values, node, 1), maybeInput(values, node, 2)),
        .MaxPool, .AveragePool => try pool(a, par, node, try input(values, node, 0), kind == .MaxPool),
        .GlobalAveragePool => try globalAverage(a, try input(values, node, 0)),
        .BatchNormalization => try batchNorm(a, node, try input(values, node, 0), try input(values, node, 1), try input(values, node, 2), try input(values, node, 3), try input(values, node, 4)),
        .Flatten => blk: {
            const x = try input(values, node, 0);
            const axis = normalizeAxis(node.int("axis", 1), x.dims.len);
            break :blk .{ .dims = try a.dupe(usize, &.{ product(x.dims[0..axis]), product(x.dims[axis..]) }), .data = x.data };
        },
        .Reshape => try reshape(a, try input(values, node, 0), try input(values, node, 1)),
        .Transpose => try transpose(a, par, node, try input(values, node, 0)),
        .Concat => try concat(a, node, values),
        .Softmax => try softmax(a, node, try input(values, node, 0)),
        .Gather => try gather(a, node, try input(values, node, 0), try input(values, node, 1)),
        .Slice => try slice(a, try input(values, node, 0), try input(values, node, 1), try input(values, node, 2), maybeInput(values, node, 3), maybeInput(values, node, 4)),
        .LayerNormalization => try layerNorm(a, node, try input(values, node, 0), try input(values, node, 1), maybeInput(values, node, 2)),
    };
    return out;
}

fn matmul(a: Allocator, x: Array, y: Array) Error!Array {
    if (x.dims.len < 2 or y.dims.len < 2) return error.ShapeMismatch;
    const m = x.dims[x.dims.len - 2];
    const k = x.dims[x.dims.len - 1];
    const n = y.dims[y.dims.len - 1];
    if (y.dims[y.dims.len - 2] != k) return error.ShapeMismatch;
    const batch = product(x.dims[0 .. x.dims.len - 2]);
    const weight_batch = product(y.dims[0 .. y.dims.len - 2]);
    if (weight_batch != 1 and weight_batch != batch) return error.ShapeMismatch;
    var dims: [max_rank]usize = undefined;
    @memcpy(dims[0 .. x.dims.len - 1], x.dims[0 .. x.dims.len - 1]);
    dims[x.dims.len - 1] = n;
    const out = try make(a, dims[0..x.dims.len]);
    for (0..batch) |b| {
        const w = y.data[(if (weight_batch == 1) 0 else b) * k * n ..][0 .. k * n];
        kernels.matmulVector(f32, .{ .m = m, .k = k, .n = n, .a = x.data[b * m * k ..][0 .. m * k], .b = w, .out = out.data[b * m * n ..][0 .. m * n] });
    }
    return out;
}

fn gemm(a: Allocator, par: kernels.Parallel, node: Node, x: Array, y: Array, c: ?Array) Error!Array {
    const trans_a = node.int("transA", 0) != 0;
    const trans_b = node.int("transB", 0) != 0;
    const alpha = node.float("alpha", 1);
    const beta = node.float("beta", 1);
    const m = if (trans_a) x.dims[1] else x.dims[0];
    const k = if (trans_a) x.dims[0] else x.dims[1];
    const n = if (trans_b) y.dims[0] else y.dims[1];
    const lhs = if (trans_a) try transposed2d(a, x) else x;
    const rhs = if (trans_b) try transposed2d(a, y) else y;
    const out = try matmul(a, .{ .dims = &.{ m, k }, .data = lhs.data }, .{ .dims = &.{ k, n }, .data = rhs.data });
    for (out.data) |*v| v.* *= alpha;
    if (c) |bias| {
        const scaled = try make(a, bias.dims);
        for (scaled.data, bias.data) |*o, v| o.* = v * beta;
        return broadcast(a, par, out, scaled, plus);
    }
    return out;
}

fn transposed2d(a: Allocator, x: Array) Error!Array {
    const out = try make(a, &.{ x.dims[1], x.dims[0] });
    for (0..x.dims[0]) |i| {
        for (0..x.dims[1]) |j| out.data[j * x.dims[0] + i] = x.data[i * x.dims[1] + j];
    }
    return out;
}

pub fn pair(list: []const i64, index: usize, default: usize) usize {
    return if (index < list.len) @intCast(list[index]) else default;
}

const Window = struct { c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, py: usize, px: usize, dy: usize, dx: usize, oh: usize, ow: usize };

const ColumnJob = struct { x: []const f32, col: []f32, v: Window };

// Builds the rows of the column matrix that belong to input channels lo to hi.
fn planarColumns(job: ColumnJob, lo: usize, hi: usize) void {
    const x, const col, const v = .{ job.x, job.col, job.v };
    const positions = v.oh * v.ow;
    for (lo..hi) |ic| {
        for (0..v.kh) |ky| {
            for (0..v.kw) |kx| {
                const row = col[((ic * v.kh + ky) * v.kw + kx) * positions ..][0..positions];
                for (0..v.oh) |oy| {
                    const iy = oy * v.sy + ky * v.dy;
                    const line = row[oy * v.ow ..][0..v.ow];
                    if (iy < v.py or iy - v.py >= v.h) {
                        @memset(line, 0);
                        continue;
                    }
                    const source = x[(ic * v.h + iy - v.py) * v.w ..][0..v.w];
                    for (line, 0..) |*o, ox| {
                        const ix = ox * v.sx + kx * v.dx;
                        o.* = if (ix < v.px or ix - v.px >= v.w) 0 else source[ix - v.px];
                    }
                }
            }
        }
    }
}

const BiasLayout = enum { nchw, nhwc };

fn addBias(par: kernels.Parallel, out: []f32, bias: []const f32, channels: usize, comptime layout: BiasLayout) void {
    const BiasJob = struct {
        out: []f32,
        bias: []const f32,
        channels: usize,
        positions: usize,

        fn run(job: @This(), lo: usize, hi: usize) void {
            if (layout == .nchw) {
                for (lo..hi) |channel| {
                    for (job.out[channel * job.positions ..][0..job.positions]) |*v| v.* += job.bias[channel];
                }
                return;
            }
            var channel = lo % job.channels;
            for (job.out[lo..hi]) |*v| {
                v.* += job.bias[channel];
                channel += 1;
                if (channel == job.channels) channel = 0;
            }
        }
    };
    const positions = if (layout == .nchw) out.len / channels else 1;
    const job = BiasJob{ .out = out, .bias = bias, .channels = channels, .positions = positions };
    switch (layout) {
        .nchw => kernels.parallelFor(par, channels, @max(1, 8192 / positions), job, BiasJob.run),
        .nhwc => kernels.parallelFor(par, out.len, 1 << 14, job, BiasJob.run),
    }
}

pub fn convolve(a: Allocator, par: kernels.Parallel, node: Node, x: Array, w: Array, bias: ?Array) Error!Array {
    if (x.dims.len != 4 or w.dims.len != 4) return error.UnsupportedOp;
    const n = x.dims[0];
    const c = x.dims[1];
    const h = x.dims[2];
    const width = x.dims[3];
    const m = w.dims[0];
    const kh = w.dims[2];
    const kw = w.dims[3];
    const groups: usize = @intCast(node.int("group", 1));
    const per_group = c / groups;
    const strides = node.ints("strides");
    const pads = node.ints("pads");
    const dilations = node.ints("dilations");
    const sy = pair(strides, 0, 1);
    const sx = pair(strides, 1, 1);
    const py = pair(pads, 0, 0);
    const px = pair(pads, 1, 0);
    const dy = pair(dilations, 0, 1);
    const dx = pair(dilations, 1, 1);
    const oh = (h + 2 * py - dy * (kh - 1) - 1) / sy + 1;
    const ow = (width + 2 * px - dx * (kw - 1) - 1) / sx + 1;
    if (w.dims[1] != per_group) return error.ShapeMismatch;
    const out = try make(a, &.{ n, m, oh, ow });
    if (groups == 1) {
        const taps = c * kh * kw;
        const positions = oh * ow;
        const col = try a.alloc(f32, taps * positions);
        const work = try a.alloc(f32, kernels.matmulTiledWorkLen(f32, .{}, par, m, taps, positions));
        const view: Window = .{ .c = c, .h = h, .w = width, .kh = kh, .kw = kw, .sy = sy, .sx = sx, .py = py, .px = px, .dy = dy, .dx = dx, .oh = oh, .ow = ow };
        for (0..n) |b| {
            kernels.parallelFor(par, c, @max(1, 4096 / (kh * kw * positions)), ColumnJob{ .x = x.data[b * c * h * width ..], .col = col, .v = view }, planarColumns);
            const image = out.data[b * m * positions ..][0 .. m * positions];
            kernels.matmulTiled(f32, .{}, par, .{ .m = m, .k = taps, .n = positions, .a = w.data, .b = col, .out = image }, work);
            const extra = bias orelse continue;
            addBias(par, image, extra.data, m, .nchw);
        }
        return out;
    }

    const Job = struct {
        x: []const f32,
        w: []const f32,
        bias: ?[]const f32,
        out: []f32,
        dims: [16]usize,
    };
    const job: Job = .{ .x = x.data, .w = w.data, .bias = if (bias) |b| b.data else null, .out = out.data, .dims = .{ n, c, h, width, m, kh, kw, groups, sy, sx, py, px, dy, dx, oh, ow } };
    const body = struct {
        fn run(j: Job, lo: usize, hi: usize) void {
            _, const cc, const hh, const ww, const mm, const khh, const kww, const gg, const ssy, const ssx, const ppy, const ppx, const ddy, const ddx, const ooh, const oow = j.dims;
            const pg = cc / gg;
            const m_per_group = mm / gg;
            for (lo..hi) |row| {
                const b = row / (mm * ooh);
                const oc = row / ooh % mm;
                const oy = row % ooh;
                const g = oc / m_per_group;
                for (0..oow) |ox| {
                    var total: f32 = if (j.bias) |bias_data| bias_data[oc] else 0;
                    for (0..pg) |ic| {
                        const channel = g * pg + ic;
                        for (0..khh) |ky| {
                            const iy = oy * ssy + ky * ddy;
                            if (iy < ppy or iy - ppy >= hh) continue;
                            for (0..kww) |kx| {
                                const ix = ox * ssx + kx * ddx;
                                if (ix < ppx or ix - ppx >= ww) continue;
                                total += j.x[((b * cc + channel) * hh + iy - ppy) * ww + ix - ppx] * j.w[((oc * pg + ic) * khh + ky) * kww + kx];
                            }
                        }
                    }
                    j.out[((b * mm + oc) * ooh + oy) * oow + ox] = total;
                }
            }
        }
    }.run;
    kernels.parallelFor(par, n * m * oh, 8, job, body);
    return out;
}

// Rearranges OIHW filters to (kh * kw * c) x m, the (ky, kx, ic) tap order the chunked convolution multiplies.
pub fn reorderOihw(a: Allocator, par: kernels.Parallel, w: Array) Error![]f32 {
    const m = w.dims[0];
    const c = w.dims[1];
    const khkw = w.dims[2] * w.dims[3];
    const wt = try a.alloc(f32, w.len());
    const Job = struct { w: []const f32, wt: []f32, c: usize, khkw: usize, m: usize };
    kernels.parallelFor(par, c * khkw, 1, Job{ .w = w.data, .wt = wt, .c = c, .khkw = khkw, .m = m }, struct {
        fn run(job: Job, lo: usize, hi: usize) void {
            for (lo..hi) |item| {
                const tap = item / job.c;
                const ic = item % job.c;
                const dst = job.wt[item * job.m ..][0..job.m];
                for (dst, 0..) |*v, oc| v.* = job.w[(oc * job.c + ic) * job.khkw + tap];
            }
        }
    }.run);
    return wt;
}

/// Runs one group-free convolution directly on NHWC activations, the layout the exported graph keeps between the
/// layout transposes; the chunked convolution builds im2col a cache-sized piece at a time and the output is already
/// NHWC, so no transpose ever runs. Square kernels, strides, pads and dilations only, and grouped convolutions
/// take the NCHW path instead.
pub fn convolveNhwc(a: Allocator, par: kernels.Parallel, node: Node, x: Array, w: Array, bias: ?Array) Error!Array {
    if (x.dims.len != 4 or w.dims.len != 4) return error.UnsupportedOp;
    if (node.int("group", 1) != 1) return error.UnsupportedOp;
    const n = x.dims[0];
    const h = x.dims[1];
    const width = x.dims[2];
    const c = x.dims[3];
    const m = w.dims[0];
    const k = w.dims[2];
    if (w.dims[3] != k or w.dims[1] != c) return error.ShapeMismatch;
    const sy = pair(node.ints("strides"), 0, 1);
    const sx = pair(node.ints("strides"), 1, 1);
    const py = pair(node.ints("pads"), 0, 0);
    const px = pair(node.ints("pads"), 1, 0);
    const dy = pair(node.ints("dilations"), 0, 1);
    const dx = pair(node.ints("dilations"), 1, 1);
    if (sx != sy or px != py or dx != dy) return error.UnsupportedOp;
    // The dilated window must fit the padded input, otherwise the unsigned output size below would underflow.
    if (dy * (k - 1) + 1 > h + 2 * py or dx * (k - 1) + 1 > width + 2 * px) return error.ShapeMismatch;
    const oh = (h + 2 * py - dy * (k - 1) - 1) / sy + 1;
    const ow = (width + 2 * px - dx * (k - 1) - 1) / sx + 1;
    const out = try make(a, &.{ n, oh, ow, m });
    const wt = try reorderOihw(a, par, w);
    const d: cv.Dims = .{ .n = @intCast(n), .h = @intCast(h), .w = @intCast(width), .c = @intCast(c), .k = @intCast(k), .stride = @intCast(sy), .pad = @intCast(py), .dilation = @intCast(dy), .c_out = @intCast(m) };
    try cpu_ops.convForward(a, par, x.data, wt, out.data, d);
    if (bias) |extra| addBias(par, out.data, extra.data, m, .nhwc);
    return out;
}

const PoolJob = struct { x: []const f32, out: []f32, v: Window, is_max: bool };

// Pools the planes (image and channel pairs) lo to hi.
fn poolPlanes(job: PoolJob, lo: usize, hi: usize) void {
    const v = job.v;
    for (lo..hi) |plane| {
        for (0..v.oh) |oy| {
            for (0..v.ow) |ox| {
                var best: f32 = -std.math.inf(f32);
                var total: f32 = 0;
                var count: usize = 0;
                for (0..v.kh) |i| {
                    const iy = oy * v.sy + i;
                    if (iy < v.py or iy - v.py >= v.h) continue;
                    for (0..v.kw) |j| {
                        const ix = ox * v.sx + j;
                        if (ix < v.px or ix - v.px >= v.w) continue;
                        const value = job.x[(plane * v.h + iy - v.py) * v.w + ix - v.px];
                        best = @max(best, value);
                        total += value;
                        count += 1;
                    }
                }
                job.out[(plane * v.oh + oy) * v.ow + ox] = if (job.is_max) best else total / @as(f32, @floatFromInt(@max(count, 1)));
            }
        }
    }
}

/// Max or average pooling of an NCHW tensor (count_include_pad off), one plane per work item on the worker pool.
pub fn pool(a: Allocator, par: kernels.Parallel, node: Node, x: Array, is_max: bool) Error!Array {
    if (x.dims.len != 4) return error.UnsupportedOp;
    const kernel = node.ints("kernel_shape");
    const strides = node.ints("strides");
    const pads = node.ints("pads");
    const ky = pair(kernel, 0, 1);
    const kx = pair(kernel, 1, 1);
    const sy = pair(strides, 0, 1);
    const sx = pair(strides, 1, 1);
    const py = pair(pads, 0, 0);
    const px = pair(pads, 1, 0);
    const n = x.dims[0];
    const c = x.dims[1];
    const h = x.dims[2];
    const w = x.dims[3];
    const oh = (h + 2 * py - ky) / sy + 1;
    const ow = (w + 2 * px - kx) / sx + 1;
    const out = try make(a, &.{ n, c, oh, ow });
    const view: Window = .{ .c = c, .h = h, .w = w, .kh = ky, .kw = kx, .sy = sy, .sx = sx, .py = py, .px = px, .dy = 1, .dx = 1, .oh = oh, .ow = ow };
    kernels.parallelFor(par, n * c, @max(1, 8192 / @max(1, oh * ow * ky * kx)), PoolJob{ .x = x.data, .out = out.data, .v = view, .is_max = is_max }, poolPlanes);
    return out;
}

fn globalAverage(a: Allocator, x: Array) Error!Array {
    const planes = x.dims[0] * x.dims[1];
    const size = product(x.dims[2..]);
    const out = try make(a, &.{ x.dims[0], x.dims[1], 1, 1 });
    for (0..planes) |p| {
        var total: f32 = 0;
        for (x.data[p * size ..][0..size]) |v| total += v;
        out.data[p] = total / @as(f32, @floatFromInt(size));
    }
    return out;
}

fn batchNorm(a: Allocator, node: Node, x: Array, scale: Array, shift: Array, mean: Array, variance: Array) Error!Array {
    const eps = node.float("epsilon", 1e-5);
    const c = x.dims[1];
    const inner = product(x.dims[2..]);
    const out = try make(a, x.dims);
    for (0..x.dims[0]) |b| {
        for (0..c) |ch| {
            const s = scale.data[ch] / @sqrt(variance.data[ch] + eps);
            const t = shift.data[ch] - mean.data[ch] * s;
            const base = (b * c + ch) * inner;
            for (out.data[base..][0..inner], x.data[base..][0..inner]) |*o, v| o.* = v * s + t;
        }
    }
    return out;
}

pub fn reshape(a: Allocator, x: Array, shape: Array) Error!Array {
    if (shape.ints.len == 0) return error.UnsupportedOp;
    const dims = try a.alloc(usize, shape.ints.len);
    var known: usize = 1;
    var unknown: ?usize = null;
    for (shape.ints, dims, 0..) |s, *d, i| {
        if (s == -1) {
            unknown = i;
            d.* = 1;
        } else {
            d.* = if (s == 0) x.dims[i] else @intCast(s);
            known *= d.*;
        }
    }
    if (unknown) |i| dims[i] = x.len() / known;
    if (product(dims) != x.len()) return error.ShapeMismatch;
    return .{ .dims = dims, .data = x.data };
}

const Groups = struct { lead: usize, middle: usize };

// Recognizes a permutation that keeps axes 0 to lead - 1 and swaps two contiguous groups of the remaining axes, lead to
// middle - 1 and middle to the end: per leading index, that is the transpose of a (rows, cols) matrix.
fn swappedGroups(perm: []const usize) ?Groups {
    var lead: usize = 0;
    while (lead < perm.len and perm[lead] == lead) lead += 1;
    if (lead == perm.len) return null;
    const middle = perm[lead];
    for (perm[lead..], 0..) |p, i| {
        const want = if (i < perm.len - middle) middle + i else lead + i - (perm.len - middle);
        if (p != want) return null;
    }
    return .{ .lead = lead, .middle = middle };
}

const TransposeJob = struct { x: []const f32, out: []f32, rows: usize, cols: usize };

// Transposes bands of 32 rows, matrix by matrix; work item i is band i % bands of matrix i / bands.
fn transposeBlocks(job: TransposeJob, lo: usize, hi: usize) void {
    const bands = (job.rows + 31) / 32;
    const size = job.rows * job.cols;
    for (lo..hi) |item| {
        const src = job.x[item / bands * size ..][0..size];
        const dst = job.out[item / bands * size ..][0..size];
        const r0 = item % bands * 32;
        const r1 = @min(r0 + 32, job.rows);
        var c0: usize = 0;
        while (c0 < job.cols) : (c0 += 32) {
            const c1 = @min(c0 + 32, job.cols);
            for (r0..r1) |r| {
                for (c0..c1) |col| dst[col * job.rows + r] = src[r * job.cols + col];
            }
        }
    }
}

/// Permutes the axes of x by the perm attribute (reversed when absent); two swapped groups of axes take a blocked matrix transpose.
pub fn transpose(a: Allocator, par: kernels.Parallel, node: Node, x: Array) Error!Array {
    const rank = x.dims.len;
    var perm: [max_rank]usize = undefined;
    const given = node.ints("perm");
    for (0..rank) |i| perm[i] = if (given.len == rank) @intCast(given[i]) else rank - 1 - i;
    var dims: [max_rank]usize = undefined;
    var source_strides: [max_rank]usize = undefined;
    var stride: usize = 1;
    var i = rank;
    while (i > 0) {
        i -= 1;
        source_strides[i] = stride;
        stride *= x.dims[i];
    }
    for (0..rank) |d| dims[d] = x.dims[perm[d]];
    const out = try make(a, dims[0..rank]);
    if (swappedGroups(perm[0..rank])) |split| {
        // NOTE: the exporter wraps every convolution and pooling in NHWC to NCHW transposes and back; both are this case.
        const job: TransposeJob = .{ .x = x.data, .out = out.data, .rows = product(x.dims[split.lead..split.middle]), .cols = product(x.dims[split.middle..]) };
        const blocks = product(x.dims[0..split.lead]) * ((job.rows + 31) / 32);
        kernels.parallelFor(par, blocks, @max(1, 8192 / (32 * @max(1, job.cols))), job, transposeBlocks);
        return out;
    }
    var index: [max_rank]usize = @splat(0);
    for (out.data) |*v| {
        var offset: usize = 0;
        for (0..rank) |d| offset += index[d] * source_strides[perm[d]];
        v.* = x.data[offset];
        var d = rank;
        while (d > 0) {
            d -= 1;
            index[d] += 1;
            if (index[d] < dims[d]) break;
            index[d] = 0;
        }
    }
    return out;
}

fn concat(a: Allocator, node: Node, values: *std.StringHashMap(Array)) Error!Array {
    const first = try input(values, node, 0);
    const axis = normalizeAxis(node.int("axis", 0), first.dims.len);
    var total: usize = 0;
    for (0..node.inputs.len) |i| total += (try input(values, node, i)).dims[axis];
    const dims = try a.dupe(usize, first.dims);
    dims[axis] = total;
    const outer = product(first.dims[0..axis]);
    const inner = product(first.dims[axis + 1 ..]);
    const out: Array = .{ .dims = dims, .data = try a.alloc(f32, product(dims)) };
    var offset: usize = 0;
    for (0..node.inputs.len) |i| {
        const part = try input(values, node, i);
        const width = part.dims[axis] * inner;
        for (0..outer) |o| @memcpy(out.data[o * total * inner + offset ..][0..width], part.data[o * width ..][0..width]);
        offset += width;
    }
    return out;
}

pub fn softmax(a: Allocator, node: Node, x: Array) Error!Array {
    const axis = normalizeAxis(node.int("axis", -1), x.dims.len);
    const outer = product(x.dims[0..axis]);
    const size = x.dims[axis];
    const inner = product(x.dims[axis + 1 ..]);
    const out = try make(a, x.dims);
    for (0..outer) |o| {
        for (0..inner) |i| {
            var largest: f32 = -std.math.inf(f32);
            for (0..size) |s| largest = @max(largest, x.data[(o * size + s) * inner + i]);
            var total: f32 = 0;
            for (0..size) |s| {
                const e = @exp(x.data[(o * size + s) * inner + i] - largest);
                out.data[(o * size + s) * inner + i] = e;
                total += e;
            }
            for (0..size) |s| out.data[(o * size + s) * inner + i] /= total;
        }
    }
    return out;
}

// Picks rows of data along axis 0. NOTE: indices are held as f32 like every other array, which is exact below 2^24.
fn gather(a: Allocator, node: Node, data: Array, indices: Array) Error!Array {
    if (node.int("axis", 0) != 0 or data.dims.len < 1) return error.UnsupportedOp;
    const inner = product(data.dims[1..]);
    const dims = try a.alloc(usize, indices.dims.len + data.dims.len - 1);
    @memcpy(dims[0..indices.dims.len], indices.dims);
    @memcpy(dims[indices.dims.len..], data.dims[1..]);
    const out: Array = .{ .dims = dims, .data = try a.alloc(f32, product(dims)) };
    for (indices.data, 0..) |raw, i| {
        const row: usize = @intFromFloat(raw);
        if (row >= data.dims[0]) return error.ShapeMismatch;
        @memcpy(out.data[i * inner ..][0..inner], data.data[row * inner ..][0..inner]);
    }
    return out;
}

// Normalizes over the last axis with the population variance, as the engine does.
fn layerNorm(a: Allocator, node: Node, x: Array, scale: Array, bias: ?Array) Error!Array {
    if (normalizeAxis(node.int("axis", -1), x.dims.len) != x.dims.len - 1) return error.UnsupportedOp;
    const eps = node.float("epsilon", 1e-5);
    const cols = x.dims[x.dims.len - 1];
    if (scale.len() != cols) return error.ShapeMismatch;
    const out = try make(a, x.dims);
    for (0..x.len() / cols) |r| {
        const row = x.data[r * cols ..][0..cols];
        var mean: f32 = 0;
        for (row) |v| mean += v;
        mean /= @floatFromInt(cols);
        var square: f32 = 0;
        for (row) |v| square += (v - mean) * (v - mean);
        const inverse = 1 / @sqrt(square / @as(f32, @floatFromInt(cols)) + eps);
        for (out.data[r * cols ..][0..cols], row, 0..) |*o, v, j| o.* = scale.data[j] * (v - mean) * inverse + (if (bias) |b| b.data[j] else 0);
    }
    return out;
}

// Cuts a range out of some axes (default: the first ones) with the given steps; negative and oversized bounds are clamped as in ONNX.
fn slice(a: Allocator, x: Array, starts: Array, ends: Array, axes: ?Array, steps: ?Array) Error!Array {
    const rank = x.dims.len;
    var first: [max_rank]usize = @splat(0);
    var step: [max_rank]usize = @splat(1);
    var dims: [max_rank]usize = undefined;
    @memcpy(dims[0..rank], x.dims);
    for (0..starts.ints.len) |k| {
        const axis = if (axes) |given| normalizeAxis(given.ints[k], rank) else k;
        const stride: i64 = if (steps) |given| given.ints[k] else 1;
        if (stride < 1) return error.UnsupportedOp;
        const size: i64 = @intCast(x.dims[axis]);
        const lo = std.math.clamp(if (starts.ints[k] < 0) starts.ints[k] + size else starts.ints[k], 0, size);
        const hi = std.math.clamp(if (ends.ints[k] < 0) ends.ints[k] + size else ends.ints[k], 0, size);
        first[axis] = @intCast(lo);
        step[axis] = @intCast(stride);
        dims[axis] = if (hi > lo) @intCast(@divTrunc(hi - lo + stride - 1, stride)) else 0;
    }
    const out = try make(a, dims[0..rank]);
    var source_strides: [max_rank]usize = undefined;
    var stride: usize = 1;
    var i = rank;
    while (i > 0) {
        i -= 1;
        source_strides[i] = stride;
        stride *= x.dims[i];
    }
    var index: [max_rank]usize = @splat(0);
    for (out.data) |*v| {
        var offset: usize = 0;
        for (0..rank) |d| offset += (first[d] + index[d] * step[d]) * source_strides[d];
        v.* = x.data[offset];
        var d = rank;
        while (d > 0) {
            d -= 1;
            index[d] += 1;
            if (index[d] < dims[d]) break;
            index[d] = 0;
        }
    }
    return out;
}
