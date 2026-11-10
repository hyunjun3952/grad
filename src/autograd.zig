const std = @import("std");
const Allocator = std.mem.Allocator;
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const kernels = @import("kernels.zig");
const gpu_mod = @import("gpu/gpu.zig");
const Tensor = tensor.Tensor;

pub const Error = gpu_mod.Error;

fn pad4(comptime dims: []const usize, comptime fill: u32) [4]u32 {
    if (dims.len > 4) @compileError("gpu kernels support tensors of rank 4 or less");
    var out: [4]u32 = @splat(fill);
    for (dims, 0..) |d, i| out[4 - dims.len + i] = @intCast(d);
    return out;
}

fn rowStrides4(comptime dims: []const usize) [4]u32 {
    var out: [4]u32 = @splat(0);
    var acc: u32 = 1;
    var i: usize = dims.len;
    while (i > 0) {
        i -= 1;
        out[4 - dims.len + i] = acc;
        acc *= @intCast(dims[i]);
    }
    return out;
}

fn reducedAxes(comptime target: []const usize) u32 {
    var mask: u32 = 0;
    for (pad4(target, 1), 0..) |d, axis| {
        if (d == 1) mask |= @as(u32, 1) << @intCast(axis);
    }
    return mask;
}

pub fn Var(comptime Tn: type) type {
    return struct { value: Tn, grad: Tn, needs_grad: bool };
}

const Node = struct {
    ctx: *anyopaque,
    run: *const fn (*Tape, *anyopaque) Error!void,
};

pub const Tape = struct {
    gpa: Allocator,
    par: kernels.Parallel = .{},
    nodes: std.ArrayList(Node) = .empty,

    fn fresh(self: *Tape, comptime Tn: type) Error!Tn {
        if (self.par.gpu) |dev| return .{ .data = try dev.scratch(Tn.Elem, Tn.len) };
        return Tn.empty(self.gpa);
    }

    fn track(self: *Tape, value: anytype, needs_grad: bool) Error!Var(@TypeOf(value)) {
        const Tn = @TypeOf(value);
        var grad: Tn = undefined;
        if (needs_grad) {
            grad = try self.fresh(Tn);
            if (self.par.gpu) |dev| try dev.fill(grad.data, 0) else @memset(grad.data, 0);
        }
        return .{ .value = value, .grad = grad, .needs_grad = needs_grad };
    }

    pub fn scalar(self: *Tape, v: anytype) Error!@TypeOf(v.value).Elem {
        if (self.par.gpu) |dev| return dev.scalar(v.value.data);
        return v.value.data[0];
    }

    fn binary(self: *Tape, comptime op: gpu_mod.BinaryOp, a: anytype, b: anytype) Error!ops.BroadcastResult(@TypeOf(a), @TypeOf(b)) {
        if (self.par.gpu) |dev| {
            const out = try self.fresh(ops.BroadcastResult(@TypeOf(a), @TypeOf(b)));
            const bc = comptime ops.broadcast(@TypeOf(a).shape, @TypeOf(b).shape);
            try dev.binary(op, a.data, b.data, out.data, pad4(&bc.dims, 1), pad4(&bc.a, 0), pad4(&bc.b, 0));
            return out;
        }
        return switch (op) {
            .add => ops.add(self.gpa, a, b),
            .mul => ops.mul(self.gpa, a, b),
        };
    }

    pub fn leaf(self: *Tape, value: anytype) Error!Var(@TypeOf(value)) {
        return self.track(value, true);
    }

    pub fn constant(self: *Tape, value: anytype) Error!Var(@TypeOf(value)) {
        return self.track(value, false);
    }

    fn push(self: *Tape, ctx: anytype, comptime run: fn (*Tape, *const @TypeOf(ctx)) Error!void) Error!void {
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

    fn accumulate(self: *Tape, v: anytype, g: anytype) Error!void {
        if (!v.needs_grad) return;
        const D = @TypeOf(v.grad);
        const G = @TypeOf(g);
        if (self.par.gpu) |dev| {
            if (comptime std.mem.eql(usize, D.shape, G.shape)) {
                try dev.addInPlace(v.grad.data, g.data);
            } else {
                try dev.sumTo(g.data, v.grad.data, pad4(G.shape, 1), rowStrides4(G.shape), comptime reducedAxes(D.shape), true);
            }
            return;
        }
        if (comptime std.mem.eql(usize, D.shape, G.shape)) {
            for (v.grad.data, g.data) |*d, x| d.* += x;
        } else {
            const r = try ops.sumTo(self.gpa, g, D.shape);
            for (v.grad.data, r.data) |*d, x| d.* += x;
        }
    }

    pub fn add(self: *Tape, a: anytype, b: anytype) Error!Var(ops.BroadcastResult(@TypeOf(a.value), @TypeOf(b.value))) {
        const out = try self.track(try self.binary(.add, a.value, b.value), a.needs_grad or b.needs_grad);
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.push(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                try t.accumulate(c.a, c.out.grad);
                try t.accumulate(c.b, c.out.grad);
            }
        }.run);
        return out;
    }

    pub fn mul(self: *Tape, a: anytype, b: anytype) Error!Var(ops.BroadcastResult(@TypeOf(a.value), @TypeOf(b.value))) {
        const out = try self.track(try self.binary(.mul, a.value, b.value), a.needs_grad or b.needs_grad);
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.push(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (c.a.needs_grad) try t.accumulate(c.a, try t.binary(.mul, c.out.grad, c.b.value));
                if (c.b.needs_grad) try t.accumulate(c.b, try t.binary(.mul, c.out.grad, c.a.value));
            }
        }.run);
        return out;
    }

    pub fn matmul(self: *Tape, a: anytype, b: anytype) Error!Var(tensor.MatmulResult(@TypeOf(a.value), @TypeOf(b.value), .{})) {
        const value = try self.fresh(tensor.MatmulResult(@TypeOf(a.value), @TypeOf(b.value), .{}));
        try tensor.matmulInto(self.gpa, self.par, a.value, b.value, value, .{});
        const out = try self.track(value, a.needs_grad or b.needs_grad);
        const Ctx = struct { a: @TypeOf(a), b: @TypeOf(b), out: @TypeOf(out) };
        if (out.needs_grad) try self.push(Ctx{ .a = a, .b = b, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (c.a.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.out.grad, c.b.value, c.a.grad, .{ .trans_b = true, .accumulate = true });
                if (c.b.needs_grad) try tensor.matmulInto(t.gpa, t.par, c.a.value, c.out.grad, c.b.grad, .{ .trans_a = true, .accumulate = true });
            }
        }.run);
        return out;
    }

    pub fn relu(self: *Tape, x: anytype) Error!Var(@TypeOf(x.value)) {
        var value: @TypeOf(x.value) = undefined;
        if (self.par.gpu) |dev| {
            value = try self.fresh(@TypeOf(x.value));
            try dev.relu(x.value.data, value.data);
        } else {
            value = try ops.relu(self.gpa, x.value);
        }
        const out = try self.track(value, x.needs_grad);
        const Ctx = struct { x: @TypeOf(x), out: @TypeOf(out) };
        if (out.needs_grad) try self.push(Ctx{ .x = x, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.reluBackward(c.x.value.data, c.out.grad.data, c.x.grad.data);
                for (c.x.grad.data, c.x.value.data, c.out.grad.data) |*gx, v, g| {
                    gx.* += if (v > 0) g else 0;
                }
            }
        }.run);
        return out;
    }

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
            logp = try self.fresh(L);
            value = try self.fresh(Scalar);
            try dev.crossEntropy(logits.value.data, logp.data, labels, value.data, rows, cols);
        } else {
            logp = try ops.logSoftmax(self.gpa, logits.value);
            var total: f64 = 0;
            for (labels, 0..) |y, i| total -= logp.data[i * cols + y];
            value = try Scalar.empty(self.gpa);
            value.data[0] = @floatCast(total / rows);
        }
        const out = try self.track(value, logits.needs_grad);

        const Ctx = struct { logits: @TypeOf(logits), logp: L, labels: []const u8, out: @TypeOf(out) };
        if (out.needs_grad) try self.push(Ctx{ .logits = logits, .logp = logp, .labels = labels, .out = out }, struct {
            fn run(t: *Tape, c: *const Ctx) Error!void {
                if (t.par.gpu) |dev| return dev.crossEntropyBackward(c.logp.data, c.labels, c.logits.grad.data, c.out.grad.data, rows, cols);
                const scale = c.out.grad.data[0] / rows;
                for (0..rows) |i| {
                    for (0..cols) |j| {
                        var p = @exp(c.logp.data[i * cols + j]);
                        if (j == c.labels[i]) p -= 1;
                        c.logits.grad.data[i * cols + j] += p * scale;
                    }
                }
            }
        }.run);
        return out;
    }

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

const X = Tensor(f32, &.{ 4, 3 });
const W1 = Tensor(f32, &.{ 3, 5 });
const B1 = Tensor(f32, &.{5});
const S = Tensor(f32, &.{ 1, 5 });
const W2 = Tensor(f32, &.{ 5, 3 });

const Params = struct { x: X, w1: W1, b1: B1, s: S, w2: W2 };
const Run = struct {
    loss: Var(Tensor(f32, &.{})),
    x: Var(X),
    w1: Var(W1),
    b1: Var(B1),
    s: Var(S),
    w2: Var(W2),
};

const test_labels = [_]u8{ 0, 2, 1, 2 };

fn forward(t: *Tape, p: Params, x_grad: bool, labels: []const u8) !Run {
    const x = if (x_grad) try t.leaf(p.x) else try t.constant(p.x);
    const w1 = try t.leaf(p.w1);
    const b1 = try t.leaf(p.b1);
    const s = try t.leaf(p.s);
    const w2 = try t.leaf(p.w2);

    const h = try t.relu(try t.add(try t.matmul(x, w1), b1));
    const logits = try t.matmul(try t.mul(h, s), w2);
    const loss = try t.crossEntropy(logits, labels);
    return .{ .loss = loss, .x = x, .w1 = w1, .b1 = b1, .s = s, .w2 = w2 };
}

fn lossOf(p: Params) !f32 {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator() };
    return (try forward(&t, p, true, &test_labels)).loss.value.data[0];
}

fn minHiddenMagnitude(p: Params) f32 {
    var smallest: f32 = std.math.floatMax(f32);
    for (0..4) |r| {
        for (0..5) |c| {
            var pre = p.b1.data[c];
            for (0..3) |k| pre += p.x.data[r * 3 + k] * p.w1.data[k * 5 + c];
            smallest = @min(smallest, @abs(pre));
        }
    }
    return smallest;
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
    var p: Params = undefined;
    inline for (.{ "x", "w1", "b1", "s", "w2" }) |name| {
        @field(p, name) = try @FieldType(Params, name).empty(gpa);
        for (@field(p, name).data) |*v| v.* = rng.float(f32) * 2 - 1;
    }

    var with_x: Tape = .{ .gpa = gpa };
    const a = try forward(&with_x, p, true, &test_labels);
    try with_x.backward(a.loss);
    var without_x: Tape = .{ .gpa = gpa };
    const b = try forward(&without_x, p, false, &test_labels);
    try without_x.backward(b.loss);

    try testing.expect(!b.x.needs_grad);
    try testing.expect(without_x.nodes.items.len == with_x.nodes.items.len);
    inline for (.{ "w1", "b1", "s", "w2" }) |name| {
        try testing.expectEqualSlices(f32, @field(a, name).grad.data, @field(b, name).grad.data);
    }
}

test "gpu tape reproduces the cpu loss and every gradient" {
    const gpu = gpu_mod.Gpu.init(testing.allocator, 64 << 20) catch |err| switch (err) {
        error.DriverNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer gpu.deinit();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var prng = std.Random.DefaultPrng.init(21);
    const rng = prng.random();
    const names = .{ "x", "w1", "b1", "s", "w2" };
    var host: Params = undefined;
    inline for (names) |name| @field(host, name) = try @FieldType(Params, name).empty(gpa);
    while (true) {
        inline for (names) |name| {
            for (@field(host, name).data) |*v| v.* = rng.float(f32) * 2 - 1;
        }
        if (minHiddenMagnitude(host) > 0.05) break;
    }

    var cpu_tape: Tape = .{ .gpa = gpa };
    const cpu = try forward(&cpu_tape, host, true, &test_labels);
    try cpu_tape.backward(cpu.loss);

    var device: Params = undefined;
    inline for (names) |name| {
        const slice = try gpu.alloc(f32, @field(host, name).data.len);
        try gpu.upload(slice, @field(host, name).data);
        @field(device, name) = .{ .data = slice };
    }
    const labels = try gpu.alloc(u8, test_labels.len);
    try gpu.upload(labels, &test_labels);

    var gpu_tape: Tape = .{ .gpa = gpa, .par = .{ .gpu = gpu } };
    const run = try forward(&gpu_tape, device, true, labels);
    try gpu_tape.backward(run.loss);

    try testing.expectApproxEqAbs(cpu.loss.value.data[0], try gpu_tape.scalar(run.loss), 1e-5);
    inline for (names) |name| {
        const want = @field(cpu, name).grad.data;
        const got = try gpa.alloc(f32, want.len);
        try gpu.download(got, @field(run, name).grad.data);
        for (want, got) |w, g| try testing.expectApproxEqAbs(w, g, 1e-5);
    }
}

test "gradients of a small network match finite differences" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const names = .{ "x", "w1", "b1", "s", "w2" };
    var p: Params = undefined;
    inline for (names) |name| @field(p, name) = try @FieldType(Params, name).empty(gpa);
    while (true) {
        inline for (names) |name| {
            for (@field(p, name).data) |*v| v.* = rng.float(f32) * 2 - 1;
        }
        if (minHiddenMagnitude(p) > 0.05) break;
    }

    var t: Tape = .{ .gpa = gpa };
    const run = try forward(&t, p, true, &test_labels);
    try t.backward(run.loss);

    const eps = 5e-3;
    inline for (names) |name| {
        for (@field(p, name).data, 0..) |*v, i| {
            const old = v.*;
            v.* = old + eps;
            const plus = try lossOf(p);
            v.* = old - eps;
            const minus = try lossOf(p);
            v.* = old;
            const numeric = (plus - minus) / (2 * eps);
            try testing.expectApproxEqAbs(numeric, @field(run, name).grad.data[i], 3e-3);
        }
    }
}
