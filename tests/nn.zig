const std = @import("std");
const lib = @import("grad");
const params = lib.nn.params;
const Conv2d = lib.nn.Conv2d;
const BatchNorm = lib.nn.BatchNorm;
const state = lib.nn.state;
const initAll = lib.nn.initAll;
const Allocator = std.mem.Allocator;
const Gru = lib.nn.Gru;
const TransformerBlock = lib.nn.TransformerBlock;
const Tape = autograd.Tape;
const ConvBn = lib.nn.ConvBn;
const optim = lib.optim;
const Tensor = lib.tensor.Tensor;
const Linear = lib.nn.Linear;
const Gpu = lib.gpu.Gpu;
const deinitAll = lib.nn.deinitAll;
const autograd = lib.autograd;

const testing = std.testing;

test "layer tuples initialise, concatenate parameters and state in order, and free everything" {
    const Layers = struct { Linear(2, 3), BatchNorm(3), Conv2d(1, 2, 3, .{}), ConvBn(2, 3, 1, .{}) };
    var prng = std.Random.DefaultPrng.init(8);
    const layers = try initAll(Layers, testing.allocator, prng.random());
    defer deinitAll(layers, testing.allocator);

    const all_params = params(layers);
    try testing.expectEqual(@as(usize, 9), all_params.len);
    try testing.expectEqual(layers[0].w.value.data.ptr, all_params[0].value.data.ptr);
    try testing.expectEqual(layers[1].gamma.value.data.ptr, all_params[2].value.data.ptr);
    try testing.expectEqual(layers[2].inner.b.value.data.ptr, all_params[5].value.data.ptr);

    const all_state = state(layers);
    try testing.expectEqual(@as(usize, 13), all_state.len);
    try testing.expectEqual(layers[1].variance.data.ptr, all_state[5].ptr);
    try testing.expectEqual(@as(usize, 2 * 3), all_state[0].len);
}

test "a gru with zero weights halves the state, and its gradients match finite differences and the gpu" {
    const gpa = testing.allocator;
    const Cell = Gru(3, 4);
    var prng = std.Random.DefaultPrng.init(5);
    const cell = try Cell.init(gpa, prng.random());
    defer cell.deinit(gpa);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const x_values = [_]f32{ 0.5, -1, 0.25, 1, 0.3, -0.7 };
    const h_values = [_]f32{ 0.2, -0.4, 0.6, 0.1, -0.3, 0.5, 0.9, -0.8 };
    const target_values = [_]f32{ 0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.0 };

    {
        var zeroed = cell;
        inline for (@typeInfo(Cell).@"struct".field_names) |name| {
            @memset(@field(zeroed, name).w.value.data, 0);
        }
        var tape: Tape = .{ .gpa = arena.allocator() };
        const h = try tape.constant(try Tensor(f32, &.{ 2, 4 }).fromSlice(arena.allocator(), &h_values));
        const x = try tape.constant(try Tensor(f32, &.{ 2, 3 }).fromSlice(arena.allocator(), &x_values));
        const next = try zeroed.forward(&tape, x, h);
        for (next.value.data, h_values) |got, old| try testing.expectApproxEqAbs(0.5 * old, got, 1e-6);
    }

    const loss_at = struct {
        fn run(a: Allocator, c: Cell, h_data: []const f32) !f32 {
            var tape: Tape = .{ .gpa = a };
            const h = try tape.leaf(try Tensor(f32, &.{ 2, 4 }).fromSlice(a, h_data));
            const x = try tape.constant(try Tensor(f32, &.{ 2, 3 }).fromSlice(a, &x_values));
            const target = try tape.constant(try Tensor(f32, &.{ 2, 4 }).fromSlice(a, &target_values));
            const loss = try tape.mse(try c.forward(&tape, x, h), target);
            return loss.value.data[0];
        }
    }.run;

    var tape: Tape = .{ .gpa = arena.allocator() };
    const h = try tape.leaf(try Tensor(f32, &.{ 2, 4 }).fromSlice(arena.allocator(), &h_values));
    const x = try tape.constant(try Tensor(f32, &.{ 2, 3 }).fromSlice(arena.allocator(), &x_values));
    const target = try tape.constant(try Tensor(f32, &.{ 2, 4 }).fromSlice(arena.allocator(), &target_values));
    const loss = try tape.mse(try cell.forward(&tape, x, h), target);
    try tape.backward(loss);
    for (0..h_values.len) |i| {
        var plus = h_values;
        var minus = h_values;
        plus[i] += 1e-2;
        minus[i] -= 1e-2;
        const numeric = (try loss_at(arena.allocator(), cell, &plus) - try loss_at(arena.allocator(), cell, &minus)) / 2e-2;
        try testing.expectApproxEqAbs(numeric, h.grad.data[i], 2e-3);
    }

    const gpu = try Gpu.forTest(1 << 22);
    defer gpu.deinit();
    const on_gpu = try cell.toGpu(gpu);
    var gpu_tape: Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu } };
    const gpu_h = try gpu_tape.leaf(Tensor(f32, &.{ 2, 4 }){ .data = try gpu.alloc(f32, h_values.len) });
    try gpu.upload(gpu_h.value.data, &h_values);
    try gpu.fill(gpu_h.grad.data, 0);
    const gpu_x = try gpu_tape.constant(Tensor(f32, &.{ 2, 3 }){ .data = try gpu.alloc(f32, x_values.len) });
    try gpu.upload(gpu_x.value.data, &x_values);
    const gpu_target = try gpu_tape.constant(Tensor(f32, &.{ 2, 4 }){ .data = try gpu.alloc(f32, target_values.len) });
    try gpu.upload(gpu_target.value.data, &target_values);
    const gpu_loss = try gpu_tape.mse(try on_gpu.forward(&gpu_tape, gpu_x, gpu_h), gpu_target);
    try gpu_tape.backward(gpu_loss);
    var back: [8]f32 = undefined;
    try gpu.download(&back, gpu_h.grad.data);
    for (back, h.grad.data) |got, want| try testing.expectApproxEqAbs(want, got, 1e-5);
}

test "a failing layer initialisation releases the layers built before it" {
    var prng = std.Random.DefaultPrng.init(9);
    for (0..8) |fail_index| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = fail_index });
        try testing.expectError(error.OutOfMemory, initAll(struct { Linear(2, 3), Linear(3, 4) }, failing.allocator(), prng.random()));
    }
}

test "forward is x*w + b and params exposes both parameters" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(1);
    const layer = try Linear(2, 3).init(gpa, prng.random());
    defer layer.deinit(gpa);
    @memcpy(layer.w.value.data, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    @memcpy(layer.b.value.data, &[_]f32{ 10, 20, 30 });

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var tape: Tape = .{ .gpa = arena.allocator() };
    const x = try tape.constant(try Tensor(f32, &.{ 2, 2 }).fromSlice(arena.allocator(), &.{ 1, 1, 2, 0 }));
    const y = try layer.forward(&tape, x);

    try testing.expectEqual(@as(usize, 3), @TypeOf(y.value).shape[1]);
    try testing.expectEqualSlices(f32, &.{ 15, 27, 39, 12, 24, 36 }, y.value.data);
    try testing.expectEqual(@as(usize, 2), layer.params().len);
}

test "initial weights are zero-mean with variance 2/n_in" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(2);
    const layer = try Linear(400, 100).init(gpa, prng.random());
    defer layer.deinit(gpa);

    var sum: f64 = 0;
    var sq: f64 = 0;
    for (layer.w.value.data) |v| {
        sum += v;
        sq += @as(f64, v) * v;
    }
    const n: f64 = @floatFromInt(layer.w.value.data.len);
    try testing.expectApproxEqAbs(@as(f64, 0), sum / n, 0.01);
    try testing.expectApproxEqRel(@as(f64, 2.0 / 400.0), sq / n, 0.05);
    for (layer.b.value.data) |v| try testing.expectEqual(@as(f32, 0), v);
}

test "adam also trains the linear layer" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const layer = try Linear(2, 2).init(gpa, prng.random());
    defer layer.deinit(gpa);
    var adam = try optim.Adam.init(gpa, .{}, 0.1, layer.params());
    defer adam.deinit();

    const x = try Tensor(f32, &.{ 8, 2 }).fromSlice(gpa, &.{ 1, 0.5, 2, -1, 0.7, 1, 1.5, -0.3, -1, 0.5, -2, -1, -0.7, 1, -1.5, 0.3 });
    defer x.deinit(gpa);
    const labels = [_]u8{ 1, 1, 1, 1, 0, 0, 0, 0 };

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..60) |step| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var tape: Tape = .{ .gpa = arena.allocator() };

        try optim.zeroGrad(.{}, layer.params());
        const loss = try tape.crossEntropy(try layer.forward(&tape, try tape.constant(x)), &labels);
        try tape.backward(loss);
        try adam.step(.{}, layer.params());

        if (step == 0) first = loss.value.data[0];
        last = loss.value.data[0];
    }
    try testing.expect(last < first / 4);
}

test "a linear layer learns a separable problem" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const layer = try Linear(2, 2).init(gpa, prng.random());
    defer layer.deinit(gpa);

    const x = try Tensor(f32, &.{ 8, 2 }).fromSlice(gpa, &.{ 1, 0.5, 2, -1, 0.7, 1, 1.5, -0.3, -1, 0.5, -2, -1, -0.7, 1, -1.5, 0.3 });
    defer x.deinit(gpa);
    const labels = [_]u8{ 1, 1, 1, 1, 0, 0, 0, 0 };

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..200) |step| {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var tape: Tape = .{ .gpa = arena.allocator() };

        try optim.zeroGrad(.{}, layer.params());
        const logits = try layer.forward(&tape, try tape.constant(x));
        const loss = try tape.crossEntropy(logits, &labels);
        try tape.backward(loss);
        try optim.sgd(.{}, 0.5, layer.params());

        if (step == 0) first = loss.value.data[0];
        last = loss.value.data[0];
    }
    try testing.expect(last < first / 4);
}

test "a transformer block is causal, its input gradient matches finite differences, and the gpu agrees with the cpu" {
    const gpa = testing.allocator;
    const Block = TransformerBlock(8, 2, 4);
    const X = Tensor(f32, &.{ 8, 8 });
    var prng = std.Random.DefaultPrng.init(21);
    const block = try Block.init(gpa, prng.random());
    defer block.deinit(gpa);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var input: [64]f32 = undefined;
    for (&input) |*v| v.* = prng.random().floatNorm(f32);
    var targets: [64]f32 = undefined;
    for (&targets, 0..) |*v, i| v.* = 0.1 * @as(f32, @floatFromInt(i % 7)) - 0.3;

    const forwardOf = struct {
        fn run(alloc: Allocator, layer: Block, values: []const f32) ![]const f32 {
            var tape: Tape = .{ .gpa = alloc };
            const x = try tape.constant(try X.fromSlice(alloc, values));
            return (try layer.forward(&tape, x)).value.data;
        }
    }.run;
    const lossOf = struct {
        fn run(alloc: Allocator, layer: Block, values: []const f32, wanted: []const f32) !f32 {
            var tape: Tape = .{ .gpa = alloc };
            const x = try tape.constant(try X.fromSlice(alloc, values));
            const target = try tape.constant(try X.fromSlice(alloc, wanted));
            return (try tape.mse(try layer.forward(&tape, x), target)).value.data[0];
        }
    }.run;

    // Rows 0 to 3 are one sequence and rows 4 to 7 another; changing row 2 may only affect rows 2 and 3.
    // NOTE: the change must differ between features, because layer norm removes a shift that is the same for all of them.
    const base = try forwardOf(a, block, &input);
    var changed = input;
    for (changed[2 * 8 ..][0..8], 0..) |*v, j| v.* += @as(f32, @floatFromInt(j)) - 3;
    const moved = try forwardOf(a, block, &changed);
    for (base[0 .. 2 * 8], moved[0 .. 2 * 8]) |want, got| try testing.expectApproxEqAbs(want, got, 1e-6);
    for (base[4 * 8 ..], moved[4 * 8 ..]) |want, got| try testing.expectApproxEqAbs(want, got, 1e-6);
    var largest: f32 = 0;
    for (base[2 * 8 .. 4 * 8], moved[2 * 8 .. 4 * 8]) |want, got| largest = @max(largest, @abs(want - got));
    try testing.expect(largest > 1e-3);

    var tape: Tape = .{ .gpa = a };
    const x = try tape.leaf(try X.fromSlice(a, &input));
    const target = try tape.constant(try X.fromSlice(a, &targets));
    try tape.backward(try tape.mse(try block.forward(&tape, x), target));
    for ([_]usize{ 0, 10, 27, 40, 63 }) |i| {
        var plus = input;
        var minus = input;
        plus[i] += 1e-2;
        minus[i] -= 1e-2;
        const numeric = (try lossOf(a, block, &plus, &targets) - try lossOf(a, block, &minus, &targets)) / 2e-2;
        try testing.expectApproxEqAbs(numeric, x.grad.data[i], 3e-3);
    }

    const gpu = try Gpu.forTest(1 << 24);
    defer gpu.deinit();
    const on_gpu = try block.toGpu(gpu);
    var gpu_tape: Tape = .{ .gpa = a, .par = .{ .gpu = gpu } };
    const gpu_x = try gpu_tape.leaf(X{ .data = try gpu.alloc(f32, input.len) });
    try gpu.upload(gpu_x.value.data, &input);
    const gpu_target = try gpu_tape.constant(X{ .data = try gpu.alloc(f32, targets.len) });
    try gpu.upload(gpu_target.value.data, &targets);
    try gpu_tape.backward(try gpu_tape.mse(try on_gpu.forward(&gpu_tape, gpu_x), gpu_target));
    var back: [64]f32 = undefined;
    try gpu.download(&back, gpu_x.grad.data);
    for (back, x.grad.data) |got, want| try testing.expectApproxEqAbs(want, got, 1e-4);
}
