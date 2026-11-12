const std = @import("std");
const lib = @import("grad");

const testing = std.testing;
const autograd = lib.autograd;
const nn = lib.nn;
const onnx = lib.onnx;
const Tensor = lib.tensor.Tensor;
const Gpu = lib.gpu.Gpu;
const Model = lib.onnx_runtime.Model;
const Array = lib.onnx_runtime.Array;
const Executor = lib.onnx_gpu.Executor;

// Runs the model on the CPU runtime and on the device executor and compares every output; returns what the executor did.
fn expectSameOutputs(model: *const Model, gpu: *Gpu, feeds: []const Model.Feed, tolerance: f32) !Executor.Stats {
    const gpa = testing.allocator;
    var want = try model.run(gpa, .{ .threads = 1 }, feeds);
    defer want.deinit();
    var executor = try Executor.init(gpa, model, gpu, .{ .io = testing.io, .threads = 2 });
    defer executor.deinit();
    var got = try executor.run(gpa, feeds);
    defer got.deinit();
    try testing.expectEqual(want.outputs.len, got.outputs.len);
    for (want.outputs, got.outputs) |w, g| {
        try testing.expectEqualSlices(usize, w.dims, g.dims);
        for (w.data, g.data) |expected, actual| try testing.expectApproxEqAbs(expected, actual, tolerance);
    }
    return executor.stats;
}

test "an exported cnn runs on the device with the same outputs as the cpu runtime" {
    const gpu = try Gpu.forTest(64 << 20);
    defer gpu.deinit();
    const gpa = testing.allocator;
    const Layers = struct { nn.ConvBn(1, 4, 3, .{ .pad = 1 }), nn.ConvBn(4, 4, 3, .{ .pad = 1, .groups = 2 }), nn.Linear(4, 3) };
    var prng = std.Random.DefaultPrng.init(12);
    const rng = prng.random();
    const layers = try nn.initAll(Layers, gpa, rng);
    defer nn.deinitAll(layers, gpa);
    inline for (nn.state(layers)) |slice| {
        for (slice) |*v| v.* += rng.float(f32) * 0.5;
    }
    layers[0].norm.variance.data[0] = 2;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var trace: onnx.Trace = .{ .gpa = gpa };
    defer trace.deinit();
    var tape: autograd.Tape = .{ .gpa = arena.allocator(), .trace = &trace };
    const image = try tape.constant(try Tensor(f32, &.{ 2, 64 }).empty(arena.allocator()));
    for (image.value.data) |*v| v.* = rng.floatNorm(f32);
    const first, const second, const fc = layers;
    const a = try tape.maxPool(try first.forwardRelu(&tape, autograd.reshape(image, &.{ 2, 8, 8, 1 })), .{});
    const b = try second.forwardAddRelu(&tape, a, a);
    const logits = try fc.forward(&tape, autograd.reshape(try tape.avgPool(try tape.avgPool(b, .{}), .{}), &.{ 2, 4 }));

    const state = nn.state(layers);
    var weights: [state.len]onnx.Weight = undefined;
    inline for (state, 0..) |slice, i| weights[i] = .{ .id = @intFromPtr(slice.ptr), .data = slice };
    const bytes = try onnx.build(gpa, .{ .trace = &trace, .weights = &weights, .inputs = &.{onnx.View.of(image)}, .outputs = &.{onnx.View.of(logits)} });
    defer gpa.free(bytes);
    var model = try Model.parse(gpa, bytes);
    defer model.deinit();

    const stats = try expectSameOutputs(&model, gpu, &.{.{ .name = "input", .dims = &.{ 2, 64 }, .data = image.value.data }}, 1e-4);
    try testing.expectEqual(@as(usize, 0), stats.host_nodes);
    try testing.expect(stats.device_nodes > 0);
}

test "a causal transformer runs on the device with the same outputs as the cpu runtime" {
    const gpu = try Gpu.forTest(64 << 20);
    defer gpu.deinit();
    const gpa = testing.allocator;
    const vocab = 16;
    const width = 8;
    const steps = 4;
    const rows = 2 * steps;
    const Layers = struct { nn.Embedding(vocab, width), nn.Embedding(steps, width), nn.TransformerBlock(width, 2, steps), nn.LayerNorm(width), nn.Linear(width, vocab) };
    var prng = std.Random.DefaultPrng.init(31);
    const rng = prng.random();
    const layers = try nn.initAll(Layers, gpa, rng);
    defer nn.deinitAll(layers, gpa);
    inline for (nn.state(layers)) |slice| {
        for (slice) |*v| v.* += rng.floatNorm(f32) * 0.3;
    }

    var tokens: [rows]u32 = undefined;
    var positions: [rows]u32 = undefined;
    for (&tokens, &positions, 0..) |*t, *p, i| {
        t.* = rng.uintLessThan(u32, vocab);
        p.* = @intCast(i % steps);
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var trace: onnx.Trace = .{ .gpa = gpa };
    defer trace.deinit();
    var tape: autograd.Tape = .{ .gpa = arena.allocator(), .trace = &trace };
    const token_table, const position_table, const block, const final_norm, const head = layers;
    const embedded = try tape.add(try token_table.forward(&tape, rows, &tokens, f32), try position_table.forward(&tape, rows, &positions, f32));
    const logits = try head.forward(&tape, try final_norm.forward(&tape, try block.forward(&tape, embedded)));

    const state = nn.state(layers);
    var weights: [state.len]onnx.Weight = undefined;
    inline for (state, 0..) |slice, i| weights[i] = .{ .id = @intFromPtr(slice.ptr), .data = slice };
    const bytes = try onnx.build(gpa, .{ .trace = &trace, .weights = &weights, .inputs = &.{ onnx.View.ofIds(&tokens), onnx.View.ofIds(&positions) }, .outputs = &.{onnx.View.of(logits)} });
    defer gpa.free(bytes);
    var model = try Model.parse(gpa, bytes);
    defer model.deinit();

    var token_values: [rows]f32 = undefined;
    var position_values: [rows]f32 = undefined;
    for (&token_values, &position_values, tokens, positions) |*tv, *pv, t, p| {
        tv.* = @floatFromInt(t);
        pv.* = @floatFromInt(p);
    }
    const stats = try expectSameOutputs(&model, gpu, &.{
        .{ .name = "input0", .dims = &.{rows}, .data = &token_values },
        .{ .name = "input1", .dims = &.{rows}, .data = &position_values },
    }, 1e-4);
    try testing.expect(stats.device_nodes > stats.host_nodes);
}

test "a gru step runs on the device, with the ops that have no kernel taken by the host" {
    const gpu = try Gpu.forTest(16 << 20);
    defer gpu.deinit();
    const gpa = testing.allocator;
    const Layers = struct { nn.Gru(3, 4), nn.Linear(4, 2) };
    var prng = std.Random.DefaultPrng.init(23);
    const rng = prng.random();
    const layers = try nn.initAll(Layers, gpa, rng);
    defer nn.deinitAll(layers, gpa);
    inline for (nn.state(layers)) |slice| {
        for (slice) |*v| v.* += rng.floatNorm(f32) * 0.3;
    }

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var trace: onnx.Trace = .{ .gpa = gpa };
    defer trace.deinit();
    var tape: autograd.Tape = .{ .gpa = arena.allocator(), .trace = &trace };
    const x = try tape.constant(try Tensor(f32, &.{ 2, 3 }).empty(arena.allocator()));
    const h = try tape.constant(try Tensor(f32, &.{ 2, 4 }).empty(arena.allocator()));
    for (x.value.data) |*v| v.* = rng.floatNorm(f32);
    for (h.value.data) |*v| v.* = rng.floatNorm(f32) * 0.5;
    const cell, const head = layers;
    const next = try cell.forward(&tape, x, h);
    const scores = try head.forward(&tape, next);

    const state = nn.state(layers);
    var weights: [state.len]onnx.Weight = undefined;
    inline for (state, 0..) |slice, i| weights[i] = .{ .id = @intFromPtr(slice.ptr), .data = slice };
    const bytes = try onnx.build(gpa, .{ .trace = &trace, .weights = &weights, .inputs = &.{ onnx.View.of(x), onnx.View.of(h) }, .outputs = &.{ onnx.View.of(next), onnx.View.of(scores) } });
    defer gpa.free(bytes);
    var model = try Model.parse(gpa, bytes);
    defer model.deinit();

    _ = try expectSameOutputs(&model, gpu, &.{
        .{ .name = "input0", .dims = &.{ 2, 3 }, .data = x.value.data },
        .{ .name = "input1", .dims = &.{ 2, 4 }, .data = h.value.data },
    }, 1e-5);
}

test "a graph in the nchw layout of other exporters runs on the device" {
    const gpu = try Gpu.forTest(16 << 20);
    defer gpu.deinit();
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();

    const arena = std.heap.ArenaAllocator.init(gpa);
    var model: Model = .{ .arena = arena, .nodes = &.{}, .initializers = .init(gpa), .inputs = &.{"x"}, .outputs = &.{"y"} };
    defer model.deinit();
    const a = model.arena.allocator();
    const random = struct {
        fn array(al: std.mem.Allocator, r: std.Random, dims: []const usize, scale: f32) !Array {
            var n: usize = 1;
            for (dims) |d| n *= d;
            const data = try al.alloc(f32, n);
            for (data) |*v| v.* = r.floatNorm(f32) * scale;
            return .{ .dims = try al.dupe(usize, dims), .data = data };
        }
    }.array;
    try model.initializers.put("w", try random(a, rng, &.{ 6, 3, 3, 3 }, 0.3));
    try model.initializers.put("b", try random(a, rng, &.{6}, 0.1));
    try model.initializers.put("scale", try random(a, rng, &.{6}, 1));
    try model.initializers.put("shift", try random(a, rng, &.{6}, 1));
    try model.initializers.put("mean", try random(a, rng, &.{6}, 1));
    const variance = try random(a, rng, &.{6}, 1);
    for (variance.data) |*v| v.* = @abs(v.*) + 0.5;
    try model.initializers.put("variance", variance);
    try model.initializers.put("fc", try random(a, rng, &.{ 6, 5 }, 0.3));
    try model.initializers.put("fc_bias", try random(a, rng, &.{5}, 0.1));
    try model.initializers.put("half", .{ .dims = &.{1}, .data = try a.dupe(f32, &.{0.5}) });
    const flat_shape = [_]i64{ 0, -1 };
    try model.initializers.put("flat", .{ .dims = &.{2}, .data = try a.dupe(f32, &.{ 0, -1 }), .ints = try a.dupe(i64, &flat_shape) });
    model.nodes = &.{
        .{ .op = "Conv", .inputs = &.{ "x", "w", "b" }, .outputs = &.{"c"}, .attributes = &.{ .{ .name = "pads", .ints = &.{ 1, 1, 1, 1 } }, .{ .name = "strides", .ints = &.{ 2, 2 } } } },
        .{ .op = "BatchNormalization", .inputs = &.{ "c", "scale", "shift", "mean", "variance" }, .outputs = &.{"n"} },
        .{ .op = "Relu", .inputs = &.{"n"}, .outputs = &.{"r"} },
        .{ .op = "MaxPool", .inputs = &.{"r"}, .outputs = &.{"m"}, .attributes = &.{ .{ .name = "kernel_shape", .ints = &.{ 2, 2 } }, .{ .name = "strides", .ints = &.{ 1, 1 } } } },
        .{ .op = "Sub", .inputs = &.{ "m", "half" }, .outputs = &.{"s"} },
        .{ .op = "GlobalAveragePool", .inputs = &.{"s"}, .outputs = &.{"g"} },
        .{ .op = "Reshape", .inputs = &.{ "g", "flat" }, .outputs = &.{"f"} },
        .{ .op = "Gemm", .inputs = &.{ "f", "fc", "fc_bias" }, .outputs = &.{"z"} },
        .{ .op = "Softmax", .inputs = &.{"z"}, .outputs = &.{"y"} },
    };
    const image = try random(a, rng, &.{ 2, 3, 6, 6 }, 1);
    const stats = try expectSameOutputs(&model, gpu, &.{.{ .name = "x", .dims = image.dims, .data = image.data }}, 1e-4);
    try testing.expectEqual(@as(usize, 0), stats.host_nodes);
    try testing.expectEqual(@as(usize, 9), stats.device_nodes);
}

test "an op without a device kernel runs on the host and its result goes back to the device" {
    const gpu = try Gpu.forTest(4 << 20);
    defer gpu.deinit();
    const gpa = testing.allocator;
    const arena = std.heap.ArenaAllocator.init(gpa);
    var model: Model = .{ .arena = arena, .nodes = &.{}, .initializers = .init(gpa), .inputs = &.{"x"}, .outputs = &.{"y"} };
    defer model.deinit();
    const a = model.arena.allocator();
    try model.initializers.put("two", .{ .dims = &.{1}, .data = try a.dupe(f32, &.{2}) });
    model.nodes = &.{
        .{ .op = "Relu", .inputs = &.{"x"}, .outputs = &.{"r"} },
        .{ .op = "Sqrt", .inputs = &.{"r"}, .outputs = &.{"q"} },
        .{ .op = "Div", .inputs = &.{ "q", "two" }, .outputs = &.{"d"} },
        .{ .op = "Mul", .inputs = &.{ "d", "d" }, .outputs = &.{"y"} },
    };
    const input = [_]f32{ -1, 4, 9, 16 };
    const stats = try expectSameOutputs(&model, gpu, &.{.{ .name = "x", .dims = &.{4}, .data = @constCast(&input) }}, 1e-5);
    try testing.expectEqual(@as(usize, 2), stats.host_nodes);
    try testing.expectEqual(@as(usize, 2), stats.device_nodes);
}
