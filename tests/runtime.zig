const std = @import("std");
const lib = @import("grad");
const softmax = lib.onnx_runtime.softmax;
const plus = lib.onnx_runtime.plus;
const input = lib.onnx_runtime.input;
const broadcast = lib.onnx_runtime.broadcast;
const pool = lib.onnx_runtime.pool;
const convolve = lib.onnx_runtime.convolve;
const Model = lib.onnx_runtime.Model;
const Array = lib.onnx_runtime.Array;

const testing = std.testing;
const autograd = lib.autograd;
const nn = lib.nn;
const onnx = lib.onnx;
const Tensor = lib.tensor.Tensor;
const Gpu = lib.gpu.Gpu;

test "an exported network, read back with runtime shapes, gives the same outputs as the tape" {
    const gpa = testing.allocator;
    const Layers = struct { nn.ConvBn(1, 4, 3, .{ .pad = 1 }), nn.ConvBn(4, 4, 3, .{ .pad = 1, .groups = 2, .dilation = 1 }), nn.Linear(4, 3) };
    var prng = std.Random.DefaultPrng.init(12);
    const rng = prng.random();
    const layers = try nn.initAll(Layers, gpa, rng);
    defer nn.deinitAll(layers, gpa);
    inline for (nn.state(layers)) |slice| {
        for (slice) |*v| v.* += rng.float(f32) * 0.5;
    }
    layers[0].norm.variance.data[0] = 2;
    layers[1].norm.variance.data[1] = 3;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var trace: onnx.Trace = .{ .gpa = gpa };
    defer trace.deinit();
    var tape: autograd.Tape = .{ .gpa = arena.allocator(), .trace = &trace };
    const image = try tape.constant(try Tensor(f32, &.{ 1, 64 }).empty(arena.allocator()));
    for (image.value.data) |*v| v.* = rng.floatNorm(f32);
    const first, const second, const fc = layers;
    const a = try tape.maxPool(try first.forwardRelu(&tape, autograd.reshape(image, &.{ 1, 8, 8, 1 })), .{});
    const b = try second.forward(&tape, a);
    const logits = try fc.forward(&tape, autograd.reshape(try tape.avgPool(try tape.avgPool(try tape.relu(b), .{}), .{}), &.{ 1, 4 }));

    const state = nn.state(layers);
    var weights: [state.len]onnx.Weight = undefined;
    inline for (state, 0..) |slice, i| weights[i] = .{ .id = @intFromPtr(slice.ptr), .data = slice };
    const bytes = try onnx.build(gpa, .{ .trace = &trace, .weights = &weights, .inputs = &.{onnx.View.of(image)}, .outputs = &.{onnx.View.of(logits)} });
    defer gpa.free(bytes);

    var model = try Model.parse(gpa, bytes);
    defer model.deinit();
    @memset(bytes, 0xAA);
    var result = try model.run(gpa, .{ .threads = 1 }, &.{.{ .name = "input", .dims = &.{ 1, 64 }, .data = image.value.data }});
    defer result.deinit();
    try testing.expectEqual(@as(usize, 1), result.outputs.len);
    try testing.expectEqualSlices(usize, &.{ 1, 3 }, result.outputs[0].dims);
    for (logits.value.data, result.outputs[0].data) |want, got| try testing.expectApproxEqAbs(want, got, 1e-4);
}

test "grouped and dilated convolutions, pooling, softmax and broadcasting follow the onnx definitions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x: Array = .{ .dims = &.{ 1, 2, 3, 3 }, .data = @constCast(&[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 1, 0, 1, 0, 1, 0, 1 }) };
    const w: Array = .{ .dims = &.{ 2, 1, 2, 2 }, .data = @constCast(&[_]f32{ 1, 0, 0, 1, 1, 1, 1, 1 }) };
    const grouped = try convolve(a, .{}, .{ .attributes = &.{.{ .name = "group", .int = 2 }} }, x, w, null);
    try testing.expectEqualSlices(usize, &.{ 1, 2, 2, 2 }, grouped.dims);
    try testing.expectEqualSlices(f32, &.{ 6, 8, 12, 14, 2, 2, 2, 2 }, grouped.data);

    const dilated = try convolve(a, .{}, .{ .attributes = &.{ .{ .name = "dilations", .ints = &.{ 2, 2 } }, .{ .name = "pads", .ints = &.{ 1, 1, 1, 1 } } } }, .{ .dims = &.{ 1, 1, 3, 3 }, .data = x.data[0..9] }, .{ .dims = &.{ 1, 1, 2, 2 }, .data = @constCast(&[_]f32{ 1, 1, 1, 1 }) }, null);
    try testing.expectEqualSlices(usize, &.{ 1, 1, 3, 3 }, dilated.dims);
    try testing.expectEqualSlices(f32, &.{ 5, 10, 5, 10, 20, 10, 5, 10, 5 }, dilated.data[0..9]);

    const maxed = try pool(a, .{ .attributes = &.{ .{ .name = "kernel_shape", .ints = &.{ 2, 2 } }, .{ .name = "strides", .ints = &.{ 1, 1 } } } }, .{ .dims = &.{ 1, 1, 3, 3 }, .data = x.data[0..9] }, true);
    try testing.expectEqualSlices(f32, &.{ 5, 6, 8, 9 }, maxed.data);

    const probabilities = try softmax(a, .{}, .{ .dims = &.{ 1, 2 }, .data = @constCast(&[_]f32{ 0, 0 }) });
    try testing.expectEqualSlices(f32, &.{ 0.5, 0.5 }, probabilities.data);

    const summed = try broadcast(a, .{ .dims = &.{ 2, 1 }, .data = @constCast(&[_]f32{ 10, 20 }) }, .{ .dims = &.{3}, .data = @constCast(&[_]f32{ 1, 2, 3 }) }, plus);
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, summed.dims);
    try testing.expectEqualSlices(f32, &.{ 11, 12, 13, 21, 22, 23 }, summed.data);
}

test "a gru step with two inputs and two outputs survives export and import" {
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
    try testing.expectEqual(@as(usize, 2), model.inputs.len);
    try testing.expectEqual(@as(usize, 2), model.outputs.len);
    var result = try model.run(gpa, .{ .threads = 1 }, &.{
        .{ .name = "input0", .dims = &.{ 2, 3 }, .data = x.value.data },
        .{ .name = "input1", .dims = &.{ 2, 4 }, .data = h.value.data },
    });
    defer result.deinit();
    for (next.value.data, result.outputs[0].data) |want, got| try testing.expectApproxEqAbs(want, got, 1e-5);
    for (scores.value.data, result.outputs[1].data) |want, got| try testing.expectApproxEqAbs(want, got, 1e-5);
}

test "a network traced on the gpu with bf16 activations exports to an f32 graph that matches it within bf16 rounding" {
    const gpu = try Gpu.forTest(8 << 20);
    defer gpu.deinit();
    const gpa = testing.allocator;
    const Layers = struct { nn.BatchNorm(4), nn.Linear(64, 3) };
    var prng = std.Random.DefaultPrng.init(13);
    const rng = prng.random();
    const host = try nn.initAll(Layers, gpa, rng);
    defer nn.deinitAll(host, gpa);
    inline for (nn.state(host)) |slice| {
        for (slice) |*v| v.* += rng.float(f32) * 0.5;
    }
    for (host[0].variance.data) |*v| v.* += 1;
    const layers = try nn.toGpuAll(host, gpu);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var trace: onnx.Trace = .{ .gpa = gpa };
    defer trace.deinit();
    var tape: autograd.Tape = .{ .gpa = arena.allocator(), .par = .{ .gpu = gpu }, .trace = &trace };
    var image_values: [64]f32 = undefined;
    for (&image_values) |*v| v.* = rng.floatNorm(f32);
    const image = try tape.constant(Tensor(f32, &.{ 1, 64 }){ .data = try gpu.alloc(f32, 64) });
    try gpu.upload(image.value.data, &image_values);
    const norm, const fc = layers;
    const activated = try norm.forwardRelu(&tape, try tape.cast(autograd.reshape(image, &.{ 1, 4, 4, 4 }), u16));
    const logits = try tape.cast(try fc.forward(&tape, autograd.reshape(activated, &.{ 1, 64 })), f32);
    var want: [3]f32 = undefined;
    try gpu.download(&want, logits.value.data);

    const state = nn.state(layers);
    var weights: [state.len]onnx.Weight = undefined;
    var copied: usize = 0;
    defer for (weights[0..copied]) |w| gpa.free(w.data);
    inline for (state, 0..) |slice, i| {
        const on_host = try gpa.alloc(f32, slice.len);
        weights[i] = .{ .id = @intFromPtr(slice.ptr), .data = on_host };
        copied += 1;
        try gpu.download(on_host, slice);
    }
    const bytes = try onnx.build(gpa, .{ .trace = &trace, .weights = &weights, .inputs = &.{onnx.View.of(image)}, .outputs = &.{onnx.View.of(logits)} });
    defer gpa.free(bytes);

    var model = try Model.parse(gpa, bytes);
    defer model.deinit();
    var result = try model.run(gpa, .{ .threads = 1 }, &.{.{ .name = "input", .dims = &.{ 1, 64 }, .data = &image_values }});
    defer result.deinit();
    try testing.expectEqualSlices(usize, &.{ 1, 3 }, result.outputs[0].dims);
    for (want, result.outputs[0].data) |w, got| try testing.expectApproxEqAbs(w, got, 0.05 + 0.02 * @abs(w));
}
