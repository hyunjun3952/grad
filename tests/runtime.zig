const std = @import("std");
const lib = @import("grad");
const softmax = lib.onnx_runtime.softmax;
const plus = lib.onnx_runtime.plus;
const input = lib.onnx_runtime.input;
const broadcast = lib.onnx_runtime.broadcast;
const pool = lib.onnx_runtime.pool;
const convolve = lib.onnx_runtime.convolve;
const transpose = lib.onnx_runtime.transpose;
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
    const b = try second.forwardAddRelu(&tape, a, a);
    const logits = try fc.forward(&tape, autograd.reshape(try tape.avgPool(try tape.avgPool(b, .{}), .{}), &.{ 1, 4 }));

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

    const maxed = try pool(a, .{}, .{ .attributes = &.{ .{ .name = "kernel_shape", .ints = &.{ 2, 2 } }, .{ .name = "strides", .ints = &.{ 1, 1 } } } }, .{ .dims = &.{ 1, 1, 3, 3 }, .data = x.data[0..9] }, true);
    try testing.expectEqualSlices(f32, &.{ 5, 6, 8, 9 }, maxed.data);

    const probabilities = try softmax(a, .{}, .{ .dims = &.{ 1, 2 }, .data = @constCast(&[_]f32{ 0, 0 }) });
    try testing.expectEqualSlices(f32, &.{ 0.5, 0.5 }, probabilities.data);

    const summed = try broadcast(a, .{}, .{ .dims = &.{ 2, 1 }, .data = @constCast(&[_]f32{ 10, 20 }) }, .{ .dims = &.{3}, .data = @constCast(&[_]f32{ 1, 2, 3 }) }, plus);
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, summed.dims);
    try testing.expectEqualSlices(f32, &.{ 11, 12, 13, 21, 22, 23 }, summed.data);
}

test "transposes and broadcasts on their fast paths give the same values as an index-by-index reference" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const par: lib.kernels.Parallel = .{ .io = testing.io, .threads = 3 };
    const dims = [_]usize{ 2, 33, 5, 7 };
    const x: Array = .{ .dims = &dims, .data = try a.alloc(f32, 2 * 33 * 5 * 7) };
    for (x.data, 0..) |*v, i| v.* = @floatFromInt(i);
    const perms = [_][]const i64{ &.{ 0, 3, 1, 2 }, &.{ 0, 2, 3, 1 }, &.{ 1, 2, 3, 0 }, &.{ 3, 2, 1, 0 }, &.{ 0, 2, 1, 3 } };
    for (perms) |perm| {
        const got = try transpose(a, par, .{ .attributes = &.{.{ .name = "perm", .ints = perm }} }, x);
        var index: [4]usize = @splat(0);
        for (got.data) |v| {
            var source: [4]usize = undefined;
            for (perm, 0..) |p, d| source[@intCast(p)] = index[d];
            try testing.expectEqual(@as(f32, @floatFromInt(((source[0] * 33 + source[1]) * 5 + source[2]) * 7 + source[3])), v);
            var d: usize = 4;
            while (d > 0) {
                d -= 1;
                index[d] += 1;
                if (index[d] < got.dims[d]) break;
                index[d] = 0;
            }
        }
    }

    const channel: Array = .{ .dims = &.{ 1, 7 }, .data = try a.alloc(f32, 7) };
    for (channel.data, 0..) |*v, i| v.* = @floatFromInt(1000 * i);
    const right = try broadcast(a, par, x, channel, lib.onnx_runtime.minus);
    const left = try broadcast(a, par, channel, x, lib.onnx_runtime.minus);
    const same = try broadcast(a, par, x, x, plus);
    try testing.expectEqualSlices(usize, &dims, right.dims);
    try testing.expectEqualSlices(usize, &dims, left.dims);
    for (x.data, right.data, left.data, same.data, 0..) |v, r, l, s, i| {
        try testing.expectEqual(v - channel.data[i % 7], r);
        try testing.expectEqual(channel.data[i % 7] - v, l);
        try testing.expectEqual(2 * v, s);
    }
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
    const weights = try onnx.WeightSet(state.len).collect(gpa, .{ .gpu = gpu }, state);
    defer weights.deinit(gpa);
    const bytes = try onnx.build(gpa, .{ .trace = &trace, .weights = &weights.items, .inputs = &.{onnx.View.of(image)}, .outputs = &.{onnx.View.of(logits)} });
    defer gpa.free(bytes);

    var model = try Model.parse(gpa, bytes);
    defer model.deinit();
    var result = try model.run(gpa, .{ .threads = 1 }, &.{.{ .name = "input", .dims = &.{ 1, 64 }, .data = &image_values }});
    defer result.deinit();
    try testing.expectEqualSlices(usize, &.{ 1, 3 }, result.outputs[0].dims);
    for (want, result.outputs[0].data) |w, got| try testing.expectApproxEqAbs(w, got, 0.05 + 0.02 * @abs(w));
}

test "a causal transformer with token and position inputs survives export and import" {
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
    var result = try model.run(gpa, .{ .threads = 1 }, &.{
        .{ .name = "input0", .dims = &.{rows}, .data = &token_values },
        .{ .name = "input1", .dims = &.{rows}, .data = &position_values },
    });
    defer result.deinit();
    try testing.expectEqualSlices(usize, &.{ rows, vocab }, result.outputs[0].dims);
    for (logits.value.data, result.outputs[0].data) |want, got| try testing.expectApproxEqAbs(want, got, 1e-4);
}
