const std = @import("std");
const lib = @import("grad");
const View = lib.onnx.View;
const Trace = lib.onnx.Trace;
const build = lib.onnx.build;
const Weight = lib.onnx.Weight;

const testing = std.testing;
const autograd = lib.autograd;
const Tensor = lib.tensor.Tensor;

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
    const bytes = try build(gpa, .{ .trace = &trace, .weights = &weights, .inputs = &.{View.of(image)}, .outputs = &.{View.of(logits)} });
    defer gpa.free(bytes);

    try testing.expectEqualSlices(u8, &.{ 0x08, 0x09 }, bytes[0..2]);
    for ([_][]const u8{ "Conv", "Gelu", "MaxPool", "MatMul", "Add", "Identity", "output", "input" }) |name| try testing.expect(countOf(bytes, name) >= 1);
    try testing.expectEqual(@as(usize, 2), countOf(bytes, "Reshape"));
    try testing.expectEqual(@as(usize, 4), countOf(bytes, "Transpose"));
    try testing.expectError(error.UnknownTensor, build(gpa, .{ .trace = &trace, .weights = weights[1..], .inputs = &.{View.of(image)}, .outputs = &.{View.of(logits)} }));
}
