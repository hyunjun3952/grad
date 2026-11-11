const std = @import("std");
const lib = @import("grad");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const load = lib.checkpoint.load;
const optim = lib.optim;
const Gpu = lib.gpu.Gpu;
const Linear = lib.nn.Linear;
const save = lib.checkpoint.save;

const testing = std.testing;

test "a checkpoint restores every tensor and counter and rejects a different layout" {
    const gpa = testing.allocator;
    const path = ".zig-cache/checkpoint-test.bin";
    defer Io.Dir.cwd().deleteFile(testing.io, path) catch {};

    var a = [_]f32{ 1, 2, 3 };
    var b = [_]f32{ -0.5, 7 };
    try save(gpa, testing.io, path, null, .{ @as([]f32, &a), @as([]f32, &b) }, &.{ 3, 9000 });

    var a2 = [_]f32{ 0, 0, 0 };
    var b2 = [_]f32{ 0, 0 };
    var counters: [2]u32 = undefined;
    try load(gpa, testing.io, path, null, .{ @as([]f32, &a2), @as([]f32, &b2) }, &counters);
    try testing.expectEqualSlices(f32, &a, &a2);
    try testing.expectEqualSlices(f32, &b, &b2);
    try testing.expectEqualSlices(u32, &.{ 3, 9000 }, &counters);

    var wrong = [_]f32{ 0, 0, 0, 0 };
    try testing.expectError(error.CheckpointMismatch, load(gpa, testing.io, path, null, .{ @as([]f32, &wrong), @as([]f32, &b2) }, &counters));
    try testing.expectError(error.CheckpointMismatch, load(gpa, testing.io, path, null, .{@as([]f32, &a2)}, &counters));
    var one: [1]u32 = undefined;
    try testing.expectError(error.CheckpointMismatch, load(gpa, testing.io, path, null, .{ @as([]f32, &a2), @as([]f32, &b2) }, &one));
}

test "a gpu checkpoint round trips through device memory" {
    const gpu = try Gpu.forTest(1 << 20);
    defer gpu.deinit();
    const path = ".zig-cache/checkpoint-gpu-test.bin";
    defer Io.Dir.cwd().deleteFile(testing.io, path) catch {};

    const device = try gpu.alloc(f32, 4);
    try gpu.upload(device, &[_]f32{ 4, 3, 2, 1 });
    try save(testing.allocator, testing.io, path, gpu, .{device}, &.{});
    try gpu.fill(device, 0);
    try load(testing.allocator, testing.io, path, gpu, .{device}, &.{});

    var back: [4]f32 = undefined;
    try gpu.download(&back, device);
    try testing.expectEqualSlices(f32, &.{ 4, 3, 2, 1 }, &back);
}

fn trainSteps(layer: Linear(2, 1), adam: *optim.Adam, from: usize, to: usize) !void {
    for (from..to) |step| {
        const s: f32 = @floatFromInt(step + 1);
        layer.w.grad.data[0] = 0.3 * s - 0.7;
        layer.w.grad.data[1] = 1.0 / s;
        layer.b.grad.data[0] = @sin(s);
        try adam.step(.{}, layer.params());
    }
}

fn freshLinear(gpa: Allocator) !Linear(2, 1) {
    var prng = std.Random.DefaultPrng.init(5);
    return Linear(2, 1).init(gpa, prng.random());
}

test "resuming from a checkpoint with optimizer state reproduces uninterrupted training exactly" {
    const gpa = testing.allocator;
    const path = ".zig-cache/checkpoint-resume-test.bin";
    defer Io.Dir.cwd().deleteFile(testing.io, path) catch {};

    const straight = try freshLinear(gpa);
    defer straight.deinit(gpa);
    var straight_adam = try optim.Adam.init(gpa, .{}, 0.05, straight.params());
    defer straight_adam.deinit();
    straight_adam.weight_decay = 0.01;
    try trainSteps(straight, &straight_adam, 0, 6);

    const first = try freshLinear(gpa);
    defer first.deinit(gpa);
    var first_adam = try optim.Adam.init(gpa, .{}, 0.05, first.params());
    defer first_adam.deinit();
    first_adam.weight_decay = 0.01;
    try trainSteps(first, &first_adam, 0, 3);
    try save(gpa, testing.io, path, null, first.state() ++ first_adam.state(), &.{ 3, first_adam.t });

    const resumed = try freshLinear(gpa);
    defer resumed.deinit(gpa);
    var resumed_adam = try optim.Adam.init(gpa, .{}, 0.05, resumed.params());
    defer resumed_adam.deinit();
    resumed_adam.weight_decay = 0.01;
    var counters: [2]u32 = undefined;
    try load(gpa, testing.io, path, null, resumed.state() ++ resumed_adam.state(), &counters);
    resumed_adam.t = counters[1];
    try trainSteps(resumed, &resumed_adam, counters[0], 6);
    try testing.expectEqualSlices(f32, straight.w.value.data, resumed.w.value.data);
    try testing.expectEqualSlices(f32, straight.b.value.data, resumed.b.value.data);

    const forgetful = try freshLinear(gpa);
    defer forgetful.deinit(gpa);
    var forgetful_adam = try optim.Adam.init(gpa, .{}, 0.05, forgetful.params());
    defer forgetful_adam.deinit();
    forgetful_adam.weight_decay = 0.01;
    var weights_only: [2]u32 = undefined;
    try load(gpa, testing.io, path, null, forgetful.state() ++ forgetful_adam.state(), &weights_only);
    @memset(forgetful_adam.m, 0);
    @memset(forgetful_adam.v, 0);
    try trainSteps(forgetful, &forgetful_adam, 3, 6);
    try testing.expect(!std.mem.eql(f32, straight.w.value.data, forgetful.w.value.data));
}
