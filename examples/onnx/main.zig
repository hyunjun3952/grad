const std = @import("std");
const grad = @import("grad");

pub fn main(init: std.process.Init) void {
    run(init) catch |err| std.debug.print("error: {s}\n", .{@errorName(err)});
}

fn run(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buffer: [1024]u8 = undefined;
    var writer: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &writer.interface;

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        try out.writeAll("usage: onnx MODEL.onnx mnist|cifar [images]\n");
        try out.flush();
        return;
    }
    const wanted: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 1000;

    var model = try grad.onnx.Model.load(gpa, init.io, args[1]);
    defer model.deinit();
    const is_cifar = std.mem.eql(u8, args[2], "cifar");
    const data = if (is_cifar) try grad.cifar.load(gpa, init.io, "data/cifar", "t10k") else try grad.mnist.load(gpa, init.io, "data/mnist", "t10k");
    defer data.deinit(gpa);

    const par: grad.kernels.Parallel = .{ .io = init.io, .threads = std.Thread.getCpuCount() catch 1 };
    const count = @min(wanted, data.count());
    var correct: usize = 0;
    for (0..count) |i| {
        var result = try model.run(gpa, par, &.{.{ .name = model.inputs[0], .dims = &.{ 1, data.pixels }, .data = data.images[i * data.pixels ..][0..data.pixels] }});
        defer result.deinit();
        const scores = result.outputs[0].data;
        var best: usize = 0;
        for (scores, 0..) |s, j| {
            if (s > scores[best]) best = j;
        }
        correct += @intFromBool(best == data.labels[i]);
        if (i == 0) {
            try out.writeAll("logits of image 0:");
            for (scores) |s| try out.print(" {d:.5}", .{s});
            try out.writeByte('\n');
        }
    }
    try out.print("{d} of {d} test images correct ({d:.2}%)\n", .{ correct, count, 100.0 * @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(count)) });
    try out.flush();
}
