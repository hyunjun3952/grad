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

    const all_args = try init.minimal.args.toSlice(init.arena.allocator());
    var positional: std.ArrayList([]const u8) = .empty;
    var use_gpu = false;
    var tf32 = false;
    var scratch_mib: usize = 1024;
    for (all_args) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            use_gpu = true;
        } else if (std.mem.eql(u8, arg, "--tf32")) {
            tf32 = true;
        } else if (std.mem.startsWith(u8, arg, "--scratch=")) {
            scratch_mib = try std.fmt.parseInt(usize, arg["--scratch=".len..], 10);
        } else {
            try positional.append(init.arena.allocator(), arg);
        }
    }
    const args = positional.items;
    if (args.len < 3) {
        try out.writeAll("usage: onnx MODEL.onnx mnist|cifar [images] [batch] [--gpu] [--tf32] [--scratch=MiB]\n");
        try out.flush();
        return;
    }
    const wanted: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 1000;
    const batch: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 1;

    var model = try grad.onnx.Model.load(gpa, init.io, args[1]);
    defer model.deinit();
    const is_cifar = std.mem.eql(u8, args[2], "cifar");
    const data = if (is_cifar) try grad.cifar.load(gpa, init.io, "data/cifar", "t10k") else try grad.mnist.load(gpa, init.io, "data/mnist", "t10k");
    defer data.deinit(gpa);

    const par: grad.kernels.Parallel = .{ .io = init.io, .threads = std.Thread.getCpuCount() catch 1 };
    // NOTE: the device keeps every activation of one run until the next run starts, so --scratch must cover a whole batch.
    var executor: ?grad.onnx_gpu.Executor = null;
    defer if (executor) |*e| e.deinit();
    var device: ?*grad.gpu.Gpu = null;
    defer if (device) |g| g.deinit();
    if (use_gpu) {
        const g = try grad.gpu.Gpu.init(gpa, .{ .step_bytes = scratch_mib << 20 });
        device = g;
        g.tf32 = tf32;
        executor = try grad.onnx_gpu.Executor.init(gpa, &model, g, par);
    }

    const count = @min(wanted, data.count());
    var correct: usize = 0;
    const started = std.Io.Clock.awake.now(init.io);
    var done: usize = 0;
    while (done < count) {
        // A dynamic-batch graph takes any leading count; a fixed one only the size it was exported with.
        const n = @min(batch, count - done);
        const feeds = [_]grad.onnx_runtime.Model.Feed{.{ .name = model.inputs[0], .dims = &.{ n, data.pixels }, .data = data.images[done * data.pixels ..][0 .. n * data.pixels] }};
        var result = if (executor) |*e| try e.run(gpa, &feeds) else try model.run(gpa, par, &feeds);
        defer result.deinit();
        const scores = result.outputs[0].data;
        const classes = scores.len / n;
        for (0..n) |row| {
            correct += @intFromBool(grad.trainer.argmax(scores[row * classes ..][0..classes]) == data.labels[done + row]);
        }
        if (done == 0) {
            try out.writeAll("logits of image 0:");
            for (scores[0..classes]) |s| try out.print(" {d:.5}", .{s});
            try out.writeByte('\n');
        }
        done += n;
    }
    const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - started.nanoseconds)) / 1e9;
    try out.print("{d} of {d} test images correct ({d:.2}%)\n", .{ correct, count, 100.0 * @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(count)) });
    if (executor) |e| try out.print("last batch: {d} nodes on the device, {d} on the host ({d} uploads, {d} downloads)\n", .{ e.stats.device_nodes, e.stats.host_nodes, e.stats.uploads, e.stats.downloads });
    try out.print("inference took {d:.3} s ({d:.1} images/s)\n", .{ seconds, @as(f64, @floatFromInt(count)) / seconds });
    try out.flush();
}
