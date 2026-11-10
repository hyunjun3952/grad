const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const grad = @import("grad");
const Tensor = grad.Tensor;
const Tape = grad.autograd.Tape;
const Var = grad.autograd.Var;
const Gpu = grad.gpu.Gpu;

const config = @import("config");

const pixels = 784;
const hidden = config.hidden;
const classes = 10;
const train_batch = config.batch;
const eval_batch = 1000;
const learning_rate = 0.002;
const weight_decay = 0.01;

const nn = grad.nn;

const MlpLayers = struct { nn.Linear(pixels, hidden), nn.Linear(hidden, classes) };
const CnnLayers = struct { nn.Conv2d(1, 16, 5, .{}), nn.BatchNorm(16), nn.Conv2d(16, 32, 5, .{}), nn.BatchNorm(32), nn.Linear(512, classes) };
const Model = if (config.cnn) CnnLayers else MlpLayers;

fn Logits(comptime X: type) type {
    return Var(Tensor(f32, &.{ X.shape[0], classes }));
}

fn forward(model: Model, tape: *Tape, x: anytype) grad.autograd.Error!Logits(@TypeOf(x.value)) {
    if (!config.cnn) {
        const l1, const l2 = model;
        return l2.forward(tape, try tape.dropout(try tape.relu(try l1.forward(tape, x)), 0.8));
    }
    const c1, const n1, const c2, const n2, const fc = model;
    const n = @TypeOf(x.value).shape[0];
    const image = grad.autograd.reshape(x, &.{ n, 28, 28, 1 });
    const a = try tape.maxPool(try tape.relu(try n1.forward(tape, try c1.forward(tape, image))));
    const b = try tape.maxPool(try tape.relu(try n2.forward(tape, try c2.forward(tape, a))));
    return fc.forward(tape, try tape.dropout(grad.autograd.reshape(b, &.{ n, 512 }), 0.7));
}
fn accuracy(gpa: Allocator, par: grad.kernels.Parallel, model: Model, data: grad.mnist.Dataset, host_labels: []const u8) !f64 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const host_logits = try gpa.alloc(f32, eval_batch * classes);
    defer gpa.free(host_logits);

    var correct: usize = 0;
    var start: usize = 0;
    while (start + eval_batch <= data.count()) : (start += eval_batch) {
        _ = arena.reset(.retain_capacity);
        if (par.gpu) |dev| try dev.resetStep();
        var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
        const x: Tensor(f32, &.{ eval_batch, pixels }) = .{ .data = data.images[start * pixels ..][0 .. eval_batch * pixels] };
        const logits = try forward(model, &tape, try tape.constant(x));
        const values: []const f32 = if (par.gpu) |dev| blk: {
            try dev.download(host_logits, logits.value.data);
            break :blk host_logits;
        } else logits.value.data;

        for (0..eval_batch) |row| {
            const scores = values[row * classes ..][0..classes];
            var best: usize = 0;
            for (scores, 0..) |s, j| {
                if (s > scores[best]) best = j;
            }
            correct += @intFromBool(best == host_labels[start + row]);
        }
    }
    const seen = data.count() / eval_batch * eval_batch;
    return 100.0 * @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(seen));
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    grad.kernels.flushDenormals();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var use_gpu = false;
    var profile = false;
    var max_shift: u32 = 0;
    var save_path: ?[]const u8 = null;
    var load_path: ?[]const u8 = null;
    var epochs: usize = 5;
    var data_dir: []const u8 = "data/mnist";
    var positional: usize = 0;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            use_gpu = true;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            profile = true;
        } else if (std.mem.startsWith(u8, arg, "--shift=")) {
            max_shift = try std.fmt.parseInt(u32, arg["--shift=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--save=")) {
            save_path = arg["--save=".len..];
        } else if (std.mem.startsWith(u8, arg, "--load=")) {
            load_path = arg["--load=".len..];
        } else {
            if (positional == 0) epochs = try std.fmt.parseInt(usize, arg, 10) else data_dir = arg;
            positional += 1;
        }
    }

    const train = try grad.mnist.load(gpa, io, data_dir, "train");
    defer train.deinit(gpa);
    const test_set = try grad.mnist.load(gpa, io, data_dir, "t10k");
    defer test_set.deinit(gpa);
    if (train.pixels != pixels) return error.UnexpectedImageSize;

    const gpu: ?*Gpu = if (use_gpu) try Gpu.init(gpa, 1 << 30) else null;
    defer if (gpu) |dev| dev.deinit();
    if (gpu) |dev| {
        if (profile) dev.startProfile(io);
    }
    const threads = std.Thread.getCpuCount() catch 1;
    const par: grad.kernels.Parallel = .{ .io = io, .threads = threads, .gpu = gpu };

    var prng = std.Random.DefaultPrng.init(1234);
    const rng = prng.random();
    const host_model = try nn.initAll(Model, gpa, rng);
    defer nn.deinitAll(host_model, gpa);
    const model = if (gpu) |dev| try nn.toGpuAll(host_model, dev) else host_model;
    const params = nn.params(model);
    if (load_path) |path| try grad.checkpoint.load(gpa, io, path, gpu, nn.state(model));
    var adam = try grad.optim.Adam.init(gpa, par, learning_rate, params);
    defer adam.deinit();
    adam.weight_decay = weight_decay;

    const train_data = if (gpu) |dev| try train.toGpu(dev) else train;
    const test_data = if (gpu) |dev| try test_set.toGpu(dev) else test_set;

    if (load_path != null) try out.print("loaded checkpoint, test accuracy {d:.2}%\n", .{try accuracy(gpa, par, model, test_data, test_set.labels)});

    const order = try gpa.alloc(u32, train.count());
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    const order_device = if (gpu) |dev| try dev.alloc(u32, order.len) else order;

    const batch: Tensor(f32, &.{ train_batch, pixels }) = .{
        .data = if (gpu) |dev| try dev.alloc(f32, train_batch * pixels) else try gpa.alloc(f32, train_batch * pixels),
    };
    defer if (gpu == null) gpa.free(batch.data);
    var labels_host: [train_batch]u8 = undefined;
    const labels: []u8 = if (gpu) |dev| try dev.alloc(u8, train_batch) else &labels_host;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const loss_total: []f32 = if (gpu) |dev| try dev.alloc(f32, 1) else &.{};

    try out.print("{s}, AdamW, batch {d}, {s}, {d} training images\n", .{
        if (config.cnn) "CNN conv5x16-conv5x32-fc" else "MLP 784-" ++ std.fmt.comptimePrint("{d}", .{hidden}) ++ "-10",
        train_batch,
        if (gpu != null) "GPU" else "CPU",
        train.count(),
    });
    try out.flush();

    const steps = train.count() / train_batch;
    for (0..epochs) |epoch| {
        const started = Io.Clock.awake.now(io);
        rng.shuffle(u32, order);
        if (gpu) |dev| try dev.upload(order_device, order);

        var loss_sum: f64 = 0;
        if (gpu) |dev| try dev.fill(loss_total, 0);
        const augment: grad.gpu.Augment = .{ .max_shift = max_shift, .seed = epoch };
        for (0..steps) |step| {
            if (gpu) |dev| {
                try dev.gather(train_data.images, train_data.labels, order_device, batch.data, labels, step * train_batch, augment);
            } else {
                for (order[step * train_batch ..][0..train_batch], 0..) |idx, row| {
                    const shift = grad.augment.shiftOf(augment.seed, step * train_batch + row, augment.max_shift);
                    for (batch.data[row * pixels ..][0..pixels], 0..) |*v, p| v.* = grad.augment.pixel(train.images[idx * pixels ..].ptr, p, grad.augment.side(pixels), shift);
                    labels[row] = train.labels[idx];
                }
            }

            _ = arena.reset(.retain_capacity);
            if (gpu) |dev| try dev.resetStep();
            const global_step = epoch * steps + step;
            adam.lr = grad.optim.cosineLr(learning_rate, global_step, epochs * steps);
            var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true, .seed = global_step * 8 };
            try grad.optim.zeroGrad(par, params);
            const logits = try forward(model, &tape, try tape.constant(batch));
            const loss = try tape.crossEntropy(logits, labels);
            try tape.backward(loss);
            try adam.step(par, params);
            if (gpu) |dev| try dev.addInPlace(loss_total, loss.value.data) else loss_sum += loss.value.data[0];
        }
        if (gpu) |dev| loss_sum = try dev.scalar(loss_total);

        const seconds = @as(f64, @floatFromInt(Io.Clock.awake.now(io).nanoseconds - started.nanoseconds)) / 1e9;
        const acc = try accuracy(gpa, par, model, test_data, test_set.labels);
        try out.print("epoch {d}/{d}  train loss {d:.4}  test accuracy {d:.2}%  ({d:.2} s)\n", .{ epoch + 1, epochs, loss_sum / @as(f64, @floatFromInt(steps)), acc, seconds });
        try out.flush();
    }
    if (save_path) |path| {
        try grad.checkpoint.save(gpa, io, path, gpu, nn.state(model));
        try out.print("saved {s}\n", .{path});
    }
    if (gpu) |dev| try dev.printProfile(out);
    try out.flush();
}
