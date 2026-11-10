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

const Mlp = struct {
    l1: grad.nn.Linear(pixels, hidden),
    l2: grad.nn.Linear(hidden, classes),

    fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Mlp {
        const l1 = try grad.nn.Linear(pixels, hidden).init(gpa, rng);
        errdefer l1.deinit(gpa);
        return .{ .l1 = l1, .l2 = try grad.nn.Linear(hidden, classes).init(gpa, rng) };
    }

    fn deinit(self: Mlp, gpa: Allocator) void {
        self.l1.deinit(gpa);
        self.l2.deinit(gpa);
    }

    fn toGpu(self: Mlp, gpu: *Gpu) !Mlp {
        return .{ .l1 = try self.l1.toGpu(gpu), .l2 = try self.l2.toGpu(gpu) };
    }

    fn params(self: Mlp) @TypeOf(self.l1.params() ++ self.l2.params()) {
        return self.l1.params() ++ self.l2.params();
    }

    fn forward(self: Mlp, tape: *Tape, x: anytype) grad.autograd.Error!Var(Tensor(f32, &.{ @TypeOf(x.value).shape[0], classes })) {
        return self.l2.forward(tape, try tape.relu(try self.l1.forward(tape, x)));
    }
};

const Cnn = struct {
    c1: grad.nn.Conv2d(1, 16, 5),
    c2: grad.nn.Conv2d(16, 32, 5),
    fc: grad.nn.Linear(512, classes),

    fn init(gpa: Allocator, rng: std.Random) Allocator.Error!Cnn {
        const c1 = try grad.nn.Conv2d(1, 16, 5).init(gpa, rng);
        errdefer c1.deinit(gpa);
        const c2 = try grad.nn.Conv2d(16, 32, 5).init(gpa, rng);
        errdefer c2.deinit(gpa);
        return .{ .c1 = c1, .c2 = c2, .fc = try grad.nn.Linear(512, classes).init(gpa, rng) };
    }

    fn deinit(self: Cnn, gpa: Allocator) void {
        self.c1.deinit(gpa);
        self.c2.deinit(gpa);
        self.fc.deinit(gpa);
    }

    fn toGpu(self: Cnn, gpu: *Gpu) !Cnn {
        return .{ .c1 = try self.c1.toGpu(gpu), .c2 = try self.c2.toGpu(gpu), .fc = try self.fc.toGpu(gpu) };
    }

    fn params(self: Cnn) @TypeOf(self.c1.params() ++ self.c2.params() ++ self.fc.params()) {
        return self.c1.params() ++ self.c2.params() ++ self.fc.params();
    }

    fn forward(self: Cnn, tape: *Tape, x: anytype) grad.autograd.Error!Var(Tensor(f32, &.{ @TypeOf(x.value).shape[0], classes })) {
        const n = @TypeOf(x.value).shape[0];
        const image = grad.autograd.reshape(x, &.{ n, 28, 28, 1 });
        const a = try tape.maxPool(try tape.relu(try self.c1.forward(tape, image)));
        const b = try tape.maxPool(try tape.relu(try self.c2.forward(tape, a)));
        return self.fc.forward(tape, grad.autograd.reshape(b, &.{ n, 512 }));
    }
};

const Model = if (config.cnn) Cnn else Mlp;

fn accuracy(gpa: Allocator, par: grad.kernels.Parallel, model: Model, data: grad.mnist.Dataset, host_labels: []const u8) !f64 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const host_logits = try gpa.alloc(f32, eval_batch * classes);
    defer gpa.free(host_logits);

    var correct: usize = 0;
    var start: usize = 0;
    while (start + eval_batch <= data.count()) : (start += eval_batch) {
        _ = arena.reset(.retain_capacity);
        if (par.gpu) |dev| dev.resetStep();
        var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
        const x: Tensor(f32, &.{ eval_batch, pixels }) = .{ .data = data.images[start * pixels ..][0 .. eval_batch * pixels] };
        const logits = try model.forward(&tape, try tape.constant(x));
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
    var epochs: usize = 5;
    var data_dir: []const u8 = "data/mnist";
    var positional: usize = 0;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            use_gpu = true;
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
    const threads = std.Thread.getCpuCount() catch 1;
    const par: grad.kernels.Parallel = .{ .io = io, .threads = threads, .gpu = gpu };

    var prng = std.Random.DefaultPrng.init(1234);
    const rng = prng.random();
    const host_model = try Model.init(gpa, rng);
    defer host_model.deinit(gpa);
    const model = if (gpu) |dev| try host_model.toGpu(dev) else host_model;
    const params = model.params();
    var adam = try grad.optim.Adam.init(gpa, par, 0.001, params);
    defer adam.deinit();

    const train_data = if (gpu) |dev| try train.toGpu(dev) else train;
    const test_data = if (gpu) |dev| try test_set.toGpu(dev) else test_set;

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

    try out.print("{s}, Adam, batch {d}, {s}, {d} training images\n", .{
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
        for (0..steps) |step| {
            if (gpu) |dev| {
                try dev.gather(train_data.images, train_data.labels, order_device, batch.data, labels, step * train_batch, train_batch, pixels);
            } else {
                for (order[step * train_batch ..][0..train_batch], 0..) |idx, row| {
                    @memcpy(batch.data[row * pixels ..][0..pixels], train.images[idx * pixels ..][0..pixels]);
                    labels[row] = train.labels[idx];
                }
            }

            _ = arena.reset(.retain_capacity);
            if (gpu) |dev| dev.resetStep();
            var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
            try grad.optim.zeroGrad(par, params);
            const logits = try model.forward(&tape, try tape.constant(batch));
            const loss = try tape.crossEntropy(logits, labels);
            try tape.backward(loss);
            try adam.step(par, params);
            loss_sum += try tape.scalar(loss);
        }

        const seconds = @as(f64, @floatFromInt(Io.Clock.awake.now(io).nanoseconds - started.nanoseconds)) / 1e9;
        const acc = try accuracy(gpa, par, model, test_data, test_set.labels);
        try out.print("epoch {d}/{d}  train loss {d:.4}  test accuracy {d:.2}%  ({d:.2} s)\n", .{ epoch + 1, epochs, loss_sum / @as(f64, @floatFromInt(steps)), acc, seconds });
        try out.flush();
    }
}
