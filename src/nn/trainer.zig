const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const augment_math = @import("../gpu/math/augment.zig");
const autograd = @import("../autograd/autograd.zig");
const checkpoint = @import("checkpoint.zig");
const Dataset = @import("../data/dataset.zig").Dataset;
const Gpu = @import("../gpu/gpu.zig").Gpu;
const kernels = @import("../cpu/kernels.zig");
const nn = @import("nn.zig");
const optim = @import("optim.zig");
const Tensor = @import("../tensor/tensor.zig").Tensor;

const Tape = autograd.Tape;

fn Runner(comptime Spec: type) type {
    const pixels = Spec.side * Spec.side * Spec.channels;

    return struct {
        fn accuracy(gpa: Allocator, par: kernels.Parallel, model: Spec.Layers, data: Dataset) !f64 {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const host_logits = try gpa.alloc(f32, Spec.eval_batch * Spec.classes);
            defer gpa.free(host_logits);
            const host_labels = try gpa.alloc(u8, data.count());
            defer gpa.free(host_labels);
            if (par.gpu) |dev| try dev.download(host_labels, data.labels) else @memcpy(host_labels, data.labels);

            var correct: usize = 0;
            var start: usize = 0;
            while (start + Spec.eval_batch <= data.count()) : (start += Spec.eval_batch) {
                _ = arena.reset(.retain_capacity);
                if (par.gpu) |dev| try dev.resetStep();
                var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
                const x: Tensor(f32, &.{ Spec.eval_batch, pixels }) = .{ .data = data.images[start * pixels ..][0 .. Spec.eval_batch * pixels] };
                const logits = try Spec.forward(model, &tape, try tape.constant(x));
                const values: []const f32 = if (par.gpu) |dev| blk: {
                    try dev.download(host_logits, logits.value.data);
                    break :blk host_logits;
                } else logits.value.data;

                for (0..Spec.eval_batch) |row| {
                    const scores = values[row * Spec.classes ..][0..Spec.classes];
                    var best: usize = 0;
                    for (scores, 0..) |s, j| {
                        if (s > scores[best]) best = j;
                    }
                    correct += @intFromBool(best == host_labels[start + row]);
                }
            }
            const seen = data.count() / Spec.eval_batch * Spec.eval_batch;
            return 100.0 * @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(seen));
        }

        fn run(init: std.process.Init) !void {
            const io = init.io;
            const gpa = init.gpa;

            var stdout_buffer: [1024]u8 = undefined;
            var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
            const out = &stdout_file_writer.interface;

            kernels.flushDenormals();
            const args = try init.minimal.args.toSlice(init.arena.allocator());
            var use_gpu = false;
            var profile = false;
            var tensor_cores = false;
            var smoothing: f32 = 0;
            var augment_config: augment_math.Config = .{ .channels = Spec.channels };
            var save_path: ?[]const u8 = null;
            var load_path: ?[]const u8 = null;
            var epochs: usize = 5;
            var until: ?usize = null;
            var data_dir: []const u8 = Spec.data_dir;
            var positional: usize = 0;
            for (args[1..]) |arg| {
                if (std.mem.eql(u8, arg, "--gpu")) {
                    use_gpu = true;
                } else if (std.mem.eql(u8, arg, "--profile")) {
                    profile = true;
                } else if (std.mem.startsWith(u8, arg, "--smoothing=")) {
                    smoothing = try std.fmt.parseFloat(f32, arg["--smoothing=".len..]);
                } else if (std.mem.eql(u8, arg, "--tensor-cores")) {
                    tensor_cores = true;
                } else if (std.mem.eql(u8, arg, "--flip")) {
                    augment_config.flip = true;
                } else if (std.mem.startsWith(u8, arg, "--shift=")) {
                    augment_config.max_shift = try std.fmt.parseInt(u32, arg["--shift=".len..], 10);
                } else if (std.mem.startsWith(u8, arg, "--rotate=")) {
                    augment_config.max_angle = try std.fmt.parseFloat(f32, arg["--rotate=".len..]);
                } else if (std.mem.startsWith(u8, arg, "--zoom=")) {
                    augment_config.max_scale = try std.fmt.parseFloat(f32, arg["--zoom=".len..]);
                } else if (std.mem.startsWith(u8, arg, "--until=")) {
                    until = try std.fmt.parseInt(usize, arg["--until=".len..], 10);
                } else if (std.mem.startsWith(u8, arg, "--save=")) {
                    save_path = arg["--save=".len..];
                } else if (std.mem.startsWith(u8, arg, "--load=")) {
                    load_path = arg["--load=".len..];
                } else {
                    if (positional == 0) epochs = try std.fmt.parseInt(usize, arg, 10) else data_dir = arg;
                    positional += 1;
                }
            }

            const train = try Spec.Loader.load(gpa, io, data_dir, "train");
            defer train.deinit(gpa);
            const test_set = try Spec.Loader.load(gpa, io, data_dir, "t10k");
            defer test_set.deinit(gpa);
            if (train.pixels != pixels) return error.UnexpectedImageSize;

            const gpu: ?*Gpu = if (use_gpu) try Gpu.init(gpa, Spec.scratch_bytes) else null;
            defer if (gpu) |dev| dev.deinit();
            if (gpu) |dev| {
                if (profile) dev.startProfile(io);
                dev.tensor_cores = tensor_cores;
            }
            const threads = std.Thread.getCpuCount() catch 1;
            const par: kernels.Parallel = .{ .io = io, .threads = threads, .gpu = gpu };

            var prng = std.Random.DefaultPrng.init(1234);
            const host_model = try nn.initAll(Spec.Layers, gpa, prng.random());
            defer nn.deinitAll(host_model, gpa);
            const model = if (gpu) |dev| try nn.toGpuAll(host_model, dev) else host_model;
            const params = nn.params(model);
            var adam = try optim.Adam.init(gpa, par, Spec.learning_rate, params);
            defer adam.deinit();
            adam.weight_decay = Spec.weight_decay;
            var start_epoch: usize = 0;
            if (load_path) |path| {
                var progress: [2]u32 = undefined;
                try checkpoint.load(gpa, io, path, gpu, nn.state(model) ++ adam.state(), &progress);
                start_epoch = progress[0];
                adam.t = progress[1];
            }

            const train_data = if (gpu) |dev| try train.toGpu(dev) else train;
            const test_data = if (gpu) |dev| try test_set.toGpu(dev) else test_set;

            if (load_path != null) try out.print("loaded checkpoint after {d} epochs, test accuracy {d:.2}%\n", .{ start_epoch, try accuracy(gpa, par, model, test_data) });

            const order = try gpa.alloc(u32, train.count());
            defer gpa.free(order);
            const order_device = if (gpu) |dev| try dev.alloc(u32, order.len) else order;

            const batch: Tensor(f32, &.{ Spec.batch, pixels }) = .{
                .data = if (gpu) |dev| try dev.alloc(f32, Spec.batch * pixels) else try gpa.alloc(f32, Spec.batch * pixels),
            };
            defer if (gpu == null) gpa.free(batch.data);
            var labels_host: [Spec.batch]u8 = undefined;
            const labels: []u8 = if (gpu) |dev| try dev.alloc(u8, Spec.batch) else &labels_host;

            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const loss_total: []f32 = if (gpu) |dev| try dev.alloc(f32, 1) else &.{};

            try out.print("{s}, AdamW, batch {d}, {s}, {d} training images\n", .{ Spec.name, Spec.batch, if (gpu != null) "GPU" else "CPU", train.count() });
            try out.flush();

            const steps = train.count() / Spec.batch;
            const stop_epoch = @min(until orelse epochs, epochs);
            for (start_epoch..@max(start_epoch, stop_epoch)) |epoch| {
                const started = Io.Clock.awake.now(io);
                for (order, 0..) |*o, i| o.* = @intCast(i);
                var shuffle_prng = std.Random.DefaultPrng.init(1234 + epoch);
                shuffle_prng.random().shuffle(u32, order);
                if (gpu) |dev| try dev.upload(order_device, order);

                var loss_sum: f64 = 0;
                if (gpu) |dev| try dev.fill(loss_total, 0);
                var augment = augment_config;
                augment.seed = epoch;
                for (0..steps) |step| {
                    if (gpu) |dev| {
                        try dev.gather(train_data.images, train_data.labels, order_device, batch.data, labels, step * Spec.batch, augment);
                    } else {
                        for (order[step * Spec.batch ..][0..Spec.batch], 0..) |idx, row| {
                            const row_pixels = batch.data[row * pixels ..][0..pixels];
                            if (augment.active()) {
                                const transform = augment_math.transformOf(step * Spec.batch + row, augment);
                                for (row_pixels, 0..) |*v, p| v.* = augment_math.pixel(train.images[idx * pixels ..].ptr, p, Spec.side, Spec.channels, transform);
                            } else {
                                @memcpy(row_pixels, train.images[idx * pixels ..][0..pixels]);
                            }
                            labels[row] = train.labels[idx];
                        }
                    }

                    _ = arena.reset(.retain_capacity);
                    if (gpu) |dev| try dev.resetStep();
                    const global_step = epoch * steps + step;
                    adam.lr = optim.cosineLr(Spec.learning_rate, global_step, epochs * steps);
                    var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true, .smoothing = smoothing, .seed = global_step * 8 };
                    try optim.zeroGrad(par, params);
                    const logits = try Spec.forward(model, &tape, try tape.constant(batch));
                    const loss = try tape.crossEntropy(logits, labels);
                    try tape.backward(loss);
                    try adam.step(par, params);
                    if (gpu) |dev| try dev.addInPlace(loss_total, loss.value.data) else loss_sum += loss.value.data[0];
                }
                if (gpu) |dev| loss_sum = try dev.scalar(loss_total);

                const seconds = @as(f64, @floatFromInt(Io.Clock.awake.now(io).nanoseconds - started.nanoseconds)) / 1e9;
                const acc = try accuracy(gpa, par, model, test_data);
                try out.print("epoch {d}/{d}  train loss {d:.4}  test accuracy {d:.2}%  ({d:.2} s)\n", .{ epoch + 1, epochs, loss_sum / @as(f64, @floatFromInt(steps)), acc, seconds });
                try out.flush();
            }
            if (save_path) |path| {
                const progress = [_]u32{ @intCast(@max(stop_epoch, start_epoch)), adam.t };
                try checkpoint.save(gpa, io, path, gpu, nn.state(model) ++ adam.state(), &progress);
                try out.print("saved {s}\n", .{path});
            }
            if (gpu) |dev| try dev.printProfile(out);
            try out.flush();
        }
    };
}

pub fn run(comptime Spec: type, init: std.process.Init) !void {
    try Runner(Spec).run(init);
}
