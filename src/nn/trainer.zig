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
const onnx = @import("../onnx/onnx.zig");
const optim = @import("optim.zig");
const Tensor = @import("../tensor/tensor.zig").Tensor;

const Tape = autograd.Tape;

// ***** command line *****

const Progress = struct { epoch: usize, epochs: usize, done: usize, steps: usize, loss: f64, elapsed_ns: i96, epochs_left: usize };

fn printProgress(out: *Io.Writer, p: Progress) !void {
    const epoch, const epochs, const done, const steps, const loss, const elapsed_ns, const epochs_left = .{ p.epoch, p.epochs, p.done, p.steps, p.loss, p.elapsed_ns, p.epochs_left };
    const width = 24;
    const filled = done * width / steps;
    var bar: [width]u8 = undefined;
    for (&bar, 0..) |*c, i| c.* = if (i < filled) '#' else '-';
    const seconds: f64 = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const per_step = seconds / @as(f64, @floatFromInt(done));
    const remaining: u64 = @intFromFloat(per_step * @as(f64, @floatFromInt((steps - done) + epochs_left * steps)));
    try out.print("\r\x1b[2Kepoch {d}/{d} [{s}] {d:>3}%  loss {d:.4}  {d:.1}s  eta {d}m{d:0>2}s", .{ epoch + 1, epochs, bar[0..], done * 100 / steps, loss, seconds, remaining / 60, remaining % 60 });
    try out.flush();
}

const Mix = struct { box: augment_math.Box = .{}, kept: f32 = 1 };

// Draws a CutMix box covering a random share of the image, or an empty box with probability 1 - probability.
// Returns the share of the original image that stays visible, which weights the two labels in the loss.
fn drawMix(side: u32, probability: f32, seed: u64) Mix {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    if (rng.float(f32) >= probability) return .{};
    const cut: u32 = @intFromFloat(@as(f32, @floatFromInt(side)) * @sqrt(1 - rng.float(f32)));
    const cx = rng.uintLessThan(u32, side);
    const cy = rng.uintLessThan(u32, side);
    const box: augment_math.Box = .{ .x0 = cx -| cut / 2, .y0 = cy -| cut / 2, .x1 = @min(side, cx + (cut + 1) / 2), .y1 = @min(side, cy + (cut + 1) / 2) };
    const area: f32 = @floatFromInt((box.x1 - box.x0) * (box.y1 - box.y0));
    return .{ .box = box, .kept = 1 - area / @as(f32, @floatFromInt(side * side)) };
}

fn valueOf(arg: [:0]const u8, comptime name: []const u8) ?[:0]const u8 {
    return if (std.mem.startsWith(u8, arg, name ++ "=")) arg[name.len + 1 ..] else null;
}

// Everything the command line can set. The first number is the epoch count and a second plain argument is the data directory.
const Options = struct {
    gpu: bool = false,
    profile: bool = false,
    tensor_cores: bool = false,
    tf32: bool = false,
    deterministic: bool = false,
    progress: bool = false,
    mirrored: bool = false,
    cudnn_dir: ?[:0]const u8 = null,
    smoothing: f32 = 0,
    cutmix: f32 = 0,
    ema_decay: f32 = 0,
    augment: augment_math.Config,
    save_path: ?[]const u8 = null,
    load_path: ?[]const u8 = null,
    data_dir: ?[]const u8 = null,
    epochs: usize = 5,
    until: ?usize = null,
    warmup_epochs: usize = 0,
    predict: usize = 0,
    batch: ?usize = null,
    learning_rate: ?f32 = null,
    onnx_path: ?[]const u8 = null,

    fn parse(args: []const [:0]const u8, channels: u32) !Options {
        var o: Options = .{ .augment = .{ .channels = channels } };
        var positional: usize = 0;
        for (args) |arg| {
            if (std.mem.eql(u8, arg, "--gpu")) {
                o.gpu = true;
            } else if (std.mem.eql(u8, arg, "--profile")) {
                o.profile = true;
            } else if (std.mem.eql(u8, arg, "--progress")) {
                o.progress = true;
            } else if (std.mem.eql(u8, arg, "--deterministic")) {
                o.deterministic = true;
            } else if (std.mem.eql(u8, arg, "--tf32")) {
                o.tf32 = true;
            } else if (std.mem.eql(u8, arg, "--tensor-cores")) {
                o.tensor_cores = true;
            } else if (std.mem.eql(u8, arg, "--flip")) {
                o.augment.flip = true;
            } else if (std.mem.eql(u8, arg, "--tta")) {
                o.mirrored = true;
            } else if (std.mem.eql(u8, arg, "--cudnn")) {
                o.cudnn_dir = "data/cudnn";
            } else if (valueOf(arg, "--cudnn")) |v| {
                o.cudnn_dir = v;
            } else if (valueOf(arg, "--smoothing")) |v| {
                o.smoothing = try std.fmt.parseFloat(f32, v);
            } else if (valueOf(arg, "--shift")) |v| {
                o.augment.max_shift = try std.fmt.parseInt(u32, v, 10);
            } else if (valueOf(arg, "--rotate")) |v| {
                o.augment.max_angle = try std.fmt.parseFloat(f32, v);
            } else if (valueOf(arg, "--zoom")) |v| {
                o.augment.max_scale = try std.fmt.parseFloat(f32, v);
            } else if (valueOf(arg, "--cutout")) |v| {
                o.augment.cutout = try std.fmt.parseFloat(f32, v);
            } else if (valueOf(arg, "--cutmix")) |v| {
                o.cutmix = try std.fmt.parseFloat(f32, v);
            } else if (valueOf(arg, "--ema")) |v| {
                o.ema_decay = try std.fmt.parseFloat(f32, v);
            } else if (valueOf(arg, "--warmup")) |v| {
                o.warmup_epochs = try std.fmt.parseInt(usize, v, 10);
            } else if (valueOf(arg, "--until")) |v| {
                o.until = try std.fmt.parseInt(usize, v, 10);
            } else if (valueOf(arg, "--batch")) |v| {
                o.batch = try std.fmt.parseInt(usize, v, 10);
            } else if (valueOf(arg, "--lr")) |v| {
                o.learning_rate = try std.fmt.parseFloat(f32, v);
            } else if (valueOf(arg, "--predict")) |v| {
                o.predict = try std.fmt.parseInt(usize, v, 10);
            } else if (valueOf(arg, "--onnx")) |v| {
                o.onnx_path = v;
            } else if (valueOf(arg, "--save")) |v| {
                o.save_path = v;
            } else if (valueOf(arg, "--load")) |v| {
                o.load_path = v;
            } else {
                if (positional == 0) o.epochs = try std.fmt.parseInt(usize, arg, 10) else o.data_dir = arg;
                positional += 1;
            }
        }
        return o;
    }
};
// ***** training loop *****

fn Runner(comptime Spec: type, comptime batch_size: usize) type {
    const pixels = Spec.side * Spec.side * Spec.channels;

    return struct {
        fn scores(par: kernels.Parallel, model: Spec.Layers, arena: *std.heap.ArenaAllocator, host: []f32, images: []f32) !void {
            _ = arena.reset(.retain_capacity);
            if (par.gpu) |dev| try dev.resetStep();
            var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
            const x: Tensor(f32, &.{ Spec.eval_batch, pixels }) = .{ .data = images };
            const logits = try Spec.forward(model, &tape, try tape.constant(x));
            if (par.gpu) |dev| try dev.download(host, logits.value.data) else @memcpy(host, logits.value.data);
        }

        fn accuracy(gpa: Allocator, par: kernels.Parallel, model: Spec.Layers, data: Dataset, mirrored_too: bool) !f64 {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const host_logits = try gpa.alloc(f32, Spec.eval_batch * Spec.classes);
            defer gpa.free(host_logits);
            const mirror_logits = try gpa.alloc(f32, Spec.eval_batch * Spec.classes);
            defer gpa.free(mirror_logits);
            const host_labels = try gpa.alloc(u8, data.count());
            defer gpa.free(host_labels);
            if (par.gpu) |dev| try dev.download(host_labels, data.labels) else @memcpy(host_labels, data.labels);

            const mirrored: []f32 = if (!mirrored_too) &.{} else if (par.gpu) |dev| try dev.alloc(f32, Spec.eval_batch * pixels) else try gpa.alloc(f32, Spec.eval_batch * pixels);
            defer if (par.gpu == null) gpa.free(mirrored);
            var identity: []u32 = &.{};
            var scratch_labels: [2][]u8 = .{ &.{}, &.{} };
            if (mirrored_too and par.gpu != null) {
                const dev = par.gpu.?;
                const host_order = try gpa.alloc(u32, data.count());
                defer gpa.free(host_order);
                for (host_order, 0..) |*o, i| o.* = @intCast(i);
                identity = try dev.alloc(u32, host_order.len);
                try dev.upload(identity, host_order);
                for (&scratch_labels) |*l| l.* = try dev.alloc(u8, Spec.eval_batch);
            }
            const flip: augment_math.Config = .{ .mirror = true, .channels = Spec.channels };

            var correct: usize = 0;
            var start: usize = 0;
            while (start + Spec.eval_batch <= data.count()) : (start += Spec.eval_batch) {
                try scores(par, model, &arena, host_logits, data.images[start * pixels ..][0 .. Spec.eval_batch * pixels]);
                if (mirrored_too) {
                    if (par.gpu) |dev| {
                        try dev.gather(.{ .images = data.images, .labels = data.labels, .order = identity }, .{ .batch = mirrored, .labels = scratch_labels[0], .other_labels = scratch_labels[1] }, start, flip);
                    } else {
                        const transform = augment_math.transformOf(0, flip);
                        for (mirrored, 0..) |*v, i| v.* = augment_math.pixel(data.images[start * pixels + i / pixels * pixels ..].ptr, i % pixels, Spec.side, Spec.channels, transform);
                    }
                    try scores(par, model, &arena, mirror_logits, mirrored);
                    for (host_logits, mirror_logits) |*a, b| a.* += b;
                }

                for (0..Spec.eval_batch) |row| {
                    const row_scores = host_logits[row * Spec.classes ..][0..Spec.classes];
                    var best: usize = 0;
                    for (row_scores, 0..) |s, j| {
                        if (s > row_scores[best]) best = j;
                    }
                    correct += @intFromBool(best == host_labels[start + row]);
                }
            }
            const seen = data.count() / Spec.eval_batch * Spec.eval_batch;
            return 100.0 * @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(seen));
        }

        fn predict(gpa: Allocator, par: kernels.Parallel, model: Spec.Layers, data: Dataset, count: usize, out: *Io.Writer) !void {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const labels = try gpa.alloc(u8, data.count());
            defer gpa.free(labels);
            if (par.gpu) |dev| try dev.download(labels, data.labels) else @memcpy(labels, data.labels);
            var row: [Spec.classes]f32 = undefined;
            for (0..@min(count, data.count())) |i| {
                _ = arena.reset(.retain_capacity);
                if (par.gpu) |dev| try dev.resetStep();
                var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
                const x: Tensor(f32, &.{ 1, pixels }) = .{ .data = data.images[i * pixels ..][0..pixels] };
                const logits = try Spec.forward(model, &tape, try tape.constant(x));
                if (par.gpu) |dev| try dev.download(&row, logits.value.data) else @memcpy(&row, logits.value.data);
                var best: usize = 0;
                var total: f32 = 0;
                for (row, 0..) |s, j| {
                    if (s > row[best]) best = j;
                }
                for (row) |s| total += @exp(s - row[best]);
                try out.print("image {d}: label {d}, predicted {d} ({d:.1}%)\n", .{ i, labels[i], best, 100 / total });
                if (i == 0) {
                    try out.writeAll("logits of image 0:");
                    for (row) |s| try out.print(" {d:.5}", .{s});
                    try out.writeByte('\n');
                }
            }
        }

        fn exportOnnx(gpa: Allocator, io: Io, par: kernels.Parallel, model: Spec.Layers, path: []const u8) !void {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            if (par.gpu) |dev| try dev.resetStep();
            var trace: onnx.Trace = .{ .gpa = gpa };
            defer trace.deinit();
            var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .trace = &trace };
            const input = try tape.constant(try tape.zeroedTensor(Tensor(f32, &.{ 1, pixels })));
            const logits = try Spec.forward(model, &tape, input);

            const slices = nn.state(model);
            var weights: [slices.len]onnx.Weight = undefined;
            var copied: usize = 0;
            defer for (weights[0..copied]) |w| gpa.free(w.data);
            inline for (slices, 0..) |slice, i| {
                const host = try gpa.alloc(f32, slice.len);
                weights[i] = .{ .id = @intFromPtr(slice.ptr), .data = host };
                copied += 1;
                if (par.gpu) |dev| try dev.download(host, slice) else @memcpy(host, slice);
            }
            try onnx.save(gpa, io, path, .{ .trace = &trace, .weights = &weights, .inputs = &.{onnx.View.of(input)}, .outputs = &.{onnx.View.of(logits)} });
        }

        fn buffer(comptime T: type, gpa: Allocator, gpu: ?*Gpu, n: usize) ![]T {
            return if (gpu) |dev| dev.alloc(T, n) else gpa.alloc(T, n);
        }

        const Feed = struct {
            gpu: ?*Gpu,
            host: Dataset,
            device: Dataset,
            order: []u32,
            order_device: []u32,
            batch: Tensor(f32, &.{ batch_size, pixels }),
            labels: []u8,
            other_labels: []u8,

            fn init(gpa: Allocator, gpu: ?*Gpu, host: Dataset, device: Dataset) !Feed {
                const order = try gpa.alloc(u32, host.count());
                return .{
                    .gpu = gpu,
                    .host = host,
                    .device = device,
                    .order = order,
                    .order_device = if (gpu) |dev| try dev.alloc(u32, order.len) else order,
                    .batch = .{ .data = try buffer(f32, gpa, gpu, batch_size * pixels) },
                    .labels = try buffer(u8, gpa, gpu, batch_size),
                    .other_labels = try buffer(u8, gpa, gpu, batch_size),
                };
            }

            fn deinit(self: Feed, gpa: Allocator) void {
                gpa.free(self.order);
                if (self.gpu != null) return;
                gpa.free(self.batch.data);
                gpa.free(self.labels);
                gpa.free(self.other_labels);
            }

            fn shuffle(self: Feed, epoch: usize) !void {
                for (self.order, 0..) |*o, i| o.* = @intCast(i);
                var prng = std.Random.DefaultPrng.init(1234 + epoch);
                prng.random().shuffle(u32, self.order);
                if (self.gpu) |dev| try dev.upload(self.order_device, self.order);
            }

            fn fill(self: Feed, step: usize, augment: augment_math.Config) !void {
                if (self.gpu) |dev| return dev.gather(.{ .images = self.device.images, .labels = self.device.labels, .order = self.order_device }, .{ .batch = self.batch.data, .labels = self.labels, .other_labels = self.other_labels }, step * batch_size, augment);
                const rows = self.order[step * batch_size ..][0..batch_size];
                for (rows, 0..) |idx, row| {
                    const partner = (row + 1) % batch_size;
                    const own = augment_math.transformOf(step * batch_size + row, augment);
                    const other = augment_math.transformOf(step * batch_size + partner, augment);
                    for (self.batch.data[row * pixels ..][0..pixels], 0..) |*v, p| {
                        const pasted = augment.box.contains(p, Spec.side, Spec.channels);
                        const source = self.host.images[(if (pasted) rows[partner] else idx) * pixels ..];
                        v.* = if (augment.active()) augment_math.pixel(source.ptr, p, Spec.side, Spec.channels, if (pasted) other else own) else source[p];
                    }
                    self.labels[row] = self.host.labels[idx];
                    self.other_labels[row] = self.host.labels[rows[partner]];
                }
            }
        };

        fn mixedLoss(tape: *Tape, logits: anytype, labels: []const u8, other_labels: []const u8, kept: f32) autograd.Error!autograd.Var(Tensor(f32, &.{})) {
            if (kept == 1) return tape.crossEntropy(logits, labels);
            const Scalar = Tensor(f32, &.{});
            return tape.add(
                try tape.mul(try tape.crossEntropy(logits, labels), try tape.constant(try tape.filledTensor(Scalar, kept))),
                try tape.mul(try tape.crossEntropy(logits, other_labels), try tape.constant(try tape.filledTensor(Scalar, 1 - kept))),
            );
        }

        fn run(init: std.process.Init) !void {
            const io = init.io;
            const gpa = init.gpa;

            var stdout_buffer: [1024]u8 = undefined;
            var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
            const out = &stdout_file_writer.interface;

            kernels.flushDenormals();
            const args = try init.minimal.args.toSlice(init.arena.allocator());
            const opts = try Options.parse(args[1..], Spec.channels);

            const train = try Spec.Loader.load(gpa, io, opts.data_dir orelse Spec.data_dir, "train");
            defer train.deinit(gpa);
            const test_set = try Spec.Loader.load(gpa, io, opts.data_dir orelse Spec.data_dir, "t10k");
            defer test_set.deinit(gpa);
            if (train.pixels != pixels) return error.UnexpectedImageSize;

            const gpu: ?*Gpu = if (opts.gpu) try Gpu.init(gpa, Spec.scratch_bytes) else null;
            defer if (gpu) |dev| dev.deinit();
            if (gpu) |dev| {
                if (opts.profile) dev.startProfile(io);
                dev.tensor_cores = opts.tensor_cores;
                dev.tf32 = opts.tf32;
                dev.deterministic = opts.deterministic;
                if (opts.cudnn_dir) |dir| dev.enableCudnn(dir) catch |err| try out.print("cuDNN unavailable ({s}), using the built-in kernels\n", .{@errorName(err)});
            }
            const par: kernels.Parallel = .{ .io = io, .threads = std.Thread.getCpuCount() catch 1, .gpu = gpu };

            var prng = std.Random.DefaultPrng.init(1234);
            const host_model = try nn.initAll(Spec.Layers, gpa, prng.random());
            defer nn.deinitAll(host_model, gpa);
            const model = if (gpu) |dev| try nn.toGpuAll(host_model, dev) else host_model;
            const params = nn.params(model);
            const learning_rate = opts.learning_rate orelse Spec.learning_rate;
            var adam = try optim.Adam.init(gpa, par, learning_rate, params);
            defer adam.deinit();
            adam.weight_decay = Spec.weight_decay;
            var average = try optim.Ema.init(gpa, par, nn.state(model));
            defer average.deinit();
            average.decay = opts.ema_decay;
            var start_epoch: usize = 0;
            if (opts.load_path) |path| {
                var counters: [2]u32 = undefined;
                try checkpoint.load(gpa, io, path, gpu, nn.state(model) ++ adam.state() ++ average.state(), &counters);
                start_epoch = counters[0];
                adam.t = counters[1];
            }

            const train_data = if (gpu) |dev| try train.toGpu(dev) else train;
            const test_data = if (gpu) |dev| try test_set.toGpu(dev) else test_set;
            if (opts.load_path != null) try out.print("loaded checkpoint after {d} epochs, test accuracy {d:.2}%\n", .{ start_epoch, try accuracy(gpa, par, model, test_data, false) });

            const feed = try Feed.init(gpa, gpu, train, train_data);
            defer feed.deinit(gpa);
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const loss_total: []f32 = if (gpu) |dev| try dev.alloc(f32, 1) else &.{};

            try out.print("{s}, AdamW, batch {d}, {s}, {d} training images\n", .{ Spec.name, batch_size, if (gpu != null) "GPU" else "CPU", train.count() });
            try out.flush();

            const epochs = opts.epochs;
            const steps = train.count() / batch_size;
            const stop_epoch = @min(opts.until orelse epochs, epochs);
            const live = opts.progress or try Io.File.stdout().isTty(io);
            if (live) Io.File.stdout().enableAnsiEscapeCodes(io) catch {};
            const progress_every = @max(1, steps / 100);
            for (start_epoch..@max(start_epoch, stop_epoch)) |epoch| {
                const started = Io.Clock.awake.now(io);
                try feed.shuffle(epoch);
                var loss_sum: f64 = 0;
                if (gpu) |dev| try dev.fill(loss_total, 0);
                var augment = opts.augment;
                augment.seed = epoch;
                for (0..steps) |step| {
                    const global_step = epoch * steps + step;
                    const mix = drawMix(Spec.side, opts.cutmix, 0x5EED0000 + global_step);
                    augment.box = mix.box;
                    try feed.fill(step, augment);

                    _ = arena.reset(.retain_capacity);
                    if (gpu) |dev| try dev.resetStep();
                    adam.lr = optim.cosineLr(learning_rate, global_step, epochs * steps, opts.warmup_epochs * steps);
                    var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true, .smoothing = opts.smoothing, .seed = global_step * 8 };
                    try optim.zeroGrad(par, params);
                    const logits = try Spec.forward(model, &tape, try tape.constant(feed.batch));
                    const loss = try mixedLoss(&tape, logits, feed.labels, feed.other_labels, mix.kept);
                    try tape.backward(loss);
                    try adam.step(par, params);
                    if (opts.ema_decay > 0) try average.update(par, adam.t);
                    if (gpu) |dev| try dev.addInPlace(loss_total, loss.value.data) else loss_sum += loss.value.data[0];
                    if (live and ((step + 1) % progress_every == 0 or step + 1 == steps)) {
                        const so_far = if (gpu) |dev| try dev.scalar(loss_total) else loss_sum;
                        try printProgress(out, .{ .epoch = epoch, .epochs = epochs, .done = step + 1, .steps = steps, .loss = so_far / @as(f64, @floatFromInt(step + 1)), .elapsed_ns = Io.Clock.awake.now(io).nanoseconds - started.nanoseconds, .epochs_left = stop_epoch - epoch - 1 });
                    }
                }
                if (gpu) |dev| loss_sum = try dev.scalar(loss_total);

                const seconds = @as(f64, @floatFromInt(Io.Clock.awake.now(io).nanoseconds - started.nanoseconds)) / 1e9;
                const acc = try accuracy(gpa, par, model, test_data, false);
                if (live) try out.writeAll("\r\x1b[2K");
                try out.print("epoch {d}/{d}  train loss {d:.4}  test accuracy {d:.2}%", .{ epoch + 1, epochs, loss_sum / @as(f64, @floatFromInt(steps)), acc });
                if (opts.ema_decay > 0) {
                    try average.swap(par);
                    try out.print("  ema {d:.2}%", .{try accuracy(gpa, par, model, test_data, false)});
                    try average.swap(par);
                }
                try out.print("  ({d:.2} s)\n", .{seconds});
                try out.flush();
            }
            if (opts.mirrored) {
                try out.print("final test accuracy {d:.2}%, with mirrored inputs averaged {d:.2}%\n", .{ try accuracy(gpa, par, model, test_data, false), try accuracy(gpa, par, model, test_data, true) });
                if (opts.ema_decay > 0) {
                    try average.swap(par);
                    try out.print("ema   test accuracy {d:.2}%, with mirrored inputs averaged {d:.2}%\n", .{ try accuracy(gpa, par, model, test_data, false), try accuracy(gpa, par, model, test_data, true) });
                    try average.swap(par);
                }
            }
            if (opts.predict > 0) try predict(gpa, par, model, test_data, opts.predict, out);
            if (opts.onnx_path) |path| {
                try exportOnnx(gpa, io, par, model, path);
                try out.print("exported {s}\n", .{path});
            }
            if (opts.save_path) |path| {
                const counters = [_]u32{ @intCast(@max(stop_epoch, start_epoch)), adam.t };
                try checkpoint.save(gpa, io, path, gpu, nn.state(model) ++ adam.state() ++ average.state(), &counters);
                try out.print("saved {s}\n", .{path});
            }
            if (gpu) |dev| try dev.printProfile(out);
            try out.flush();
        }
    };
}

/// Runs a whole program for a model description: parses the flags, loads data, trains, evaluates and writes the optional outputs.
/// Spec supplies the layers, the forward pass, the data loader and the batch sizes it was built for; --batch picks one of them.
pub fn run(comptime Spec: type, init: std.process.Init) !void {
    const allowed = if (@hasDecl(Spec, "batches")) Spec.batches else [_]usize{Spec.batch};
    var wanted: usize = Spec.batch;
    for ((try init.minimal.args.toSlice(init.arena.allocator()))[1..]) |arg| {
        if (valueOf(arg, "--batch")) |v| wanted = try std.fmt.parseInt(usize, v, 10);
    }
    inline for (allowed) |size| {
        if (size == wanted) return Runner(Spec, size).run(init);
    }
    std.debug.print("--batch={d} is not available in this build; it supports", .{wanted});
    inline for (allowed) |size| std.debug.print(" {d}", .{size});
    std.debug.print("\n", .{});
    return error.UnsupportedBatchSize;
}
