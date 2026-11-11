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

fn printProgress(out: *Io.Writer, epoch: usize, epochs: usize, done: usize, steps: usize, loss: f64, elapsed_ns: i96, epochs_left: usize) !void {
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

fn Runner(comptime Spec: type) type {
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
                        try dev.gather(data.images, data.labels, identity, mirrored, scratch_labels[0], scratch_labels[1], start, flip);
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
            var tf32 = false;
            var cudnn_dir: ?[:0]const u8 = null;
            var force_progress = false;
            var smoothing: f32 = 0;
            var augment_config: augment_math.Config = .{ .channels = Spec.channels };
            var save_path: ?[]const u8 = null;
            var load_path: ?[]const u8 = null;
            var epochs: usize = 5;
            var until: ?usize = null;
            var warmup_epochs: usize = 0;
            var ema_decay: f32 = 0;
            var cutmix: f32 = 0;
            var report_mirrored = false;
            var data_dir: []const u8 = Spec.data_dir;
            var positional: usize = 0;
            for (args[1..]) |arg| {
                if (std.mem.eql(u8, arg, "--gpu")) {
                    use_gpu = true;
                } else if (std.mem.eql(u8, arg, "--profile")) {
                    profile = true;
                } else if (std.mem.eql(u8, arg, "--progress")) {
                    force_progress = true;
                } else if (std.mem.startsWith(u8, arg, "--smoothing=")) {
                    smoothing = try std.fmt.parseFloat(f32, arg["--smoothing=".len..]);
                } else if (std.mem.eql(u8, arg, "--cudnn")) {
                    cudnn_dir = "data/cudnn";
                } else if (std.mem.startsWith(u8, arg, "--cudnn=")) {
                    cudnn_dir = arg["--cudnn=".len..];
                } else if (std.mem.eql(u8, arg, "--tf32")) {
                    tf32 = true;
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
                } else if (std.mem.startsWith(u8, arg, "--cutout=")) {
                    augment_config.cutout = try std.fmt.parseFloat(f32, arg["--cutout=".len..]);
                } else if (std.mem.eql(u8, arg, "--tta")) {
                    report_mirrored = true;
                } else if (std.mem.startsWith(u8, arg, "--cutmix=")) {
                    cutmix = try std.fmt.parseFloat(f32, arg["--cutmix=".len..]);
                } else if (std.mem.startsWith(u8, arg, "--ema=")) {
                    ema_decay = try std.fmt.parseFloat(f32, arg["--ema=".len..]);
                } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
                    warmup_epochs = try std.fmt.parseInt(usize, arg["--warmup=".len..], 10);
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
                dev.tf32 = tf32;
                if (cudnn_dir) |dir| dev.enableCudnn(dir) catch |err| try out.print("cuDNN unavailable ({s}), using the built-in kernels\n", .{@errorName(err)});
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
            var average = try optim.Ema.init(gpa, par, nn.state(model));
            defer average.deinit();
            average.decay = ema_decay;
            var start_epoch: usize = 0;
            if (load_path) |path| {
                var progress: [2]u32 = undefined;
                try checkpoint.load(gpa, io, path, gpu, nn.state(model) ++ adam.state() ++ average.state(), &progress);
                start_epoch = progress[0];
                adam.t = progress[1];
            }

            const train_data = if (gpu) |dev| try train.toGpu(dev) else train;
            const test_data = if (gpu) |dev| try test_set.toGpu(dev) else test_set;

            if (load_path != null) try out.print("loaded checkpoint after {d} epochs, test accuracy {d:.2}%\n", .{ start_epoch, try accuracy(gpa, par, model, test_data, false) });

            const order = try gpa.alloc(u32, train.count());
            defer gpa.free(order);
            const order_device = if (gpu) |dev| try dev.alloc(u32, order.len) else order;

            const batch: Tensor(f32, &.{ Spec.batch, pixels }) = .{
                .data = if (gpu) |dev| try dev.alloc(f32, Spec.batch * pixels) else try gpa.alloc(f32, Spec.batch * pixels),
            };
            defer if (gpu == null) gpa.free(batch.data);
            var labels_host: [Spec.batch]u8 = undefined;
            const labels: []u8 = if (gpu) |dev| try dev.alloc(u8, Spec.batch) else &labels_host;
            var other_host: [Spec.batch]u8 = undefined;
            const other_labels: []u8 = if (gpu) |dev| try dev.alloc(u8, Spec.batch) else &other_host;
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const loss_total: []f32 = if (gpu) |dev| try dev.alloc(f32, 1) else &.{};

            try out.print("{s}, AdamW, batch {d}, {s}, {d} training images\n", .{ Spec.name, Spec.batch, if (gpu != null) "GPU" else "CPU", train.count() });
            try out.flush();

            const steps = train.count() / Spec.batch;
            const stop_epoch = @min(until orelse epochs, epochs);
            const progress = force_progress or try Io.File.stdout().isTty(io);
            if (progress) Io.File.stdout().enableAnsiEscapeCodes(io) catch {};
            const progress_every = @max(1, steps / 100);
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
                    const mix = drawMix(Spec.side, cutmix, 0x5EED0000 + epoch * steps + step);
                    var step_augment = augment;
                    step_augment.box = mix.box;
                    if (gpu) |dev| {
                        try dev.gather(train_data.images, train_data.labels, order_device, batch.data, labels, other_labels, step * Spec.batch, step_augment);
                    } else {
                        const rows = order[step * Spec.batch ..][0..Spec.batch];
                        for (rows, 0..) |idx, row| {
                            const partner = (row + 1) % Spec.batch;
                            const row_pixels = batch.data[row * pixels ..][0..pixels];
                            const own = augment_math.transformOf(step * Spec.batch + row, step_augment);
                            const other = augment_math.transformOf(step * Spec.batch + partner, step_augment);
                            for (row_pixels, 0..) |*v, p| {
                                const pasted = step_augment.box.contains(p, Spec.side, Spec.channels);
                                const source = train.images[(if (pasted) rows[partner] else idx) * pixels ..];
                                v.* = if (step_augment.active()) augment_math.pixel(source.ptr, p, Spec.side, Spec.channels, if (pasted) other else own) else source[p];
                            }
                            labels[row] = train.labels[idx];
                            other_labels[row] = train.labels[rows[partner]];
                        }
                    }

                    _ = arena.reset(.retain_capacity);
                    if (gpu) |dev| try dev.resetStep();
                    const global_step = epoch * steps + step;
                    adam.lr = optim.cosineLr(Spec.learning_rate, global_step, epochs * steps, warmup_epochs * steps);
                    var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true, .smoothing = smoothing, .seed = global_step * 8 };
                    try optim.zeroGrad(par, params);
                    const logits = try Spec.forward(model, &tape, try tape.constant(batch));
                    const loss = if (mix.kept == 1) try tape.crossEntropy(logits, labels) else try tape.add(
                        try tape.mul(try tape.crossEntropy(logits, labels), try tape.constant(try tape.filledTensor(Tensor(f32, &.{}), mix.kept))),
                        try tape.mul(try tape.crossEntropy(logits, other_labels), try tape.constant(try tape.filledTensor(Tensor(f32, &.{}), 1 - mix.kept))),
                    );
                    try tape.backward(loss);
                    try adam.step(par, params);
                    if (ema_decay > 0) try average.update(par, adam.t);
                    if (gpu) |dev| try dev.addInPlace(loss_total, loss.value.data) else loss_sum += loss.value.data[0];
                    if (progress and ((step + 1) % progress_every == 0 or step + 1 == steps)) {
                        const so_far = if (gpu) |dev| try dev.scalar(loss_total) else loss_sum;
                        try printProgress(out, epoch, epochs, step + 1, steps, so_far / @as(f64, @floatFromInt(step + 1)), Io.Clock.awake.now(io).nanoseconds - started.nanoseconds, stop_epoch - epoch - 1);
                    }
                }
                if (gpu) |dev| loss_sum = try dev.scalar(loss_total);

                const seconds = @as(f64, @floatFromInt(Io.Clock.awake.now(io).nanoseconds - started.nanoseconds)) / 1e9;
                const acc = try accuracy(gpa, par, model, test_data, false);
                var ema_acc: f64 = 0;
                if (ema_decay > 0) {
                    try average.swap(par);
                    ema_acc = try accuracy(gpa, par, model, test_data, false);
                    try average.swap(par);
                }
                if (progress) try out.writeAll("\r\x1b[2K");
                try out.print("epoch {d}/{d}  train loss {d:.4}  test accuracy {d:.2}%", .{ epoch + 1, epochs, loss_sum / @as(f64, @floatFromInt(steps)), acc });
                if (ema_decay > 0) try out.print("  ema {d:.2}%", .{ema_acc});
                try out.print("  ({d:.2} s)\n", .{seconds});
                try out.flush();
            }
            if (report_mirrored) {
                try out.print("final test accuracy {d:.2}%, with mirrored inputs averaged {d:.2}%\n", .{ try accuracy(gpa, par, model, test_data, false), try accuracy(gpa, par, model, test_data, true) });
                if (ema_decay > 0) {
                    try average.swap(par);
                    try out.print("ema   test accuracy {d:.2}%, with mirrored inputs averaged {d:.2}%\n", .{ try accuracy(gpa, par, model, test_data, false), try accuracy(gpa, par, model, test_data, true) });
                    try average.swap(par);
                }
            }
            if (save_path) |path| {
                const counters = [_]u32{ @intCast(@max(stop_epoch, start_epoch)), adam.t };
                try checkpoint.save(gpa, io, path, gpu, nn.state(model) ++ adam.state() ++ average.state(), &counters);
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
