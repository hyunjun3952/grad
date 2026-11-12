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
const replicas_mod = @import("replicas.zig");
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
    scratch_mib: ?usize = null,
    replicas: usize = 1,
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
            } else if (valueOf(arg, "--scratch")) |v| {
                o.scratch_mib = try std.fmt.parseInt(usize, v, 10);
            } else if (valueOf(arg, "--replicas")) |v| {
                o.replicas = try std.fmt.parseInt(usize, v, 10);
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
            try par.download(host, logits.value.data);
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
            try par.download(host_labels, data.labels);

            const mirrored: []f32 = if (mirrored_too) try par.alloc(f32, gpa, Spec.eval_batch * pixels) else &.{};
            defer par.free(gpa, mirrored);
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
            try par.download(labels, data.labels);
            var row: [Spec.classes]f32 = undefined;
            for (0..@min(count, data.count())) |i| {
                _ = arena.reset(.retain_capacity);
                if (par.gpu) |dev| try dev.resetStep();
                var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
                const x: Tensor(f32, &.{ 1, pixels }) = .{ .data = data.images[i * pixels ..][0..pixels] };
                const logits = try Spec.forward(model, &tape, try tape.constant(x));
                try par.download(&row, logits.value.data);
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
            // NOTE: traced with two images, not one, so that the leading axis of the exported graph can be told from a size-one axis; it is dynamic.
            const input = try tape.constant(try tape.zeroedTensor(Tensor(f32, &.{ 2, pixels })));
            const logits = try Spec.forward(model, &tape, input);

            const slices = nn.state(model);
            const weights = try onnx.WeightSet(slices.len).collect(gpa, par, slices);
            defer weights.deinit(gpa);
            try onnx.save(gpa, io, path, .{ .trace = &trace, .weights = &weights.items, .inputs = &.{onnx.View.of(input)}, .outputs = &.{onnx.View.of(logits)}, .dynamic_batch = true });
        }

        const Feed = struct {
            par: kernels.Parallel,
            host: Dataset,
            device: Dataset,
            order: []u32,
            order_device: []u32,
            batch: Tensor(f32, &.{ batch_size, pixels }),
            labels: []u8,
            other_labels: []u8,

            // NOTE: on the CPU the shuffled order is read in place, so order_device aliases order.
            fn init(gpa: Allocator, par: kernels.Parallel, host: Dataset, device: Dataset) !Feed {
                const order = try gpa.alloc(u32, host.count());
                errdefer gpa.free(order);
                const order_device = if (par.gpu) |dev| try dev.alloc(u32, order.len) else order;
                const batch = try par.alloc(f32, gpa, batch_size * pixels);
                errdefer par.free(gpa, batch);
                const labels = try par.alloc(u8, gpa, batch_size);
                errdefer par.free(gpa, labels);
                return .{
                    .par = par,
                    .host = host,
                    .device = device,
                    .order = order,
                    .order_device = order_device,
                    .batch = .{ .data = batch },
                    .labels = labels,
                    .other_labels = try par.alloc(u8, gpa, batch_size),
                };
            }

            fn deinit(self: Feed, gpa: Allocator) void {
                gpa.free(self.order);
                self.par.free(gpa, self.batch.data);
                self.par.free(gpa, self.labels);
                self.par.free(gpa, self.other_labels);
            }

            fn shuffle(self: Feed, epoch: usize) !void {
                for (self.order, 0..) |*o, i| o.* = @intCast(i);
                var prng = std.Random.DefaultPrng.init(1234 + epoch);
                prng.random().shuffle(u32, self.order);
                if (self.par.gpu) |dev| try dev.upload(self.order_device, self.order);
            }

            // Fills the batch with the slice number slice of the shuffled order; slices are batch_size rows long.
            fn fill(self: Feed, slice: usize, augment: augment_math.Config) !void {
                if (self.par.gpu) |dev| return dev.gather(
                    .{ .images = self.device.images, .labels = self.device.labels, .order = self.order_device },
                    .{ .batch = self.batch.data, .labels = self.labels, .other_labels = self.other_labels },
                    slice * batch_size,
                    augment,
                );
                const rows = self.order[slice * batch_size ..][0..batch_size];
                for (rows, 0..) |idx, row| {
                    const partner = (row + 1) % batch_size;
                    const own = augment_math.transformOf(slice * batch_size + row, augment);
                    const other = augment_math.transformOf(slice * batch_size + partner, augment);
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

        // The CutMix loss: kept times the loss on the labels plus 1 - kept times the loss on the pasted images' labels.
        // weights, when given, is a device buffer holding kept and 1 - kept that is refilled every step; a recorded step
        // uses it so that the graph does not bake the share in, and then always has the mixed form.
        fn mixedLoss(tape: *Tape, logits: anytype, labels: []const u8, other_labels: []const u8, kept: f32, weights: ?[]f32) autograd.Error!autograd.Var(Tensor(f32, &.{})) {
            if (kept == 1 and weights == null) return tape.crossEntropy(logits, labels);
            const Scalar = Tensor(f32, &.{});
            const kept_weight: Scalar = if (weights) |w| .{ .data = w[0..1] } else try tape.filledTensor(Scalar, kept);
            const other_weight: Scalar = if (weights) |w| .{ .data = w[1..2] } else try tape.filledTensor(Scalar, 1 - kept);
            return tape.add(
                try tape.mul(try tape.crossEntropy(logits, labels), try tape.constant(kept_weight)),
                try tape.mul(try tape.crossEntropy(logits, other_labels), try tape.constant(other_weight)),
            );
        }

        // State shared by all replica threads. Datasets are read-only; every replica copies what it needs to its own device.
        const Shared = struct {
            gpa: Allocator,
            io: Io,
            opts: Options,
            train: Dataset,
            test_set: Dataset,
            group: replicas_mod.Replicas,
            errors: []?anyerror,
            out: *Io.Writer,
        };

        fn workMain(shared: *Shared, index: usize) void {
            work(shared, index) catch |err| {
                shared.errors[index] = err;
                shared.group.stop();
            };
        }

        fn initGpu(shared: *Shared, index: usize) !?*Gpu {
            const opts = shared.opts;
            // NOTE: --scratch=N reserves N MiB for each arena instead of Spec.scratch_bytes and a quarter of it; the zero arena
            // holds the gradients, which need about as much as the activations in bf16, hence the same size for both.
            const scratch_bytes: usize = if (opts.scratch_mib) |mib| mib << 20 else Spec.scratch_bytes;
            const zero_bytes: ?usize = if (opts.scratch_mib) |mib| mib << 20 else null;
            const gpu: ?*Gpu = if (opts.gpu) try Gpu.init(shared.gpa, .{
                .step_bytes = scratch_bytes,
                .zero_bytes = zero_bytes,
                .device = index % @max(1, Gpu.deviceCount()),
                .tensor_cores = opts.tensor_cores,
                .tf32 = opts.tf32,
                .deterministic = opts.deterministic,
            }) else null;
            if (gpu) |dev| {
                if (opts.profile and index == 0) dev.startProfile(shared.io);
                if (opts.cudnn_dir) |dir| dev.enableCudnn(dir) catch |err| if (index == 0) try shared.out.print("cuDNN unavailable ({s}), using the built-in kernels\n", .{@errorName(err)});
            }
            return gpu;
        }

        const ModelState = struct {
            host_model: Spec.Layers,
            model: Spec.Layers,
            adam: optim.Adam,
            average: optim.Ema,
            learning_rate: f32,

            fn init(shared: *Shared, gpu: ?*Gpu, par: kernels.Parallel) !ModelState {
                const gpa = shared.gpa;
                var prng = std.Random.DefaultPrng.init(1234);
                const host_model = try nn.initAll(Spec.Layers, gpa, prng.random());
                errdefer nn.deinitAll(host_model, gpa);
                const model = if (gpu) |dev| try nn.toGpuAll(host_model, dev) else host_model;
                const learning_rate = shared.opts.learning_rate orelse Spec.learning_rate;
                var adam = try optim.Adam.init(gpa, par, learning_rate, nn.params(model));
                errdefer adam.deinit();
                adam.weight_decay = Spec.weight_decay;
                var average = try optim.Ema.init(gpa, par, nn.state(model));
                average.decay = shared.opts.ema_decay;
                return .{ .host_model = host_model, .model = model, .adam = adam, .average = average, .learning_rate = learning_rate };
            }

            fn deinit(self: ModelState, gpa: Allocator) void {
                self.average.deinit();
                self.adam.deinit();
                nn.deinitAll(self.host_model, gpa);
            }

            fn loadCheckpoint(self: *ModelState, shared: *Shared, gpu: ?*Gpu) !usize {
                const path = shared.opts.load_path orelse return 0;
                var counters: [2]u32 = undefined;
                try checkpoint.load(shared.gpa, shared.io, path, gpu, nn.state(self.model) ++ self.adam.state() ++ self.average.state(), &counters);
                self.adam.t = counters[1];
                return counters[0];
            }
        };

        const Loop = struct {
            shared: *Shared,
            index: usize,
            gpu: ?*Gpu,
            par: kernels.Parallel,
            state: *ModelState,
            start_epoch: usize,
            train_data: Dataset,
            test_data: Dataset,
            feed: Feed,
            arena: *std.heap.ArenaAllocator,
            loss_total: []f32,
            use_graph: bool,
            mix_weights: ?[]f32,
            graph: ?Gpu.Graph = null,
            steps_run: usize = 0,
            graph_loss: autograd.Var(Tensor(f32, &.{})) = undefined,
            loss_sum: f64 = 0,

            const StepInput = struct { global_step: usize, feed_slice: usize, augment: augment_math.Config, kept: f32 };

            fn trainStep(self: *Loop, input: StepInput) !void {
                const shared = self.shared;
                const opts = shared.opts;
                const state = self.state;
                const params = nn.params(state.model);
                try self.feed.fill(input.feed_slice, input.augment);

                const steps = shared.train.count() / (batch_size * opts.replicas);
                state.adam.lr = optim.cosineLr(state.learning_rate, input.global_step, opts.epochs * steps, opts.warmup_epochs * steps);
                // NOTE: dropout seeds are (global_step * replicas + index) * 8 plus the op number; a recorded step keeps
                // index * 8 on the tape and takes the step part from the device, which gives the same masks.
                const step_seed = input.global_step * opts.replicas * 8;
                if (self.use_graph) {
                    const dev = self.gpu.?;
                    try dev.setStepSeed(step_seed);
                    if (self.mix_weights) |w| try dev.upload(w, &[2]f32{ input.kept, 1 - input.kept });
                }
                const record = self.use_graph and self.graph == null and (opts.cudnn_dir == null or self.steps_run > 0);
                if (!self.use_graph or self.graph == null) {
                    _ = self.arena.reset(.retain_capacity);
                    if (self.gpu) |dev| {
                        try dev.resetStep();
                        if (record) try dev.beginCapture();
                    }
                    var tape: Tape = .{ .gpa = self.arena.allocator(), .par = self.par, .training = true, .smoothing = opts.smoothing, .seed = self.index * 8 + if (self.use_graph) 0 else step_seed };
                    try optim.zeroGrad(self.par, params);
                    const logits = try Spec.forward(state.model, &tape, try tape.constant(self.feed.batch));
                    self.graph_loss = try mixedLoss(&tape, logits, self.feed.labels, self.feed.other_labels, input.kept, self.mix_weights);
                    try tape.backward(self.graph_loss);
                    if (record) self.graph = try self.gpu.?.endCapture();
                }
                if (self.graph) |recorded| try self.gpu.?.replay(recorded);
                self.steps_run += 1;
                try shared.group.averageGrads(self.index, self.par, params);
                try state.adam.step(self.par, params);
                if (opts.ema_decay > 0) try state.average.update(self.par, state.adam.t);
                if (self.gpu) |dev| try dev.addInPlace(self.loss_total, self.graph_loss.value.data) else self.loss_sum += self.graph_loss.value.data[0];
            }

            fn finishEpoch(self: *Loop, epoch: usize, live: bool, started_ns: i96) !void {
                const opts = self.shared.opts;
                const out = self.shared.out;
                const state = self.state;
                if (self.gpu) |dev| self.loss_sum = try dev.scalar(self.loss_total);
                const seconds = @as(f64, @floatFromInt(Io.Clock.awake.now(self.shared.io).nanoseconds - started_ns)) / 1e9;
                const acc = try accuracy(self.shared.gpa, self.par, state.model, self.test_data, false);
                if (live) try out.writeAll("\r\x1b[2K");
                const steps = self.shared.train.count() / (batch_size * opts.replicas);
                try out.print("epoch {d}/{d}  train loss {d:.4}  test accuracy {d:.2}%", .{ epoch + 1, opts.epochs, self.loss_sum / @as(f64, @floatFromInt(steps)), acc });
                if (opts.ema_decay > 0) {
                    try state.average.swap(self.par);
                    try out.print("  ema {d:.2}%", .{try accuracy(self.shared.gpa, self.par, state.model, self.test_data, false)});
                    try state.average.swap(self.par);
                }
                try out.print("  ({d:.2} s)\n", .{seconds});
                try out.flush();
            }

            fn run(self: *Loop) !void {
                const shared = self.shared;
                const opts = shared.opts;
                const epochs = opts.epochs;
                const steps = shared.train.count() / (batch_size * opts.replicas);
                const stop_epoch = @min(opts.until orelse epochs, epochs);
                const live = self.index == 0 and (opts.progress or try Io.File.stdout().isTty(shared.io));
                if (live) Io.File.stdout().enableAnsiEscapeCodes(shared.io) catch {};
                const progress_every = @max(1, steps / 100);
                for (self.start_epoch..@max(self.start_epoch, stop_epoch)) |epoch| {
                    const started = Io.Clock.awake.now(shared.io);
                    try self.feed.shuffle(epoch);
                    self.loss_sum = 0;
                    if (self.gpu) |dev| try dev.fill(self.loss_total, 0);
                    var augment = opts.augment;
                    augment.seed = epoch;
                    for (0..steps) |step| {
                        const global_step = epoch * steps + step;
                        const mix = drawMix(Spec.side, opts.cutmix, 0x5EED0000 + global_step);
                        augment.box = mix.box;
                        // NOTE: replica r takes slice step * replicas + r, so together the replicas see the rows a single replica
                        // with the whole batch would. CutMix pairs rows inside a slice only.
                        try self.trainStep(.{ .global_step = global_step, .feed_slice = step * opts.replicas + self.index, .augment = augment, .kept = mix.kept });
                        if (live and ((step + 1) % progress_every == 0 or step + 1 == steps)) {
                            const so_far = if (self.gpu) |dev| try dev.scalar(self.loss_total) else self.loss_sum;
                            try printProgress(shared.out, .{
                                .epoch = epoch,
                                .epochs = epochs,
                                .done = step + 1,
                                .steps = steps,
                                .loss = so_far / @as(f64, @floatFromInt(step + 1)),
                                .elapsed_ns = Io.Clock.awake.now(shared.io).nanoseconds - started.nanoseconds,
                                .epochs_left = stop_epoch - epoch - 1,
                            });
                        }
                    }
                    if (self.index == 0) try self.finishEpoch(epoch, live, started.nanoseconds);
                }
            }

            fn finish(self: *Loop) !void {
                const shared = self.shared;
                const opts = shared.opts;
                const out = shared.out;
                const state = self.state;
                // NOTE: only replica 0 evaluates and writes outputs. Its loss is the mean over its own slice, and batch norm
                // statistics are those of its slices, so they differ slightly from a single replica with the whole batch.
                if (self.index != 0) return;
                if (opts.mirrored) {
                    try out.print("final test accuracy {d:.2}%, with mirrored inputs averaged {d:.2}%\n", .{ try accuracy(shared.gpa, self.par, state.model, self.test_data, false), try accuracy(shared.gpa, self.par, state.model, self.test_data, true) });
                    if (opts.ema_decay > 0) {
                        try state.average.swap(self.par);
                        try out.print("ema   test accuracy {d:.2}%, with mirrored inputs averaged {d:.2}%\n", .{ try accuracy(shared.gpa, self.par, state.model, self.test_data, false), try accuracy(shared.gpa, self.par, state.model, self.test_data, true) });
                        try state.average.swap(self.par);
                    }
                }
                if (opts.predict > 0) try predict(shared.gpa, self.par, state.model, self.test_data, opts.predict, out);
                if (opts.onnx_path) |path| {
                    try exportOnnx(shared.gpa, shared.io, self.par, state.model, path);
                    try out.print("exported {s}\n", .{path});
                }
                if (opts.save_path) |path| {
                    const stop_epoch = @min(opts.until orelse opts.epochs, opts.epochs);
                    const counters = [_]u32{ @intCast(@max(stop_epoch, self.start_epoch)), state.adam.t };
                    try checkpoint.save(shared.gpa, shared.io, path, self.gpu, nn.state(state.model) ++ state.adam.state() ++ state.average.state(), &counters);
                    try out.print("saved {s}\n", .{path});
                }
                if (self.gpu) |dev| {
                    const peak = dev.peakUsage();
                    try out.print("device scratch peak {d} MiB of {d} MiB, zero arena peak {d} MiB of {d} MiB\n", .{ peak.step >> 20, dev.step_size >> 20, peak.zero >> 20, dev.zero_size >> 20 });
                    try dev.printProfile(out);
                }
                try out.flush();
            }
        };

        // Trains one replica. Replica 0 runs on the main thread, owns the console output, the evaluation and the optional outputs.
        // NOTE: every replica builds its own Gpu inside its thread, because a CUDA context is current per thread.
        fn work(shared: *Shared, index: usize) !void {
            const gpa = shared.gpa;
            const opts = shared.opts;
            const out = shared.out;
            const gpu = try initGpu(shared, index);
            defer if (gpu) |dev| dev.deinit();
            // NOTE: replica threads must not share the CPU worker pool, which is a single global one.
            const par: kernels.Parallel = .{ .io = if (opts.replicas > 1) null else shared.io, .threads = std.Thread.getCpuCount() catch 1, .gpu = gpu };

            var state = try ModelState.init(shared, gpu, par);
            defer state.deinit(gpa);
            const start_epoch = try state.loadCheckpoint(shared, gpu);
            const train_data = if (gpu) |dev| try shared.train.toGpu(dev) else shared.train;
            const test_data = if (gpu) |dev| try shared.test_set.toGpu(dev) else shared.test_set;
            if (opts.load_path != null and index == 0) try out.print("loaded checkpoint after {d} epochs, test accuracy {d:.2}%\n", .{ start_epoch, try accuracy(gpa, par, state.model, test_data, false) });

            const feed = try Feed.init(gpa, par, shared.train, train_data);
            defer feed.deinit(gpa);
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const loss_total: []f32 = if (gpu) |dev| try dev.alloc(f32, 1) else &.{};
            if (index == 0) {
                try out.print("{s}, AdamW, batch {d}, {s}, {d} training images", .{ Spec.name, batch_size * opts.replicas, if (gpu != null) "GPU" else "CPU", shared.train.count() });
                if (opts.replicas > 1) try out.print(", {d} replicas of {d}", .{ opts.replicas, batch_size });
                try out.writeByte('\n');
                try out.flush();
            }

            // NOTE: a training step is the same sequence of launches every time, so on the GPU the forward and backward pass is
            // recorded once and replayed. What changes per step stays outside or in device buffers: the batch is gathered in
            // place, the dropout seed offset and the CutMix weights are uploaded before each replay, and Adam, EMA and the
            // loss sum run after it. Replicas record too, since their gradient exchange happens after the replay. Only profiling
            // cannot record, because it waits for the device after every launch, so a profiled run launches every step normally.
            // cuDNN chooses its algorithms by timing real convolutions the first time it sees a shape, which a recording cannot
            // do, so with it the first step runs normally and the second one is recorded.
            const use_graph = gpu != null and !opts.profile;
            const mix_weights: ?[]f32 = if (use_graph and opts.cutmix > 0) try gpu.?.alloc(f32, 2) else null;
            var loop: Loop = .{
                .shared = shared,
                .index = index,
                .gpu = gpu,
                .par = par,
                .state = &state,
                .start_epoch = start_epoch,
                .train_data = train_data,
                .test_data = test_data,
                .feed = feed,
                .arena = &arena,
                .loss_total = loss_total,
                .use_graph = use_graph,
                .mix_weights = mix_weights,
            };
            try loop.run();
            try loop.finish();
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
            if (opts.replicas > 1 and !opts.gpu) return error.ReplicasNeedGpu;

            const train = try Spec.Loader.load(gpa, io, opts.data_dir orelse Spec.data_dir, "train");
            defer train.deinit(gpa);
            const test_set = try Spec.Loader.load(gpa, io, opts.data_dir orelse Spec.data_dir, "t10k");
            defer test_set.deinit(gpa);
            if (train.pixels != pixels) return error.UnexpectedImageSize;

            // The gradient buffer length comes from a throwaway host model, so the barrier state can exist before any replica does.
            const grad_len = blk: {
                var seed = std.Random.DefaultPrng.init(1234);
                const probe = try nn.initAll(Spec.Layers, gpa, seed.random());
                defer nn.deinitAll(probe, gpa);
                break :blk replicas_mod.gradLen(nn.params(probe));
            };
            var shared: Shared = .{
                .gpa = gpa,
                .io = io,
                .opts = opts,
                .train = train,
                .test_set = test_set,
                .group = try replicas_mod.Replicas.init(gpa, opts.replicas, grad_len),
                .errors = try gpa.alloc(?anyerror, opts.replicas),
                .out = out,
            };
            defer shared.group.deinit();
            defer gpa.free(shared.errors);
            @memset(shared.errors, null);

            const threads = try gpa.alloc(std.Thread, opts.replicas - 1);
            defer gpa.free(threads);
            var spawned: usize = 0;
            while (spawned < threads.len) : (spawned += 1) {
                threads[spawned] = std.Thread.spawn(.{}, workMain, .{ &shared, spawned + 1 }) catch |err| {
                    shared.group.stop();
                    for (threads[0..spawned]) |t| t.join();
                    return err;
                };
            }
            workMain(&shared, 0);
            for (threads) |t| t.join();
            // Stopped only says that another replica failed, so report the first real error.
            for (shared.errors) |e| {
                if (e != null and e.? != error.Stopped) return e.?;
            }
            for (shared.errors) |e| {
                if (e) |err| return err;
            }
        }
    };
}

/// Runs a whole program for a model description: parses the flags, loads data, trains, evaluates and writes the optional outputs.
/// Spec supplies the layers, the forward pass, the data loader and the batch sizes it was built for; --batch picks one of them.
/// scratch_bytes sizes the device scratch arena (the zero arena gets a quarter); --scratch=N overrides both, and a GPU run
/// prints the peak use of each so that N can be chosen.
/// With --replicas=N (GPU only) the batch is split into N slices, one per replica, so batch / N must be one of the built sizes.
pub fn run(comptime Spec: type, init: std.process.Init) !void {
    const allowed = if (@hasDecl(Spec, "batches")) Spec.batches else [_]usize{Spec.batch};
    var total: usize = Spec.batch;
    var replicas: usize = 1;
    for ((try init.minimal.args.toSlice(init.arena.allocator()))[1..]) |arg| {
        if (valueOf(arg, "--batch")) |v| total = try std.fmt.parseInt(usize, v, 10);
        if (valueOf(arg, "--replicas")) |v| replicas = try std.fmt.parseInt(usize, v, 10);
    }
    if (replicas == 0 or total % replicas != 0) return error.BatchNotDivisible;
    const wanted = total / replicas;
    inline for (allowed) |size| {
        if (size == wanted) return Runner(Spec, size).run(init);
    }
    std.debug.print("--batch={d} with {d} replicas needs a per-replica batch of {d}, which is not available in this build; it supports", .{ total, replicas, wanted });
    inline for (allowed) |size| std.debug.print(" {d}", .{size});
    std.debug.print("\n", .{});
    return error.UnsupportedBatchSize;
}
