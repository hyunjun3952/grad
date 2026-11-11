const std = @import("std");
const grad = @import("grad");

const nn = grad.nn;
const Tensor = grad.Tensor;
const Tape = grad.autograd.Tape;
const Var = grad.autograd.Var;

const vocab = 128;
const hidden = 256;
const batch = 32;
const unroll = 64;
const Net = struct { nn.Gru(vocab, hidden), nn.Linear(hidden, vocab) };
const Input = Tensor(f32, &.{ batch, vocab });
const State = Tensor(f32, &.{ batch, hidden });

fn step(model: Net, tape: *Tape, x: Input, h: Var(State)) !struct { Var(State), Var(Tensor(f32, &.{ batch, vocab })) } {
    const cell, const out = model;
    const next = try cell.forward(tape, try tape.constant(x), h);
    return .{ next, try out.forward(tape, next) };
}

fn sample(rng: std.Random, logits: []const f32, temperature: f32) u8 {
    var largest: f32 = -std.math.inf(f32);
    for (logits) |l| largest = @max(largest, l);
    var weights: [vocab]f32 = undefined;
    var total: f32 = 0;
    for (logits, &weights) |l, *w| {
        w.* = @exp((l - largest) / temperature);
        total += w.*;
    }
    var pick = rng.float(f32) * total;
    for (weights, 0..) |w, i| {
        pick -= w;
        if (pick <= 0) return @intCast(i);
    }
    return vocab - 1;
}

fn generate(par: grad.kernels.Parallel, gpa: std.mem.Allocator, model: Net, rng: std.Random, out: *std.Io.Writer, seed_text: []const u8, length: usize) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const input = try par.alloc(f32, gpa, batch * vocab);
    const state = try par.alloc(f32, gpa, batch * hidden);
    const logits = try gpa.alloc(f32, batch * vocab);
    defer gpa.free(logits);
    const hidden_host = try gpa.alloc(f32, batch * hidden);
    defer gpa.free(hidden_host);
    @memset(hidden_host, 0);
    var host_input = try gpa.alloc(f32, batch * vocab);
    defer gpa.free(host_input);
    defer par.free(gpa, input);
    defer par.free(gpa, state);

    var current: u8 = seed_text[0];
    try out.writeByte(current);
    var at: usize = 1;
    for (0..length) |_| {
        _ = arena.reset(.retain_capacity);
        if (par.gpu) |dev| try dev.resetStep();
        var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
        @memset(host_input, 0);
        for (0..batch) |row| host_input[row * vocab + current] = 1;
        try par.upload(input, host_input);
        try par.upload(state, hidden_host);
        const next, const scores = try step(model, &tape, .{ .data = input }, try tape.constant(State{ .data = state }));
        try par.download(hidden_host, next.value.data);
        try par.download(logits, scores.value.data);
        if (at < seed_text.len) {
            current = seed_text[at];
            at += 1;
        } else {
            current = sample(rng, logits[0..vocab], 0.8);
        }
        try out.writeByte(current);
    }
    try out.writeByte('\n');
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &writer.interface;

    var steps: usize = 1500;
    var use_gpu = false;
    var profile = false;
    var path: []const u8 = "data/text/shakespeare.txt";
    var positional: usize = 0;
    for ((try init.minimal.args.toSlice(init.arena.allocator()))[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            use_gpu = true;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            profile = true;
        } else {
            if (positional == 0) steps = try std.fmt.parseInt(usize, arg, 10) else path = arg;
            positional += 1;
        }
    }

    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    if (text.len < unroll + 2) return error.TextTooShort;

    const gpu: ?*grad.gpu.Gpu = if (use_gpu) try grad.gpu.Gpu.init(gpa, .{ .step_bytes = 1 << 30 }) else null;
    defer if (gpu) |dev| dev.deinit();
    if (gpu) |dev| {
        if (profile) dev.startProfile(init.io);
    }
    const par: grad.kernels.Parallel = .{ .io = init.io, .threads = std.Thread.getCpuCount() catch 1, .gpu = gpu };

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const host_model = try nn.initAll(Net, gpa, rng);
    defer nn.deinitAll(host_model, gpa);
    const model = if (gpu) |dev| try nn.toGpuAll(host_model, dev) else host_model;
    const params = nn.params(model);
    var adam = try grad.optim.Adam.init(gpa, par, 0.003, params);
    defer adam.deinit();

    const inputs = try par.alloc(f32, gpa, unroll * batch * vocab);
    const targets = try par.alloc(u8, gpa, unroll * batch);
    defer if (gpu == null) {
        gpa.free(inputs);
        gpa.free(targets);
    };
    const host_inputs = try gpa.alloc(f32, inputs.len);
    defer gpa.free(host_inputs);
    const host_targets = try gpa.alloc(u8, targets.len);
    defer gpa.free(host_targets);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    try out.print("character-level GRU, {d} hidden units, {d} characters of text, {s}\n", .{ hidden, text.len, if (gpu != null) "GPU" else "CPU" });
    try out.flush();

    // NOTE: the training time excludes start-up and the sample, so runs can be compared; the loss read-back each step synchronizes the device.
    const started = std.Io.Clock.awake.now(init.io);
    var running: f64 = 0;
    // NOTE: profiling synchronizes after every launch, which a recording cannot do, so a profiled run launches normally.
    const use_graph = gpu != null and !profile;
    var graph: ?grad.gpu.Gpu.Graph = null;
    var mean: Var(Tensor(f32, &.{})) = undefined;
    for (0..steps) |iteration| {
        @memset(host_inputs, 0);
        for (0..batch) |row| {
            const start = rng.uintLessThan(usize, text.len - unroll - 1);
            for (0..unroll) |t| {
                host_inputs[(t * batch + row) * vocab + (text[start + t] & 0x7f)] = 1;
                host_targets[t * batch + row] = text[start + t + 1] & 0x7f;
            }
        }
        try par.upload(inputs, host_inputs);
        try par.upload(targets, host_targets);

        // NOTE: the step is launch-bound (thousands of tiny kernels), so on the GPU it is recorded once into a graph and replayed.
        // Nothing in it changes between steps: the inputs and targets are refilled in place and the scratch addresses repeat.
        if (!use_graph or graph == null) {
            _ = arena.reset(.retain_capacity);
            if (gpu) |dev| {
                try dev.resetStep();
                if (use_graph) try dev.beginCapture();
            }
            var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true };
            try grad.optim.zeroGrad(par, params);
            var state = try tape.constant(try tape.zeroedTensor(State));
            var total: ?Var(Tensor(f32, &.{})) = null;
            for (0..unroll) |t| {
                const next, const scores = try step(model, &tape, .{ .data = inputs[t * batch * vocab ..][0 .. batch * vocab] }, state);
                const loss = try tape.crossEntropy(scores, targets[t * batch ..][0..batch]);
                total = if (total) |sum| try tape.add(sum, loss) else loss;
                state = next;
            }
            mean = try tape.mul(total.?, try tape.constant(try tape.filledTensor(Tensor(f32, &.{}), 1.0 / @as(f32, unroll))));
            try tape.backward(mean);
            if (use_graph) graph = try grad.gpu.Gpu.endCapture(gpu.?);
        }
        if (use_graph) try gpu.?.replay(graph.?);
        adam.lr = grad.optim.cosineLr(0.003, iteration, steps, 50);
        try adam.step(par, params);

        var value: [1]f32 = undefined;
        try par.download(&value, mean.value.data);
        running = if (iteration == 0) value[0] else 0.98 * running + 0.02 * value[0];
        if ((iteration + 1) % 100 == 0) {
            try out.print("step {d:>5}  loss {d:.3} nats/char\n", .{ iteration + 1, running });
            try out.flush();
        }
    }

    const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - started.nanoseconds)) / 1e9;
    try out.print("training took {d:.2} s\n", .{seconds});
    if (gpu) |dev| try dev.printProfile(out);
    try out.writeAll("\n--- sample ---\n");
    try generate(par, gpa, model, rng, out, "ROMEO:", 600);
    try out.flush();
}
