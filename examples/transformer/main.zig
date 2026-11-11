const std = @import("std");
const grad = @import("grad");
const config = @import("config");

const nn = grad.nn;
const Tensor = grad.Tensor;
const Tape = grad.autograd.Tape;
const Var = grad.autograd.Var;

const vocab = 128;
const width = 128;
const heads = 4;
const context = 64;
const batch = 32;
const rows = batch * context;
const Net = struct {
    nn.Embedding(vocab, width),
    nn.Embedding(context, width),
    nn.TransformerBlock(width, heads, context),
    nn.TransformerBlock(width, heads, context),
    nn.TransformerBlock(width, heads, context),
    nn.LayerNorm(width),
    nn.Linear(width, vocab),
};
// With -Dbf16=true the activations between the layers are bf16 (GPU only); parameters, sums and the loss stay f32.
const Act = if (config.bf16) u16 else f32;
fn Logits(comptime n: usize) type {
    return Tensor(Act, &.{ n, vocab });
}

// Token and position embeddings are added, passed through three causal blocks, and projected to one score per character.
// n is the number of rows, a whole number of sequences of context characters.
fn forward(model: Net, tape: *Tape, comptime n: usize, tokens: []const u32, positions: []const u32) !Var(Logits(n)) {
    const token_table, const position_table, const first, const second, const third, const final_norm, const head = model;
    const embedded = try tape.add(try token_table.forward(tape, n, tokens, Act), try position_table.forward(tape, n, positions, Act));
    const mixed = try third.forward(tape, try second.forward(tape, try first.forward(tape, embedded)));
    return head.forward(tape, try final_norm.forward(tape, mixed));
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

// Writes length characters after seed_text. Every batch row holds the same window, because the model works on whole batches.
// NOTE: attention is causal, so the scores at the last filled position do not depend on the padding after it.
fn generate(par: grad.kernels.Parallel, gpa: std.mem.Allocator, model: Net, rng: std.Random, out: *std.Io.Writer, seed_text: []const u8, length: usize) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const tokens = try par.alloc(u32, gpa, rows);
    const positions = try par.alloc(u32, gpa, rows);
    defer par.free(gpa, tokens);
    defer par.free(gpa, positions);
    const host_tokens = try gpa.alloc(u32, rows);
    defer gpa.free(host_tokens);
    const host_positions = try gpa.alloc(u32, rows);
    defer gpa.free(host_positions);
    for (host_positions, 0..) |*p, i| p.* = @intCast(i % context);
    try par.upload(positions, host_positions);
    const logits = try gpa.alloc(f32, rows * vocab);
    defer gpa.free(logits);

    var window: [context]u8 = @splat(0);
    var filled: usize = 0;
    for (seed_text) |c| {
        try out.writeByte(c);
        if (filled == context) {
            std.mem.copyForwards(u8, window[0 .. context - 1], window[1..]);
            filled -= 1;
        }
        window[filled] = c & 0x7f;
        filled += 1;
    }
    for (0..length) |_| {
        _ = arena.reset(.retain_capacity);
        if (par.gpu) |dev| try dev.resetStep();
        for (0..batch) |row| {
            for (0..context) |t| host_tokens[row * context + t] = window[t];
        }
        try par.upload(tokens, host_tokens);
        var tape: Tape = .{ .gpa = arena.allocator(), .par = par };
        const scores = try tape.cast(try forward(model, &tape, rows, tokens, positions), f32);
        try par.download(logits, scores.value.data);
        const next = sample(rng, logits[(filled - 1) * vocab ..][0..vocab], 0.8);
        try out.writeByte(next);
        if (filled == context) {
            std.mem.copyForwards(u8, window[0 .. context - 1], window[1..]);
            filled -= 1;
        }
        window[filled] = next;
        filled += 1;
    }
    try out.writeByte('\n');
}

// Writes the model as ONNX for one sequence of context characters. The graph takes the character codes and the positions as
// int32 inputs "input0" and "input1" and gives the scores of every position; it is traced with the first characters of text.
fn exportOnnx(par: grad.kernels.Parallel, gpa: std.mem.Allocator, io: std.Io, model: Net, text: []const u8, path: []const u8, out: *std.Io.Writer) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const tokens = try par.alloc(u32, gpa, context);
    const positions = try par.alloc(u32, gpa, context);
    defer par.free(gpa, tokens);
    defer par.free(gpa, positions);
    var host_tokens: [context]u32 = undefined;
    var host_positions: [context]u32 = undefined;
    for (&host_tokens, &host_positions, 0..) |*t, *p, i| {
        t.* = text[i] & 0x7f;
        p.* = @intCast(i);
    }
    try par.upload(tokens, &host_tokens);
    try par.upload(positions, &host_positions);

    if (par.gpu) |dev| try dev.resetStep();
    var trace: grad.onnx.Trace = .{ .gpa = gpa };
    defer trace.deinit();
    var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .trace = &trace };
    const logits = try tape.cast(try forward(model, &tape, context, tokens, positions), f32);

    const slices = nn.state(model);
    const weights = try grad.onnx.WeightSet(slices.len).collect(gpa, par, slices);
    defer weights.deinit(gpa);
    try grad.onnx.save(gpa, io, path, .{ .trace = &trace, .weights = &weights.items, .inputs = &.{ grad.onnx.View.ofIds(tokens), grad.onnx.View.ofIds(positions) }, .outputs = &.{grad.onnx.View.of(logits)} });

    // The scores of the last position, for scripts/check-onnx.py to compare with.
    const scores = try gpa.alloc(f32, context * vocab);
    defer gpa.free(scores);
    try par.download(scores, logits.value.data);
    try out.print("exported {s}\nlogits of the last position:", .{path});
    for (scores[(context - 1) * vocab ..]) |s| try out.print(" {d:.5}", .{s});
    try out.writeByte('\n');
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &writer.interface;

    var iterations: usize = 1500;
    var use_gpu = false;
    var profile = false;
    var path: []const u8 = "data/text/shakespeare.txt";
    var onnx_path: ?[]const u8 = null;
    var positional: usize = 0;
    for ((try init.minimal.args.toSlice(init.arena.allocator()))[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            use_gpu = true;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            profile = true;
        } else if (std.mem.startsWith(u8, arg, "--onnx=")) {
            onnx_path = arg["--onnx=".len..];
        } else {
            if (positional == 0) iterations = try std.fmt.parseInt(usize, arg, 10) else path = arg;
            positional += 1;
        }
    }

    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    if (text.len < context + 2) return error.TextTooShort;

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

    const tokens = try par.alloc(u32, gpa, rows);
    const positions = try par.alloc(u32, gpa, rows);
    const targets = try par.alloc(u8, gpa, rows);
    defer if (gpu == null) {
        gpa.free(tokens);
        gpa.free(positions);
        gpa.free(targets);
    };
    const host_tokens = try gpa.alloc(u32, rows);
    defer gpa.free(host_tokens);
    const host_positions = try gpa.alloc(u32, rows);
    defer gpa.free(host_positions);
    const host_targets = try gpa.alloc(u8, rows);
    defer gpa.free(host_targets);
    for (host_positions, 0..) |*p, i| p.* = @intCast(i % context);
    try par.upload(positions, host_positions);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    try out.print("character-level Transformer, {d} blocks, width {d}, {d} heads, context {d}, {d} characters of text, {s}\n", .{ 3, width, heads, context, text.len, if (gpu != null) "GPU" else "CPU" });
    try out.flush();

    // NOTE: the training time excludes start-up and the sample, so runs can be compared; the loss read-back each step synchronizes the device.
    const started = std.Io.Clock.awake.now(init.io);
    var running: f64 = 0;
    for (0..iterations) |iteration| {
        for (0..batch) |row| {
            const start = rng.uintLessThan(usize, text.len - context - 1);
            for (0..context) |t| {
                host_tokens[row * context + t] = text[start + t] & 0x7f;
                host_targets[row * context + t] = text[start + t + 1] & 0x7f;
            }
        }
        try par.upload(tokens, host_tokens);
        try par.upload(targets, host_targets);

        _ = arena.reset(.retain_capacity);
        if (gpu) |dev| try dev.resetStep();
        var tape: Tape = .{ .gpa = arena.allocator(), .par = par, .training = true };
        try grad.optim.zeroGrad(par, params);
        const loss = try tape.crossEntropy(try forward(model, &tape, rows, tokens, positions), targets);
        try tape.backward(loss);
        adam.lr = grad.optim.cosineLr(0.003, iteration, iterations, 50);
        try adam.step(par, params);

        var value: [1]f32 = undefined;
        try par.download(&value, loss.value.data);
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
    if (onnx_path) |p| try exportOnnx(par, gpa, init.io, model, text, p, out);
    try out.flush();
}
