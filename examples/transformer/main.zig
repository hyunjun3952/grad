const std = @import("std");
const grad = @import("grad");

const nn = grad.nn;
const Tensor = grad.Tensor;
const Tape = grad.autograd.Tape;
const Var = grad.autograd.Var;

const vocab = 128;
const width = 64;
const heads = 4;
const context = 64;
const batch = 16;
const rows = batch * context;
const Net = struct {
    nn.Embedding(vocab, width),
    nn.Embedding(context, width),
    nn.TransformerBlock(width, heads, context),
    nn.TransformerBlock(width, heads, context),
    nn.LayerNorm(width),
    nn.Linear(width, vocab),
};
const Logits = Tensor(f32, &.{ rows, vocab });

const Run = struct {
    gpu: ?*grad.gpu.Gpu,
    par: grad.kernels.Parallel,

    fn buffer(self: Run, comptime T: type, gpa: std.mem.Allocator, n: usize) ![]T {
        return if (self.gpu) |dev| dev.alloc(T, n) else gpa.alloc(T, n);
    }

    fn put(self: Run, dst: anytype, src: anytype) !void {
        if (self.gpu) |dev| try dev.upload(dst, src) else @memcpy(dst, src);
    }

    fn get(self: Run, dst: anytype, src: anytype) !void {
        if (self.gpu) |dev| try dev.download(dst, src) else @memcpy(dst, src);
    }
};

// Token and position embeddings are added, passed through two causal blocks, and projected to one score per character.
fn forward(model: Net, tape: *Tape, tokens: []const u32, positions: []const u32) !Var(Logits) {
    const token_table, const position_table, const first, const second, const final_norm, const head = model;
    const embedded = try tape.add(try token_table.forward(tape, rows, tokens), try position_table.forward(tape, rows, positions));
    const mixed = try second.forward(tape, try first.forward(tape, embedded));
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
fn generate(run: Run, gpa: std.mem.Allocator, model: Net, rng: std.Random, out: *std.Io.Writer, seed_text: []const u8, length: usize) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const tokens = try run.buffer(u32, gpa, rows);
    const positions = try run.buffer(u32, gpa, rows);
    defer if (run.gpu == null) {
        gpa.free(tokens);
        gpa.free(positions);
    };
    const host_tokens = try gpa.alloc(u32, rows);
    defer gpa.free(host_tokens);
    const host_positions = try gpa.alloc(u32, rows);
    defer gpa.free(host_positions);
    for (host_positions, 0..) |*p, i| p.* = @intCast(i % context);
    try run.put(positions, host_positions);
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
        if (run.gpu) |dev| try dev.resetStep();
        for (0..batch) |row| {
            for (0..context) |t| host_tokens[row * context + t] = window[t];
        }
        try run.put(tokens, host_tokens);
        var tape: Tape = .{ .gpa = arena.allocator(), .par = run.par };
        const scores = try forward(model, &tape, tokens, positions);
        try run.get(logits, scores.value.data);
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &writer.interface;

    var iterations: usize = 1500;
    var use_gpu = false;
    var path: []const u8 = "data/text/shakespeare.txt";
    var positional: usize = 0;
    for ((try init.minimal.args.toSlice(init.arena.allocator()))[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            use_gpu = true;
        } else {
            if (positional == 0) iterations = try std.fmt.parseInt(usize, arg, 10) else path = arg;
            positional += 1;
        }
    }

    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    if (text.len < context + 2) return error.TextTooShort;

    const gpu: ?*grad.gpu.Gpu = if (use_gpu) try grad.gpu.Gpu.init(gpa, 1 << 30) else null;
    defer if (gpu) |dev| dev.deinit();
    const run: Run = .{ .gpu = gpu, .par = .{ .io = init.io, .threads = std.Thread.getCpuCount() catch 1, .gpu = gpu } };

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const host_model = try nn.initAll(Net, gpa, rng);
    defer nn.deinitAll(host_model, gpa);
    const model = if (gpu) |dev| try nn.toGpuAll(host_model, dev) else host_model;
    const params = nn.params(model);
    var adam = try grad.optim.Adam.init(gpa, run.par, 0.003, params);
    defer adam.deinit();

    const tokens = try run.buffer(u32, gpa, rows);
    const positions = try run.buffer(u32, gpa, rows);
    const targets = try run.buffer(u8, gpa, rows);
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
    try run.put(positions, host_positions);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    try out.print("character-level Transformer, {d} blocks, width {d}, {d} heads, context {d}, {d} characters of text, {s}\n", .{ 2, width, heads, context, text.len, if (gpu != null) "GPU" else "CPU" });
    try out.flush();

    var running: f64 = 0;
    for (0..iterations) |iteration| {
        for (0..batch) |row| {
            const start = rng.uintLessThan(usize, text.len - context - 1);
            for (0..context) |t| {
                host_tokens[row * context + t] = text[start + t] & 0x7f;
                host_targets[row * context + t] = text[start + t + 1] & 0x7f;
            }
        }
        try run.put(tokens, host_tokens);
        try run.put(targets, host_targets);

        _ = arena.reset(.retain_capacity);
        if (gpu) |dev| try dev.resetStep();
        var tape: Tape = .{ .gpa = arena.allocator(), .par = run.par, .training = true };
        try grad.optim.zeroGrad(run.par, params);
        const loss = try tape.crossEntropy(try forward(model, &tape, tokens, positions), targets);
        try tape.backward(loss);
        adam.lr = grad.optim.cosineLr(0.003, iteration, iterations, 50);
        try adam.step(run.par, params);

        var value: [1]f32 = undefined;
        try run.get(&value, loss.value.data);
        running = if (iteration == 0) value[0] else 0.98 * running + 0.02 * value[0];
        if ((iteration + 1) % 100 == 0) {
            try out.print("step {d:>5}  loss {d:.3} nats/char\n", .{ iteration + 1, running });
            try out.flush();
        }
    }

    try out.writeAll("\n--- sample ---\n");
    try generate(run, gpa, model, rng, out, "ROMEO:", 600);
    try out.flush();
}
