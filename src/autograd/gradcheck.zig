const std = @import("std");
const Allocator = std.mem.Allocator;
const Tape = @import("autograd.zig").Tape;
const Gpu = @import("../gpu/gpu.zig").Gpu;

const testing = std.testing;

fn smoothingOf(comptime Case: type) f32 {
    return if (@hasDecl(Case, "smoothing")) Case.smoothing else 0;
}

pub fn randomInputs(comptime Case: type, gpa: Allocator, seed: u64) !Case.Inputs {
    var prng = std.Random.DefaultPrng.init(seed);
    var p: Case.Inputs = undefined;
    inline for (Case.names) |name| @field(p, name) = try @FieldType(Case.Inputs, name).empty(gpa);
    while (true) {
        inline for (Case.names) |name| {
            for (@field(p, name).data) |*v| v.* = prng.random().float(f32) * 2 - 1;
        }
        if (!@hasDecl(Case, "acceptable") or try Case.acceptable(gpa, p)) return p;
    }
}

fn lossOf(comptime Case: type, p: Case.Inputs) !f32 {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t: Tape = .{ .gpa = arena.allocator(), .training = true, .smoothing = smoothingOf(Case) };
    return (try Case.build(&t, p, &Case.labels)).loss.value.data[0];
}

pub fn expectFiniteDifferences(comptime Case: type, seed: u64, eps: f32, tolerance: f32) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try randomInputs(Case, arena.allocator(), seed);

    var t: Tape = .{ .gpa = arena.allocator(), .training = true, .smoothing = smoothingOf(Case) };
    const run = try Case.build(&t, p, &Case.labels);
    try t.backward(run.loss);

    inline for (Case.names) |name| {
        for (@field(p, name).data, 0..) |*v, i| {
            const old = v.*;
            v.* = old + eps;
            const plus = try lossOf(Case, p);
            v.* = old - eps;
            const minus = try lossOf(Case, p);
            v.* = old;
            try testing.expectApproxEqAbs((plus - minus) / (2 * eps), @field(run, name).grad.data[i], tolerance);
        }
    }
}

pub fn expectGpuMatchesCpu(comptime Case: type, seed: u64, tolerance: f32) !void {
    return compareWithGpu(Case, seed, tolerance, false);
}

pub fn expectTensorCoresClose(comptime Case: type, seed: u64, tolerance: f32) !void {
    return compareWithGpu(Case, seed, tolerance, true);
}

fn compareWithGpu(comptime Case: type, seed: u64, tolerance: f32, tensor_cores: bool) !void {
    const gpu = Gpu.init(testing.allocator, 64 << 20) catch |err| switch (err) {
        error.DriverNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer gpu.deinit();
    gpu.tensor_cores = tensor_cores;
    gpu.tensor_core_min_work = 0;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const host = try randomInputs(Case, gpa, seed);

    var cpu_tape: Tape = .{ .gpa = gpa, .training = true, .smoothing = smoothingOf(Case) };
    const cpu = try Case.build(&cpu_tape, host, &Case.labels);
    try cpu_tape.backward(cpu.loss);

    var device: Case.Inputs = undefined;
    inline for (Case.names) |name| {
        const slice = try gpu.alloc(f32, @field(host, name).data.len);
        try gpu.upload(slice, @field(host, name).data);
        @field(device, name) = .{ .data = slice };
    }
    const labels = try gpu.alloc(u8, Case.labels.len);
    try gpu.upload(labels, &Case.labels);
    var gpu_tape: Tape = .{ .gpa = gpa, .par = .{ .gpu = gpu }, .training = true, .smoothing = smoothingOf(Case) };
    const run = try Case.build(&gpu_tape, device, labels);
    try gpu_tape.backward(run.loss);

    try testing.expectApproxEqAbs(cpu.loss.value.data[0], try gpu_tape.scalar(run.loss), tolerance);
    inline for (Case.names) |name| try expectSameOnGpu(gpu, gpa, @field(cpu, name).grad.data, @field(run, name).grad.data, tolerance);
    if (@hasDecl(Case, "value_names")) {
        inline for (Case.value_names) |name| try expectSameOnGpu(gpu, gpa, @field(cpu, name).data, @field(run, name).data, tolerance);
    }
}

fn expectSameOnGpu(gpu: *Gpu, gpa: Allocator, want: []const f32, device: []const f32, tolerance: f32) !void {
    const got = try gpa.alloc(f32, want.len);
    try gpu.download(got, device);
    for (want, got) |w, g| try testing.expectApproxEqAbs(w, g, tolerance);
}
