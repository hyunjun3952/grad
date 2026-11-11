const std = @import("std");
const lib = @import("grad");

const testing = std.testing;
const Gpu = lib.gpu.Gpu;
const Replicas = lib.replicas.Replicas;
const Tensor = lib.tensor.Tensor;
const Var = lib.autograd.Var;

const Vector = Tensor(f32, &.{4});
const rows_per_replica = 4;

// Fits the mean of the rows with Adam on the loss mean((w - x)^2), the gradient being computed by hand on the host.
// Each replica sees its own share of the rows, so the averaged gradient equals the gradient of the whole batch.
const Job = struct {
    group: *Replicas,
    index: usize,
    use_gpu: bool,
    rows: []const f32,
    result: [4]f32 = undefined,
    failure: ?anyerror = null,

    fn main(job: *Job) void {
        job.run() catch |err| {
            job.failure = err;
            job.group.stop();
        };
    }

    fn run(job: *Job) !void {
        const gpa = testing.allocator;
        const gpu: ?*Gpu = if (job.use_gpu) try Gpu.init(gpa, .{ .step_bytes = 1 << 20 }) else null;
        defer if (gpu) |dev| dev.deinit();
        const par: lib.kernels.Parallel = .{ .gpu = gpu };

        const host_value = Vector{ .data = try gpa.alloc(f32, 4) };
        defer gpa.free(host_value.data);
        const host_grad = Vector{ .data = try gpa.alloc(f32, 4) };
        defer gpa.free(host_grad.data);
        @memset(host_value.data, 0);
        const param: Var(Vector) = if (gpu) |dev| blk: {
            const value = Vector{ .data = try dev.alloc(f32, 4) };
            const grad = Vector{ .data = try dev.alloc(f32, 4) };
            try dev.upload(value.data, host_value.data);
            break :blk .{ .value = value, .grad = grad, .needs_grad = true };
        } else .{ .value = host_value, .grad = host_grad, .needs_grad = true };

        var adam = try lib.optim.Adam.init(gpa, par, 0.1, .{param});
        defer adam.deinit();
        for (0..5) |_| {
            if (gpu) |dev| try dev.download(host_value.data, param.value.data);
            const mine = job.rows[job.index * job.rows.len / job.group.count ..][0 .. job.rows.len / job.group.count];
            @memset(host_grad.data, 0);
            for (0..mine.len / 4) |r| {
                for (host_grad.data, 0..) |*g, j| g.* += 2 * (host_value.data[j] - mine[r * 4 + j]) / @as(f32, @floatFromInt(mine.len / 4));
            }
            if (gpu) |dev| try dev.upload(param.grad.data, host_grad.data);
            try job.group.averageGrads(job.index, par, .{param});
            try adam.step(par, .{param});
        }
        if (gpu) |dev| try dev.download(&job.result, param.value.data) else @memcpy(&job.result, param.value.data);
    }
};

fn fit(use_gpu: bool, replicas: usize, rows: []const f32) ![4]f32 {
    var group = try Replicas.init(testing.allocator, replicas, 4);
    defer group.deinit();
    var jobs: [2]Job = undefined;
    var threads: [2]std.Thread = undefined;
    for (0..replicas) |i| jobs[i] = .{ .group = &group, .index = i, .use_gpu = use_gpu, .rows = rows };
    for (1..replicas) |i| threads[i] = try std.Thread.spawn(.{}, Job.main, .{&jobs[i]});
    jobs[0].main();
    for (1..replicas) |i| threads[i].join();
    for (jobs[0..replicas]) |job| if (job.failure) |err| return err;
    // Every replica must end with the same weights.
    for (jobs[1..replicas]) |job| try testing.expectEqualSlices(f32, &jobs[0].result, &job.result);
    return jobs[0].result;
}

fn rowsOf(rng: std.Random) [2 * rows_per_replica * 4]f32 {
    var rows: [2 * rows_per_replica * 4]f32 = undefined;
    for (&rows) |*v| v.* = rng.float(f32) * 4 - 1;
    return rows;
}

test "two replicas on the cpu train like one replica with the whole batch" {
    var prng = std.Random.DefaultPrng.init(21);
    const rows = rowsOf(prng.random());
    const single = try fit(false, 1, &rows);
    const double = try fit(false, 2, &rows);
    try testing.expect(single[0] != 0);
    for (single, double) |a, b| try testing.expectApproxEqAbs(a, b, 1e-5);
}

test "two replicas on one gpu train like one replica with the whole batch" {
    if (Gpu.deviceCount() == 0) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(22);
    const rows = rowsOf(prng.random());
    const single = try fit(true, 1, &rows);
    const double = try fit(true, 2, &rows);
    try testing.expect(single[0] != 0);
    for (single, double) |a, b| try testing.expectApproxEqAbs(a, b, 1e-5);
}

test "a failing replica releases the others from the barrier" {
    var group = try Replicas.init(testing.allocator, 2, 4);
    defer group.deinit();
    const waiter = try std.Thread.spawn(.{}, struct {
        fn run(g: *Replicas, result: *?anyerror) void {
            g.wait() catch |err| {
                result.* = err;
            };
        }
    }.run, .{ &group, &failure_slot });
    group.stop();
    waiter.join();
    try testing.expectEqual(@as(?anyerror, error.Stopped), failure_slot);
}

var failure_slot: ?anyerror = null;
