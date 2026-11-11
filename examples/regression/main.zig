const std = @import("std");
const grad = @import("grad");

const nn = grad.nn;
const Tensor = grad.Tensor;
const Tape = grad.autograd.Tape;

const batch = 64;
const Net = struct { nn.Linear(1, 32), nn.Linear(32, 32), nn.Linear(32, 1) };

fn curve(x: f32) f32 {
    return @sin(3 * x) + 0.3 * x;
}

fn predict(net: Net, tape: *Tape, x: Tensor(f32, &.{ batch, 1 })) grad.autograd.Error!grad.autograd.Var(Tensor(f32, &.{ batch, 1 })) {
    const first, const second, const last = net;
    const a = try tape.relu(try first.forward(tape, try tape.constant(x)));
    return last.forward(tape, try tape.relu(try second.forward(tape, a)));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var buffer: [1024]u8 = undefined;
    var writer: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &writer.interface;

    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const net = try nn.initAll(Net, gpa, rng);
    defer nn.deinitAll(net, gpa);
    const params = nn.params(net);
    var adam = try grad.optim.Adam.init(gpa, .{}, 0.01, params);
    defer adam.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const steps = 3000;
    for (0..steps) |step| {
        _ = arena.reset(.retain_capacity);
        var tape: Tape = .{ .gpa = arena.allocator() };
        const xs = try Tensor(f32, &.{ batch, 1 }).empty(arena.allocator());
        const ys = try Tensor(f32, &.{ batch, 1 }).empty(arena.allocator());
        for (xs.data, ys.data) |*x, *y| {
            x.* = rng.float(f32) * 4 - 2;
            y.* = curve(x.*);
        }
        try grad.optim.zeroGrad(.{}, params);
        const loss = try tape.mse(try predict(net, &tape, xs), try tape.constant(ys));
        try tape.backward(loss);
        try adam.step(.{}, params);
        if ((step + 1) % 500 == 0) try out.print("step {d:>4}  mean squared error {d:.5}\n", .{ step + 1, loss.value.data[0] });
    }

    _ = arena.reset(.retain_capacity);
    var tape: Tape = .{ .gpa = arena.allocator() };
    const grid = try Tensor(f32, &.{ batch, 1 }).empty(arena.allocator());
    for (grid.data, 0..) |*x, i| x.* = -2 + 4 * @as(f32, @floatFromInt(i)) / (batch - 1);
    const fitted = (try predict(net, &tape, grid)).value;
    try out.writeAll("\n      x    sin(3x)+0.3x    network\n");
    for (0..9) |k| {
        const i = k * (batch - 1) / 8;
        try out.print("{d:>7.3} {d:>12.4} {d:>12.4}\n", .{ grid.data[i], curve(grid.data[i]), fitted.data[i] });
    }
    try out.flush();
}
