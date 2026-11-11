const std = @import("std");
const grad = @import("grad");
const config = @import("config");

const nn = grad.nn;
const Tensor = grad.Tensor;
const Tape = grad.autograd.Tape;
const Var = grad.autograd.Var;

const digits = 10;
const hidden = config.hidden;

const MlpLayers = struct { nn.Linear(784, hidden), nn.Linear(hidden, digits) };
const CnnLayers = struct { nn.Conv2d(1, 16, 5, .{}), nn.BatchNorm(16), nn.Conv2d(16, 32, 5, .{}), nn.BatchNorm(32), nn.Linear(512, digits) };

const Mnist = struct {
    pub const name = if (config.cnn) "MNIST CNN conv5x16-conv5x32-fc" else "MNIST MLP 784-" ++ std.fmt.comptimePrint("{d}", .{hidden}) ++ "-10";
    pub const Layers = if (config.cnn) CnnLayers else MlpLayers;
    pub const Loader = grad.mnist;
    pub const data_dir = "data/mnist";
    pub const side = 28;
    pub const channels = 1;
    pub const classes = digits;
    pub const batch = config.batch;
    pub const batches = [_]usize{ 16, 32, 64, 128, 256, 512 };
    pub const eval_batch = 1000;
    pub const learning_rate = 0.002;
    pub const weight_decay = 0.01;
    pub const scratch_bytes = 1 << 30;

    pub fn forward(model: Layers, tape: *Tape, x: anytype) grad.autograd.Error!Var(Tensor(f32, &.{ @TypeOf(x.value).shape[0], classes })) {
        if (!config.cnn) {
            const l1, const l2 = model;
            return l2.forward(tape, try tape.dropout(try tape.relu(try l1.forward(tape, x)), 0.8));
        }
        const c1, const n1, const c2, const n2, const fc = model;
        const n = @TypeOf(x.value).shape[0];
        const image = grad.autograd.reshape(x, &.{ n, 28, 28, 1 });
        const a = try tape.maxPool(try n1.forwardRelu(tape, try c1.forward(tape, image)), .{});
        const b = try tape.maxPool(try n2.forwardRelu(tape, try c2.forward(tape, a)), .{});
        return fc.forward(tape, try tape.dropout(grad.autograd.reshape(b, &.{ n, 512 }), 0.7));
    }
};

pub fn main(init: std.process.Init) !void {
    try grad.trainer.run(Mnist, init);
}
