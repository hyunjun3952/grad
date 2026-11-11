const std = @import("std");
const grad = @import("grad");
const config = @import("config");

const nn = grad.nn;
const Tensor = grad.Tensor;
const Tape = grad.autograd.Tape;
const Var = grad.autograd.Var;

const categories = if (config.cifar100) 100 else 10;
const w1 = 64;
const w2 = if (config.wide) 192 else 128;
const w3 = if (config.wide) 384 else 256;
const same = grad.autograd.ConvOpts{ .pad = 1 };

const Network = struct {
    nn.ConvBn(3, w1, 3, same),
    nn.ConvBn(w1, w1, 3, same),
    nn.ConvBn(w1, w1, 3, same),
    nn.ConvBn(w1, w1, 3, same),
    nn.ConvBn(w1, w1, 3, same),
    nn.ConvBn(w1, w2, 3, same),
    nn.ConvBn(w2, w2, 3, same),
    nn.ConvBn(w1, w2, 1, .{}),
    nn.ConvBn(w2, w2, 3, same),
    nn.ConvBn(w2, w2, 3, same),
    nn.ConvBn(w2, w3, 3, same),
    nn.ConvBn(w3, w3, 3, same),
    nn.ConvBn(w2, w3, 1, .{}),
    nn.ConvBn(w3, w3, 3, same),
    nn.ConvBn(w3, w3, 3, same),
    nn.Linear(w3, categories),
};

fn residual(tape: *Tape, first: anytype, second: anytype, x: anytype) grad.autograd.Error!@TypeOf(x) {
    return tape.addRelu(try second.forward(tape, try first.forwardRelu(tape, x)), x);
}

const Cifar = struct {
    pub const name = std.fmt.comptimePrint("CIFAR-{d} ResNet-14 {d}-{d}-{d}", .{ categories, w1, w2, w3 });
    pub const Layers = Network;
    pub const Loader = if (config.cifar100) grad.cifar100 else grad.cifar;
    pub const data_dir = if (config.cifar100) "data/cifar100" else "data/cifar";
    pub const side = 32;
    pub const channels = 3;
    pub const classes = categories;
    pub const batch = 64;
    pub const batches = [_]usize{ 32, 64, 128 };
    pub const eval_batch = 100;
    pub const learning_rate = 0.002;
    pub const weight_decay = 0.01;
    pub const scratch_bytes = 3 << 30;

    pub fn forward(model: Layers, tape: *Tape, x: anytype) grad.autograd.Error!Var(Tensor(f32, &.{ @TypeOf(x.value).shape[0], classes })) {
        const stem, const a1, const b1, const a2, const b2, const a3, const b3, const skip3, const a4, const b4, const a5, const b5, const skip5, const a6, const b6, const fc = model;
        const n = @TypeOf(x.value).shape[0];
        const image = grad.autograd.reshape(x, &.{ n, 32, 32, 3 });

        const stage1 = try tape.maxPool(try residual(tape, a2, b2, try residual(tape, a1, b1, try stem.forwardRelu(tape, image))), .{});
        const down2 = try tape.addRelu(try b3.forward(tape, try a3.forwardRelu(tape, stage1)), try skip3.forward(tape, stage1));
        const stage2 = try tape.maxPool(try residual(tape, a4, b4, down2), .{});
        const down3 = try tape.addRelu(try b5.forward(tape, try a5.forwardRelu(tape, stage2)), try skip5.forward(tape, stage2));
        const stage3 = try tape.maxPool(try residual(tape, a6, b6, down3), .{});
        const pooled = try tape.avgPool(try tape.avgPool(stage3, .{}), .{});
        return fc.forward(tape, try tape.dropout(grad.autograd.reshape(pooled, &.{ n, w3 }), 0.2));
    }
};

pub fn main(init: @import("std").process.Init) !void {
    try grad.trainer.run(Cifar, init);
}
