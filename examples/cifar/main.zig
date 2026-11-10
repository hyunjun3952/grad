const grad = @import("grad");

const nn = grad.nn;
const Tensor = grad.Tensor;
const Tape = grad.autograd.Tape;
const Var = grad.autograd.Var;

const categories = 10;
const same = grad.autograd.ConvOpts{ .pad = 1 };

const Network = struct {
    nn.ConvBn(3, 64, 3, same),
    nn.ConvBn(64, 64, 3, same),
    nn.ConvBn(64, 64, 3, same),
    nn.ConvBn(64, 64, 3, same),
    nn.ConvBn(64, 64, 3, same),
    nn.ConvBn(64, 128, 3, same),
    nn.ConvBn(128, 128, 3, same),
    nn.ConvBn(64, 128, 1, .{}),
    nn.ConvBn(128, 128, 3, same),
    nn.ConvBn(128, 128, 3, same),
    nn.ConvBn(128, 256, 3, same),
    nn.ConvBn(256, 256, 3, same),
    nn.ConvBn(128, 256, 1, .{}),
    nn.ConvBn(256, 256, 3, same),
    nn.ConvBn(256, 256, 3, same),
    nn.Linear(256, categories),
};

fn residual(tape: *Tape, first: anytype, second: anytype, x: anytype) grad.autograd.Error!@TypeOf(x) {
    return tape.addRelu(try second.forward(tape, try first.forwardRelu(tape, x)), x);
}

const Cifar = struct {
    pub const name = "CIFAR-10 ResNet-14 64-128-256";
    pub const Layers = Network;
    pub const Loader = grad.cifar;
    pub const data_dir = "data/cifar";
    pub const side = 32;
    pub const channels = 3;
    pub const classes = categories;
    pub const batch = 64;
    pub const eval_batch = 100;
    pub const learning_rate = 0.002;
    pub const weight_decay = 0.01;
    pub const scratch_bytes = 3 << 30;

    pub fn forward(model: Layers, tape: *Tape, x: anytype) grad.autograd.Error!Var(Tensor(f32, &.{ @TypeOf(x.value).shape[0], classes })) {
        const stem, const a1, const b1, const a2, const b2, const a3, const b3, const skip3, const a4, const b4, const a5, const b5, const skip5, const a6, const b6, const fc = model;
        const n = @TypeOf(x.value).shape[0];
        const image = grad.autograd.reshape(x, &.{ n, 32, 32, 3 });

        const stage1 = try tape.maxPool(try residual(tape, a2, b2, try residual(tape, a1, b1, try stem.forwardRelu(tape, image))));
        const down2 = try tape.addRelu(try b3.forward(tape, try a3.forwardRelu(tape, stage1)), try skip3.forward(tape, stage1));
        const stage2 = try tape.maxPool(try residual(tape, a4, b4, down2));
        const down3 = try tape.addRelu(try b5.forward(tape, try a5.forwardRelu(tape, stage2)), try skip5.forward(tape, stage2));
        const stage3 = try tape.maxPool(try residual(tape, a6, b6, down3));
        const pooled = try tape.avgPool(try tape.avgPool(stage3));
        return fc.forward(tape, try tape.dropout(grad.autograd.reshape(pooled, &.{ n, 256 }), 0.2));
    }
};

pub fn main(init: @import("std").process.Init) !void {
    try grad.trainer.run(Cifar, init);
}
