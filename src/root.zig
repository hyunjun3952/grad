pub const augment = @import("gpu/math/augment.zig");
pub const autograd = @import("autograd/autograd.zig");
pub const checkpoint = @import("nn/checkpoint.zig");
pub const cifar = @import("data/cifar.zig");
pub const cifar100 = @import("data/cifar100.zig");
pub const Dataset = @import("data/dataset.zig").Dataset;
pub const gpu = @import("gpu/gpu.zig");
pub const kernels = @import("cpu/kernels.zig");
pub const mnist = @import("data/mnist.zig");
pub const nn = @import("nn/nn.zig");
pub const onnx = @import("onnx/onnx.zig");
pub const ops = @import("tensor/ops.zig");
pub const optim = @import("nn/optim.zig");
pub const tensor = @import("tensor/tensor.zig");
pub const trainer = @import("nn/trainer.zig");
pub const Tensor = tensor.Tensor;
pub const matmul = tensor.matmul;

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("gpu/parity.zig");
    _ = @import("cpu/ops.zig");
    _ = @import("onnx/onnx.zig");
    _ = @import("onnx/runtime.zig");
    _ = @import("gpu/math/conv.zig");
    _ = @import("gpu/math/dropout.zig");
    _ = @import("gpu/math/activation.zig");
    _ = @import("gpu/math/softmax.zig");
    _ = @import("gpu/math/augment.zig");
    _ = @import("gpu/math/norm.zig");
    _ = @import("gpu/math/bf16.zig");
}
