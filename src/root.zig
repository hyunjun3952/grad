pub const augment = @import("gpu/math/augment.zig");
pub const autograd = @import("autograd/autograd.zig");
pub const checkpoint = @import("nn/checkpoint.zig");
pub const gpu = @import("gpu/gpu.zig");
pub const kernels = @import("cpu/kernels.zig");
pub const mnist = @import("data/mnist.zig");
pub const nn = @import("nn/nn.zig");
pub const ops = @import("tensor/ops.zig");
pub const optim = @import("nn/optim.zig");
pub const tensor = @import("tensor/tensor.zig");
pub const Tensor = tensor.Tensor;
pub const matmul = tensor.matmul;

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("gpu/parity.zig");
    _ = @import("gpu/math/conv.zig");
    _ = @import("gpu/math/dropout.zig");
    _ = @import("gpu/math/augment.zig");
    _ = @import("gpu/math/norm.zig");
}
