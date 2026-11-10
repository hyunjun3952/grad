pub const autograd = @import("autograd.zig");
pub const gpu = @import("gpu/gpu.zig");
pub const kernels = @import("kernels.zig");
pub const mnist = @import("mnist.zig");
pub const nn = @import("nn.zig");
pub const ops = @import("ops.zig");
pub const optim = @import("optim.zig");
pub const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const matmul = tensor.matmul;

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("gpu/parity.zig");
}
