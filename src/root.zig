pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");
pub const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const matmul = tensor.matmul;

test {
    @import("std").testing.refAllDecls(@This());
}
