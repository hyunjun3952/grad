// Every test file of the repository. Run them with zig build test; -Dtest-filter=text selects tests by name.
test {
    _ = @import("activation.zig");
    _ = @import("augment.zig");
    _ = @import("autograd.zig");
    _ = @import("bf16.zig");
    _ = @import("checkpoint.zig");
    _ = @import("cifar.zig");
    _ = @import("cifar100.zig");
    _ = @import("conv.zig");
    _ = @import("cpu_ops.zig");
    _ = @import("dropout.zig");
    _ = @import("gpu.zig");
    _ = @import("kernels.zig");
    _ = @import("layernorm.zig");
    _ = @import("mnist.zig");
    _ = @import("nn.zig");
    _ = @import("onnx.zig");
    _ = @import("ops.zig");
    _ = @import("optim.zig");
    _ = @import("parity.zig");
    _ = @import("replicas.zig");
    _ = @import("runtime.zig");
    _ = @import("softmax.zig");
    _ = @import("tensor.zig");
}
