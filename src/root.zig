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
pub const replicas = @import("nn/replicas.zig");
pub const tensor = @import("tensor/tensor.zig");
pub const trainer = @import("nn/trainer.zig");
pub const Tensor = tensor.Tensor;
pub const matmul = tensor.matmul;

// The modules below are internals that the programs do not need but the tests in tests/ reach into.
pub const cpu_ops = @import("cpu/ops.zig");
pub const dataset = @import("data/dataset.zig");
pub const gpu_args = @import("gpu/args.zig");
pub const gpu_cuda = @import("gpu/cuda.zig");
pub const gpu_cudnn = @import("gpu/cudnn.zig");
pub const onnx_gpu = @import("onnx/gpu.zig");
pub const onnx_runtime = @import("onnx/runtime.zig");
pub const trace = @import("autograd/trace.zig");

/// Per-element math shared by the CPU and the device code.
pub const math = struct {
    pub const activation = @import("gpu/math/activation.zig");
    pub const augment = @import("gpu/math/augment.zig");
    pub const bf16 = @import("gpu/math/bf16.zig");
    pub const conv = @import("gpu/math/conv.zig");
    pub const dropout = @import("gpu/math/dropout.zig");
    pub const layernorm = @import("gpu/math/layernorm.zig");
    pub const norm = @import("gpu/math/norm.zig");
    pub const softmax = @import("gpu/math/softmax.zig");
};
