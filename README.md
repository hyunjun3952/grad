# grad

An experimental deep-learning engine written in Zig 0.17. It trains MLPs and CNNs on the CPU or on an NVIDIA GPU with the same code, and it needs no CUDA toolkit: the GPU kernels are Zig functions compiled to PTX by the Zig compiler and launched through the driver API (`nvcuda.dll`).

The goal is an engine that sits between "C with pointers" and "PyTorch": plain structs, explicit allocators and explicit parallelism, but shapes are checked by the compiler and autograd is a normal library.

| Task | Model | Accuracy | Time per epoch |
| --- | --- | --- | --- |
| MNIST | MLP 784-128-10 | 98.1% (10 epochs) | 0.19 s GPU, 0.26 s CPU (6 threads) |
| MNIST | CNN conv5x16-conv5x32 + BatchNorm | 99.5% (30 epochs, light augmentation) | 0.9 s GPU, 19.5 s CPU |
| CIFAR-10 | ResNet-14, widths 64-128-256 | 93.3-93.6% (50 epochs, flip + shift + label smoothing; varies by run) | 17.6 s GPU (bf16 tensor cores), 54.9 s GPU (f32) |

Measured on an i5-8500 (6 cores) and an RTX 3060 12 GB. About 5,200 lines including 66 tests, with no comments except for explained magic numbers.

## Highlights

- **Compile-time shapes.** `Tensor(f32, &.{ 64, 784 })` carries its shape in the type, so a wrong `matmul`, `add` or `reshape` is a compile error. Data lives on the heap.
- **Small autograd.** A type-erased `Tape` records closures; `Var{ value, grad, needs_grad }` is a plain struct. Ops: `add`, `mul`, `matmul`, `relu`, `conv2d`, `maxPool`, `avgPool`, `batchNorm`, `dropout`, `crossEntropy`, `reshape`.
- **One code path for CPU and GPU.** Pass `kernels.Parallel{ .io, .threads, .gpu }` and everything (tape, optimizers, matmul) dispatches to the CPU worker pool or to the device. Per-element math such as `im2col`, dropout, batch-norm and augmentation is written once and shared by both backends.
- **GPU without a toolkit.** `src/gpu/device.zig` is compiled for `nvptx64-cuda` / `sm_86`, embedded as PTX and loaded with `cuModuleLoadData`. It contains tiled FP32 GEMM (about 5 TFLOPS), a bf16 tensor-core GEMM using `cp.async`, `ldmatrix` and `mma.sync` with a three-stage pipeline (about 21 TFLOPS at 4096, 23.7 without operand packing), reductions, fused multi-tensor Adam, and so on. Convolutions with 64 or more input channels run as implicit GEMMs: the tensor-core kernel gathers its operand tiles straight from the NHWC activations with zero-filled `cp.async` (forward, input gradient via rotated weights, and weight gradient), so no im2col buffer is ever written. Batch norm + ReLU and add + ReLU are fused ops. On the CIFAR ResNet-14 the bf16 path makes an epoch 3.1x faster than FP32 with the same accuracy.
- **Tested against references.** Every GPU op is compared with the CPU, gradients with finite differences, convolutions with a direct implementation, and bf16 GEMM with a CPU reference built from the same rounded inputs. Resuming from a checkpoint reproduces uninterrupted training bit for bit.

## Requirements

- Zig 0.17.1.
- For the GPU path: Windows and an NVIDIA GPU (the driver loader uses `kernel32` and `nvcuda.dll`; the kernels target Ampere, `sm_86`). GPU tests are skipped when no driver is found.
- No CUDA toolkit, no Python.

## Quick start

```powershell
zig build test
powershell -File scripts/fetch-data.ps1            # MNIST and CIFAR-10 into data/

zig build mnist -- 5                                # MLP on the CPU
zig build mnist -- 5 --gpu                          # MLP on the GPU
zig build mnist -Dcnn=true -- 30 --gpu --shift=1 --rotate=8 --zoom=0.05
zig build cifar -- 50 --gpu --flip --shift=4 --tensor-cores --smoothing=0.1
```

Build options such as `-Dcnn=true -Dhidden=2048 -Dbatch=256` go before the `--`; the arguments after it belong to the program.

### Trainer flags

| Flag | Meaning |
| --- | --- |
| `[epochs]` | Total number of epochs; the cosine learning-rate schedule spans all of them (default 5). |
| `--gpu` | Train on the GPU. |
| `--tensor-cores` | Use bf16 tensor-core GEMMs for large matrix products (f32 accumulate, f32 master weights). |
| `--shift=N`, `--rotate=DEG`, `--zoom=F`, `--flip` | Random affine augmentation applied on the fly, on the CPU or on the GPU. |
| `--cudnn[=DIR]` | Run convolutions through cuDNN (f32 with TF32 tensor cores) instead of the built-in kernels. The DLLs are loaded at run time from `data/cudnn` (or DIR); if they cannot be found the built-in kernels are used. Fetch them with `scripts/fetch-cudnn.ps1` (about 1.3 GB, never committed). |
| `--smoothing=F` | Label smoothing for the cross-entropy loss (CPU and GPU). |
| `--save=PATH`, `--load=PATH` | Write or read a checkpoint (weights, batch-norm statistics, Adam state, progress). |
| `--until=N` | Stop after epoch `N` but keep the schedule of the full run, so that a later `--load` continues exactly. |
| `--progress` | Force the live progress bar (it is shown automatically on a terminal; redirected output only gets one line per epoch). |
| `--profile` | Print per-kernel GPU timings (every op is synchronised, so absolute numbers are inflated). |

The optimizer is AdamW with a cosine schedule. Checkpoints are portable between CPU and GPU.

## Optional cuDNN backend

The engine does not need cuDNN, but it can use it: `src/gpu/cudnn.zig` loads `cudnn64_9.dll` dynamically (the same way `nvcuda.dll` is loaded) and drives the legacy descriptor API, with the best algorithm per shape chosen by `cudnnFind*` and cached. `zig build bench -- cudnn` times it on the CIFAR shapes: cuDNN reaches 8 to 10 TFLOPS in f32 (TF32) and 20 to 24 TFLOPS in bf16, versus about 5 TFLOPS for the built-in f32 GEMM and 17 to 18 TFLOPS for the built-in bf16 implicit GEMM. On the CIFAR ResNet-14 an f32 epoch takes 52.8 s with the built-in kernels and 24.5 s with `--cudnn`.

## Benchmarks

```powershell
zig build bench                 # CPU matmul kernels, allocation overhead, thread scaling
zig build bench -- step         # per-op profile of an MLP training step
zig build bench -- gpu          # FP32 GEMM on the device
zig build bench -- tc           # FP32 vs bf16 tensor-core GEMM, packing included, implicit-GEMM convolutions
zig build bench -- cudnn        # cuDNN convolutions on the CIFAR shapes (needs the DLLs)
```

On this machine: the CPU GEMM reaches about 108 GFLOPS per thread (packed B, 6x2 register tile); the GPU FP32 GEMM about 5 TFLOPS; the bf16 tensor-core kernel 12.8 to 21.6 TFLOPS on square and MLP shapes (end to end, packing included), with a relative RMS error of about 2e-3 against FP32.

## Layout

```
src/
  root.zig               public API (grad.nn, grad.optim, grad.trainer, ...)
  tensor/                Tensor(T, shape), broadcasting ops
  cpu/kernels.zig        worker pool, tiled matmul, flush-denormals
  autograd/              Tape and Var, finite-difference / GPU-vs-CPU test helpers
  nn/                    Linear, Conv2d, BatchNorm, ConvBn, Adam, checkpoints, trainer
  data/                  MNIST and CIFAR-10 loaders
  gpu/                   driver loader, runtime, device kernels (PTX), parity tests
    math/                per-element math shared by the CPU and the device
examples/
  mnist/  cifar/         the two training programs
scripts/fetch-data.ps1   dataset download with checksum verification
```

## A complete training program

This is `examples/mnist` reduced to one small convolutional network (it trains to 97.8% in a single epoch):

```zig
const Network = struct { nn.Conv2d(1, 16, 5, .{}), nn.BatchNorm(16), nn.Linear(2304, 10) };

fn apply(model: Network, tape: *Tape, x: anytype) autograd.Error!Var(Tensor(f32, &.{ @TypeOf(x.value).shape[0], 10 })) {
    const conv, const norm, const fc = model;
    const n = @TypeOf(x.value).shape[0];
    const image = autograd.reshape(x, &.{ n, 28, 28, 1 });
    const a = try tape.maxPool(try tape.relu(try norm.forward(tape, try conv.forward(tape, image))));
    return fc.forward(tape, autograd.reshape(a, &.{ n, 2304 }));
}

const Spec = struct {
    pub const name = "MNIST CNN";
    pub const Layers = Network;
    pub const forward = apply;
    pub const Loader = grad.mnist;
    pub const data_dir = "data/mnist";
    pub const side = 28;
    pub const channels = 1;
    pub const classes = 10;
    pub const batch = 64;
    pub const eval_batch = 1000;
    pub const learning_rate = 0.002;
    pub const weight_decay = 0.01;
    pub const scratch_bytes = 1 << 30;
};

pub fn main(init: std.process.Init) !void {
    try grad.trainer.run(Spec, init);
}
```

A layer is any struct with `init`, `deinit`, `toGpu`, `forward`, `params` and `state`. `nn.initAll`, `nn.params` and `nn.state` walk a tuple of layers, and `grad.trainer.run` takes care of batching, shuffling, augmentation, evaluation, checkpoints and the GPU.
## Limitations

- Windows-only GPU loader, f32 tensors (bf16 only inside GEMM), activations in NHWC, tensors of rank 4 or less on the GPU.
- 2x2 pooling only; convolutions have stride and padding but no dilation or groups.
- Batch sizes are compile-time constants.
- Training is single-GPU and deterministic on the CPU; GPU runs differ in the last bits because of atomic accumulation.
