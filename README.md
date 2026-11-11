# grad

An experimental deep-learning engine written in Zig 0.17. It trains MLPs and CNNs on the CPU or on an NVIDIA GPU with the same code, and it needs no CUDA toolkit: the GPU kernels are Zig functions compiled to PTX by the Zig compiler and launched through the driver API (`nvcuda.dll`).

The goal is an engine that sits between "C with pointers" and "PyTorch": plain structs, explicit allocators and explicit parallelism, but shapes are checked by the compiler and autograd is a normal library.

| Task | Model | Accuracy | Time per epoch |
| --- | --- | --- | --- |
| MNIST | MLP 784-128-10 | 98.1% (10 epochs) | 0.19 s GPU, 0.26 s CPU (6 threads) |
| MNIST | CNN conv5x16-conv5x32 + BatchNorm | 99.5% (30 epochs, light augmentation) | 0.9 s GPU, 19.5 s CPU |
| CIFAR-10 | ResNet-14, widths 64-128-256 | 94.1% (50 epochs) to 94.9% (100 epochs); flip + shift + CutMix + label smoothing + EMA, +0.4 with `--tta` | 17.2 s GPU (bf16 tensor cores), 24.6 s (TF32), 52.8 s (strict f32) |
| CIFAR-10 | ResNet-14, widths 64-192-384 (`-Dwide=true`) | 94.8% / 95.3% with `--tta` (50 epochs), 95.4% / **96.2%** with `--tta` (100 epochs); same recipe | 24.7 s GPU (bf16 tensor cores) |

Measured on an i5-8500 (6 cores) and an RTX 3060 12 GB. About 7,600 lines including 81 tests, with no comments except for explained magic numbers.

## Highlights

- **Compile-time shapes.** `Tensor(f32, &.{ 64, 784 })` carries its shape in the type, so a wrong `matmul`, `add` or `reshape` is a compile error. Data lives on the heap.
- **Small autograd.** A type-erased `Tape` records closures; `Var{ value, grad, needs_grad }` is a plain struct. Ops: `add`, `mul`, `matmul`, `relu`, `conv2d`, `maxPool`, `avgPool`, `batchNorm`, `dropout`, `crossEntropy`, `mse`, `sigmoid`/`tanh`/`gelu` (`Tape.activate`), `reshape`. Layers: `Linear`, `Conv2d` (stride, padding, dilation, groups), `BatchNorm`, `ConvBn`, `Gru`. `maxPool`/`avgPool` take a window, stride and padding (`PoolOpts`).
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
powershell -File scripts/fetch-data.ps1            # MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 and Tiny Shakespeare into data/ (or -Dataset NAME)

zig build regression                                # fit a curve with a small MLP (library API without the trainer)
zig build charlm -- 1500 --gpu                      # character-level GRU on Tiny Shakespeare, then sample text
zig build mnist -- 5                                # MLP on the CPU
zig build mnist -- 5 --gpu                          # MLP on the GPU
zig build mnist -Dcnn=true -- 30 --gpu --shift=1 --rotate=8 --zoom=0.05
zig build cifar -- 50 --gpu --flip --shift=4 --tensor-cores --smoothing=0.1
zig build mnist -Dcnn=true -- 20 data/fashion --gpu      # Fashion-MNIST: same program, other data directory
zig build cifar -Dcifar100=true -- 50 --gpu --flip --shift=4 --tensor-cores
zig build cifar -Dwide=true -- 50 --gpu --flip --shift=4 --cutmix=0.5 --tensor-cores --smoothing=0.1 --ema=0.999 --warmup=1 --tta
```

Build options such as `-Dcnn=true -Dhidden=2048 -Dbatch=256` go before the `--`; the arguments after it belong to the program.

### Trainer flags

| Flag | Meaning |
| --- | --- |
| `[epochs]` | Total number of epochs; the cosine learning-rate schedule spans all of them (default 5). |
| `--gpu` | Train on the GPU. |
| `--tensor-cores` | Use bf16 tensor-core GEMMs for large matrix products (f32 accumulate, f32 master weights). |
| `--shift=N`, `--rotate=DEG`, `--zoom=F`, `--flip` | Random affine augmentation applied on the fly, on the CPU or on the GPU. |
| `--tf32` | Use TF32 tensor cores for the large convolutions while keeping every tensor in f32 (activations are read directly, no conversion or im2col buffer). About 2x faster than the strict f32 kernels; operands are rounded to 10 mantissa bits, accumulation stays f32. |
| `--cudnn[=DIR]` | Run convolutions through cuDNN (f32 with TF32 tensor cores) instead of the built-in kernels. The DLLs are loaded at run time from `data/cudnn` (or DIR); if they cannot be found the built-in kernels are used. Fetch them with `scripts/fetch-cudnn.ps1` (about 1.3 GB, never committed). |
| `--cutout=N` | Blank a random N x N square of every training image (CPU and GPU, same math). |
| `--cutmix=P` | With probability P per batch, paste a random square from the neighbouring image and weight the two labels by the pasted area (CutMix; CPU and GPU). |
| `--tta` | After training, also report the test accuracy with every test image averaged with its mirror image (about +0.4 points on CIFAR-10). With `--load` and `--until=0` it evaluates a checkpoint. |
| `--warmup=N` | Ramp the learning rate linearly over the first N epochs, then follow the cosine schedule. |
| `--ema=D` | Keep an exponential moving average of all weights and batch-norm statistics (decay D, e.g. 0.999) and report its test accuracy next to the raw one. Stored in checkpoints. |
| `--batch=N` | Training batch size. The shapes are compile-time, so only the sizes listed in the program's Spec.batches exist in the binary (MNIST: 16 to 512, CIFAR: 32, 64, 128); other values are refused. |
| `--lr=F` | Base learning rate (default: the program's own). |
| `--predict=N` | Print the prediction and confidence for the first N test images (and the logits of image 0). With `--load` this classifies with a saved checkpoint. |
| `--onnx=PATH` | Export the model as an ONNX file (see below). |
| `--smoothing=F` | Label smoothing for the cross-entropy loss (CPU and GPU). |
| `--save=PATH`, `--load=PATH` | Write or read a checkpoint (weights, batch-norm statistics, Adam state, progress). |
| `--until=N` | Stop after epoch `N` but keep the schedule of the full run, so that a later `--load` continues exactly. |
| `--progress` | Force the live progress bar (it is shown automatically on a terminal; redirected output only gets one line per epoch). |
| `--profile` | Print per-kernel GPU timings (every op is synchronised, so absolute numbers are inflated). |

The optimizer is AdamW with a cosine schedule. Checkpoints are portable between CPU and GPU.

## ONNX export

`--onnx=PATH` writes the trained (or loaded) network as an ONNX model (opset 20, no dependencies, hand-written protobuf). The network is run once in evaluation mode while the tape records every op (`Tape.trace`); the recording is turned into a graph with the weights as initialisers. Convolutions and pools are wrapped in layout transposes because the engine is NHWC and ONNX is NCHW, batch norm becomes a per-channel scale and shift, dropout disappears, and reshapes are inserted where a tensor is viewed with another shape. Supported ops: `add`, `mul`, `matmul`, `relu`, `sigmoid`, `tanh`, `gelu`, `conv2d`, `maxPool`, `avgPool`, `batchNorm`, `addRelu`.

```powershell
zig build cifar -- 1 --gpu --predict=1 --onnx=model.onnx
pip install numpy onnx
python scripts/check-onnx.py model.onnx cifar <the ten logits printed above>
```

The script runs the model in the ONNX reference evaluator and compares it with the engine; the MLP, the MNIST CNN and the CIFAR ResNet-14 agree to 5e-6, so the file can be loaded by onnxruntime, OpenVINO and so on, from Python or any other language. Importing is a separate, run-time-shaped inference engine (grad.onnx.Model, src/onnx/runtime.zig), because a graph read at run time has no compile-time type to live in. It runs on the CPU and supports Conv (strides, pads, dilations, groups), MaxPool, AveragePool, GlobalAveragePool, BatchNormalization, Gemm, MatMul, Add/Sub/Mul/Div with broadcasting, Relu, Sigmoid, Tanh, Gelu, Softmax, Flatten, Reshape, Transpose, Concat, Identity, Dropout. zig build onnx -- model.onnx mnist|cifar 1000 loads a model and measures its accuracy; it reproduces the logits of the engine that exported it to 5 digits. Training an imported graph is not possible.

## Optional cuDNN backend

The engine does not need cuDNN, but it can use it: `src/gpu/cudnn.zig` loads `cudnn64_9.dll` dynamically (the same way `nvcuda.dll` is loaded) and drives the legacy descriptor API, with the best algorithm per shape chosen by `cudnnFind*` and cached. `zig build bench -- cudnn` times it on the CIFAR shapes: cuDNN reaches 8 to 10 TFLOPS in f32 (TF32) and 20 to 24 TFLOPS in bf16, versus about 5 TFLOPS for the built-in f32 GEMM and 17 to 18 TFLOPS for the built-in bf16 implicit GEMM. On the CIFAR ResNet-14 an f32 epoch takes 52.8 s with the strict f32 kernels, 24.6 s with the built-in `--tf32` implicit GEMM and 24.5 s with `--cudnn`; the bf16 path (`--tensor-cores`) takes 17.2 s.

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
  nn/                    Linear, Conv2d, BatchNorm, ConvBn, Gru, Adam, checkpoints, trainer
  onnx/                  ONNX export from a recorded tape
  data/                  MNIST / Fashion-MNIST and CIFAR-10 / CIFAR-100 loaders
  gpu/                   driver loader, runtime, device kernels (PTX), parity tests
    math/                per-element math shared by the CPU and the device
examples/
  mnist/  cifar/         image-classification programs built on the trainer
  regression/            a hand-written training loop: MLP, mse, Adam
  charlm/                a character-level GRU unrolled over 64 steps, with text generation
scripts/fetch-data.ps1   dataset download with checksum verification
scripts/check-onnx.py    compare an exported ONNX file with the engine (needs numpy and onnx)
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
- Grouped and dilated convolutions use a simple one-thread-per-output kernel (correct but not tuned); only the dense stride-1 shapes get the tensor-core paths.
- Batch sizes are compile-time constants; a program lists the sizes it is built for (`--batch=N` picks one).
- ONNX files can be imported for CPU inference only, and only single-input graphs can be exported.
- Training is single-GPU and deterministic on the CPU; GPU runs differ in the last bits because of atomic accumulation.
