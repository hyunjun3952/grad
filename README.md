# grad

An experimental deep-learning engine written in Zig 0.17. It trains MLPs and CNNs on the CPU or on an NVIDIA GPU with the same code, and it needs no CUDA toolkit: the GPU kernels are Zig functions compiled to PTX by the Zig compiler and launched through the driver API (`nvcuda.dll`).

The goal is an engine that sits between "C with pointers" and "PyTorch": plain structs, explicit allocators and explicit parallelism, but shapes are checked by the compiler and autograd is a normal library.

| Task | Model | Accuracy | Time per epoch |
| --- | --- | --- | --- |
| MNIST | MLP 784-128-10 | 98.1% (10 epochs) | 0.21 s GPU, 0.32 s CPU (6 threads) |
| MNIST | CNN conv5x16-conv5x32 + BatchNorm | 99.5% (30 epochs, light augmentation) | 0.87 s GPU, 13.2 s CPU |
| CIFAR-10 | ResNet-14, widths 64-128-256 | 94.1% (50 epochs) to 94.9% (100 epochs); flip + shift + CutMix + label smoothing + EMA, +0.4 with `--tta` | 17.2 s GPU (bf16 tensor cores), 25.0 s (TF32), 53.7 s (strict f32) |
| CIFAR-10 | ResNet-14, widths 64-192-384 (`-Dwide=true`) | 94.8% / 95.3% with `--tta` (50 epochs), 95.4% / **96.2%** with `--tta` (100 epochs); same recipe | 24.7 s GPU (bf16 tensor cores) |

Measured on an i5-8500 (6 cores) and an RTX 3060 12 GB. About 8,200 lines of code in src and examples, plus 3,400 lines of tests (110 tests).

## Highlights

- **Compile-time shapes.** `Tensor(f32, &.{ 64, 784 })` carries its shape in the type, so a wrong `matmul`, `add` or `reshape` is a compile error. Data lives on the heap.
- **Small autograd.** A type-erased `Tape` records closures; `Var{ value, grad, needs_grad }` is a plain struct. Ops: `add`, `mul`, `matmul`, `relu`, `conv2d`, `maxPool`, `avgPool`, `batchNorm`, `dropout`, `crossEntropy`, `mse`, `sigmoid`/`tanh`/`gelu` (`Tape.activate`), `gruGates`/`gruNext` (the two fused halves of a GRU step), `reshape`. Layers: `Linear`, `Conv2d` (stride, padding, dilation, groups), `BatchNorm`, `ConvBn`, `Gru`. `maxPool`/`avgPool` take a window, stride and padding (`PoolOpts`).
- **One code path for CPU and GPU.** Pass `kernels.Parallel{ .io, .threads, .gpu }` and everything (tape, optimizers, matmul) dispatches to the CPU worker pool or to the device. Per-element math such as `im2col`, dropout, batch-norm and augmentation is written once and shared by both backends.
- **GPU without a toolkit.** `src/gpu/device.zig` is compiled for `nvptx64-cuda` / `sm_86`, embedded as PTX and loaded with `cuModuleLoadData`. It contains tiled FP32 GEMM (about 5 TFLOPS), a bf16 tensor-core GEMM using `cp.async`, `ldmatrix` and `mma.sync` with a three-stage pipeline (about 21 TFLOPS at 4096, 23.7 without operand packing), reductions, fused multi-tensor Adam, and so on. Convolutions with 64 or more input channels run as implicit GEMMs: the tensor-core kernel gathers its operand tiles straight from the NHWC activations with zero-filled `cp.async` (forward, input gradient via rotated weights, and weight gradient), so no im2col buffer is ever written. Batch norm + ReLU and add + ReLU are fused ops. On the CIFAR ResNet-14 the bf16 path makes an epoch 3.1x faster than FP32 with the same accuracy.
- **Tested against references.** Every GPU op is compared with the CPU, gradients with finite differences, convolutions with a direct implementation, and bf16 GEMM with a CPU reference built from the same rounded inputs. Resuming from a checkpoint reproduces uninterrupted training bit for bit.

## Requirements

- Zig 0.17.1.
- For the GPU path: an NVIDIA GPU with Ampere or newer (the kernels target `sm_86`). The driver and cuDNN are loaded at run time through one small loader (`src/gpu/dynlib.zig`): `nvcuda.dll` / `cudnn64_9.dll` on Windows, `libcuda.so.1` (then `libcuda.so`) / `libcudnn.so.9` on Linux, the same names and order that other CUDA wrappers such as [oxicuda](https://docs.rs/oxicuda-driver/0.1.7/src/oxicuda_driver/loader.rs.html) and the LLVM OpenMP [CUDA plugin](https://llvm.googlesource.com/llvm-project/+/890c4bece26e005cd9fa5511fe0efa7307794de5/offload/plugins-nextgen/cuda/dynamic_cuda/cuda.cpp) use. Windows is what the project is developed and tested on; all programs also cross-compile for Linux (`zig build -Dtarget=x86_64-linux-gnu`), but that path has not been run on a Linux machine. Pick a GPU with `CUDA_VISIBLE_DEVICES`. GPU tests are skipped when no driver is found.
- No CUDA toolkit, no Python.

## Quick start

```powershell
zig build test
powershell -File scripts/fetch-data.ps1            # MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 and Tiny Shakespeare into data/ (or -Dataset NAME)

zig build regression                                # fit a curve with a small MLP (library API without the trainer)
zig build charlm -- 1500 --gpu                      # character-level GRU on Tiny Shakespeare, then sample text
zig build transformer -- 1500 --gpu                 # character-level Transformer (3 causal blocks, width 128), then sample text
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
| `--replicas=N` | Data parallel training on the GPU: the batch is split into N slices, one replica (thread, own copy of the model) per slice, gradients averaged through host memory before the identical Adam step. Replicas use devices round-robin; on one GPU this is only a correctness check and is slower than a single replica. `--batch` stays the whole batch, so batch / N must be a built size. Not measured on real multi-GPU hardware. Loss, evaluation and batch-norm statistics come from replica 0. |
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

`--onnx=PATH` writes the trained (or loaded) network as an ONNX model (opset 20, no dependencies, hand-written protobuf). The network is run once in evaluation mode while the tape records every op (`Tape.trace`); the recording is turned into a graph with the weights as initialisers. Convolutions and pools are wrapped in layout transposes because the engine is NHWC and ONNX is NCHW, batch norm becomes a per-channel scale and shift, dropout disappears, and reshapes are inserted where a tensor is viewed with another shape. Supported ops: `add`, `mul`, `matmul`, `batchedMatmul`, `embedding`, `layerNorm`, `softmax` (also the causal one, exported as a mask added before Softmax), `relu`, `sigmoid`, `tanh`, `gelu`, `conv2d`, `maxPool`, `avgPool`, `batchNorm`, `addRelu`. An embedding takes its ids as a graph input (int32, `View.ofIds`), so the Transformer example (`zig build transformer -- 20 --gpu --onnx=model.onnx`, one sequence of 64 characters) exports a graph with the inputs `input0` (characters) and `input1` (positions); `python scripts/check-onnx.py model.onnx text <the 128 scores it prints>` agrees with the engine to 5e-6 in f32, and to about 0.02 for a model run with `-Dbf16=true`, which is the rounding of its bf16 activations (the graph itself is f32).

```powershell
zig build cifar -- 1 --gpu --predict=1 --onnx=model.onnx
pip install numpy onnx
python scripts/check-onnx.py model.onnx cifar <the ten logits printed above>
```

The script runs the model in the ONNX reference evaluator and compares it with the engine; the MLP, the MNIST CNN and the CIFAR ResNet-14 agree to 5e-6, so the file can be loaded by onnxruntime, OpenVINO and so on, from Python or any other language. Importing is a separate, run-time-shaped inference engine (grad.onnx.Model, src/onnx/runtime.zig), because a graph read at run time has no compile-time type to live in. It runs on the CPU and supports Conv (strides, pads, dilations, groups), MaxPool, AveragePool, GlobalAveragePool, BatchNormalization, Gemm, MatMul, Add/Sub/Mul/Div with broadcasting, Relu, Sigmoid, Tanh, Gelu, Softmax, LayerNormalization, Gather (along axis 0), Slice, Flatten, Reshape, Transpose, Concat, Identity, Dropout. zig build onnx -- model.onnx mnist|cifar 1000 loads a model and measures its accuracy; it reproduces the logits of the engine that exported it to 5 digits. Training an imported graph is not possible.

## bf16 activations

`Tensor(T, shape)` is generic over its element type, and on the GPU the activations of a network can be kept in bf16 (stored as `u16`) between layers while parameters, parameter gradients, optimizer state and the loss stay in f32. `Tape.cast(x, u16)` and `Tape.cast(x, f32)` convert in both directions and pass gradients through; batch norm (also fused with ReLU), `addRelu`, `add` and `mul` (with broadcasting), `relu`, `activate`, `dropout`, `layerNorm`, `softmax` (also causal), max and average pooling and stride-1 convolutions accept bf16 tensors directly; layer norm keeps its scale, shift and statistics in f32, and sums in every op run in f32. A bf16 convolution gathers its input straight from the bf16 activations and writes bf16, so the pack and unpack passes of the f32 path disappear, and the element-wise kernels move half the bytes. `zig build cifar -Dbf16=true -- 50 --gpu --tensor-cores ...` casts after the stem and back before the classifier; the CIFAR epoch goes from 18.3 s to 15.4 s in the same session. Limits: GPU only, convolutions need a multiple of 64 input and 32 output channels (checked at compile time), and everything is covered except the CPU. `matmul` multiplies bf16 activations by f32 weights (or two bf16 matrices) on tensor cores with f32 accumulation and a rounded result; `batchedMatmul` widens its bf16 operands once, multiplies each small matrix in f32 and rounds once, because packing attention-sized matrices for tensor cores costs more than the products; `embedding` gathers bf16 rows from an f32 table; the losses convert to f32 first. The ONNX exporter treats a cast as an identity, so a bf16 network exports as an f32 graph (the ops that the exporter does not know, such as layer norm, softmax, dropout and embedding, are not exported in either type).

## Optional cuDNN backend

The engine does not need cuDNN, but it can use it: `src/gpu/cudnn.zig` loads `cudnn64_9.dll` dynamically (the same way `nvcuda.dll` is loaded) and drives the legacy descriptor API, with the best algorithm per shape chosen by `cudnnFind*` and cached. `zig build bench -- cudnn` times it on the CIFAR shapes: cuDNN reaches 8 to 10 TFLOPS in f32 (TF32) and 20 to 24 TFLOPS in bf16, versus about 5 TFLOPS for the built-in f32 GEMM and 17 to 18 TFLOPS for the built-in bf16 implicit GEMM. On the CIFAR ResNet-14 an f32 epoch takes 53.7 s with the strict f32 kernels, 25.0 s with the built-in `--tf32` implicit GEMM and 24.1 s with `--cudnn`; the bf16 path (`--tensor-cores`) takes 17.2 s.

## Benchmarks

Measured on an Intel i5-8500 (6 cores, no hyper-threading) and an NVIDIA RTX 3060 12 GB, Windows 11, Zig 0.17.1, `ReleaseFast`. Times are the last epoch of a short run (the first one includes warm-up); GPU kernel numbers are the best of 20 runs. Every number can be reproduced with the commands below. To compare two builds, `scripts/compare.ps1 -Base <old exe> -New <new exe> -ProgramArgs ... -Pattern ...` runs them alternately after a warm-up and prints the minimum and median; with the training time the examples print, five rounds agree to about 1%, while whole-process times drifted by 10% and more.

```powershell
zig build bench                 # CPU matmul kernels, allocation overhead, MLP step profile
zig build bench -- cpu          # per-op CPU times of the MNIST CNN
zig build bench -- gpu          # FP32 GEMM on the device
zig build bench -- tc           # FP32 vs bf16 tensor-core GEMM, implicit-GEMM convolutions
zig build bench -- cudnn        # cuDNN convolutions on the CIFAR shapes (needs the DLLs)
```

### Training, seconds per epoch

| Model | Backend | Epoch |
| --- | --- | --- |
| MNIST MLP 784-128-10, batch 64 | CPU, 6 threads | 0.29 s (0.32 s before the element-wise ops ran on the worker pool) |
| | GPU | 0.21 s |
| MNIST / Fashion-MNIST CNN | CPU, 6 threads | 9.4 s (11.0 s before the convolution input gradient also ran in chunks of images, 11.8 s before the im2col matrix was built in cache-sized chunks, 12.8 s before the element-wise ops ran on the worker pool and the example fused batch norm with ReLU; median of five alternating runs) |
| | GPU | 0.87 s |
| CIFAR-10 ResNet-14 64-128-256, batch 64 | GPU, bf16 tensor cores (`--tensor-cores`) | **17.2 s** |
| | GPU, TF32 tensor cores (`--tf32`) | 25.0 s |
| | GPU, cuDNN TF32 (`--cudnn`) | 24.1 s |
| | GPU, strict f32 | 53.7 s |
| | GPU, bf16, `--deterministic` | 18.7 s |
| CIFAR-100 ResNet-14 (`-Dcifar100=true`) | GPU, bf16 | 17.3 s |
| CIFAR-10 ResNet-14 64-192-384 (`-Dwide=true`) | GPU, bf16 | 25.3 s |
| Character GRU, 256 units, 32 x 64 steps | GPU | about 70 ms per step (whole run, including start-up and the 600-character sample); after the forward and backward pass was recorded once into a CUDA graph (`Gpu.beginCapture`, `endCapture`, `replay`) and replayed, 300 steps took 17.1 s against 20.1 s before in one session, both times with the same losses. The step had about 4500 tiny kernels, so the replay showed that launch cost was only part of the 70 ms. After the three projections of the GRU were fused (one product for the input, one for the update and reset parts of the state, one for the candidate) and the gate arithmetic became two kernels (`Tape.gruGates`, `Tape.gruNext`), 300 steps took 11.2 to 12.5 s against 17.5 to 20.4 s for the unfused layer in the same session; the loss after 300 steps is 2.163 against 2.140 (a different initialisation, since the weights have other shapes). An exported GRU is written out as Slice, Add, Sigmoid, Tanh, Mul and Sub nodes. The f32 GEMM then got a block-count rule: a product whose 128 x 128 tiles, even with the largest K split, would give fewer blocks than the card has multiprocessors (the GRU step has 32 rows and weight gradients with a K of 32) takes the 32 x 32 tile. Measured as training time only (the examples print it; median of five alternating runs with `scripts/compare.ps1`, spread under 1%), 300 steps went from 10.60 s to 4.01 s with that rule and the warp-per-row loss kernels, and to 3.81 s with `Tape.linear` (bias added in place, no separate sum), same loss; `zig build bench -- gpu` and the CIFAR epoch stayed the same. The same trick on the Transformer example gave no measurable change and is not used there |
| Character Transformer, 3 blocks, width 128, 4 heads, 32 x 64 characters | GPU | 84.6 s for 1500 steps (whole run, same accounting as the GRU row); loss 1.580 nats/char, the GRU reaches 1.579 in 101.5 s in the same session; with `-Dbf16=true` (bf16 activations) the same run takes 67.2 s against 66.8 s for f32 in a later session, loss 1.581, so on a model this small bf16 is not faster; after batched matmul became one launch with a batch axis (f32 kernels), 600 steps took 30.5 to 31.1 s before and 11.9 s after in one session (f32 and bf16 alike, identical losses), and 1500 steps took 22.9 s in total (not compared against the old build at 1500 steps); later, measured as training time only (median of five alternating runs), 600 steps went from 7.49 s to 6.31 s with the GEMM tile rule and softmax and cross entropy running one warp per row, and to 5.99 s with `Tape.linear`, same loss; the Transformer has about 640K parameters against about 330K for the GRU (counted by hand, not measured) |

The MNIST CNN on the CPU went from 24.6 s to 13.2 s with the row-wise im2col/col2im, the stored max-pool winners and the vectorised batch-norm statistics (see `zig build bench -- cpu`); the bf16 CIFAR epoch has been 31.7, 25.0, 17.6 and now 17.2 s over the last rounds (fused ops, implicit GEMM, 64x128 and 128x128 tiles).

### GPU kernels (TFLOPS, kernel only)

| Kernel | Shape | TFLOPS |
| --- | --- | --- |
| FP32 SIMT GEMM | 4096 x 4096 x 4096 | 5.1 |
| bf16 tensor-core GEMM (cp.async, ldmatrix, mma.sync) | 4096 x 4096 x 4096 | 24.1 (21.7 with operand packing) |
| bf16 implicit-GEMM conv forward | 3x3, 64 -> 64, 32x32, batch 64 | 17.8 |
| | 3x3, 128 -> 128, 16x16 (128x128 tile) | 19.8 |
| | 3x3, 256 -> 256, 8x8 | 18.6 |
| bf16 implicit-GEMM conv weight gradient | the same three shapes | 13.4 / 18.5 / 17.6 |
| TF32 implicit-GEMM conv forward | the same three shapes | 9.1 / 8.9 / 8.8 |
| TF32 implicit-GEMM conv weight gradient | the same three shapes | 5.8 / 7.7 / 7.7 |
| cuDNN 9.28, bf16 forward / dgrad / wgrad | 64 -> 64 | 23.8 / 19.1 / 16.2 |
| | 128 -> 128 | 21.0 / 23.6 / 17.5 |
| | 256 -> 256 | 22.5 / 19.5 / 19.9 |
| cuDNN 9.28, TF32 forward | 64 / 128 / 256 channels | 8.4 / 10.0 / 10.6 |

The bf16 tensor-core path has a relative RMS error of about 2e-3 against FP32. The built-in bf16 convolution reaches 75 to 90% of cuDNN; the element-wise kernels (batch norm, ReLU, packing) already run at the memory bandwidth of the card (about 430 GB/s measured on batch-norm apply), which is why activations stored in bf16 are the next speed-up.

### CPU kernels

| Kernel | Result |
| --- | --- |
| tiled + packed GEMM, 6x2 register tile, 1 thread | 100 to 112 GFLOPS (512 x 512: 111.7) |
| `@Vector` triple loop / naive triple loop | 17 to 20 / 1.3 to 2.2 GFLOPS |
| MLP training step, batch 64, 1 thread / 6 threads | 378 us (2.6k steps/s) / 149 us (6.7k steps/s) |

Per-op CPU time of the MNIST CNN at batch 64 (forward / backward, ms): conv 5x5 1->16 0.71 / 0.20, conv 5x5 16->32 0.52 / 2.45, max-pool 0.87 / 0.06, batch norm + ReLU 0.62 / 0.54. The first backward went from 0.39 to 0.20 ms when the tiled GEMM learned to split K across threads for products with few outputs and a long K (its weight gradient, 25 x 16 outputs over 36864 positions); the epoch did not change measurably (13.50 against 13.49 s) because most of a step was not in the convolutions: timing the layers of the forward pass in a real run showed ReLU, bias add and gradient accumulation at 0.9 ms or more each on tensors of 2.4 MB, since those loops ran on one thread. They now run on the worker pool for equal shapes and for a vector added to every row, and the example uses the fused batch-norm ReLU. Thread scaling is still poor (24.3 s on one thread, 13.1 s on six): the direct convolution builds an im2col buffer of 15 MB for the first layer, and a path without it is the next step.

### Inference through ONNX (`zig build onnx`, CPU, 6 threads)

| Model | Speed |
| --- | --- |
| MNIST CNN | 1.6k images/s |
| CIFAR ResNet-14 | 37 images/s (convolutions as im2col + tiled GEMM; it was 6 images/s with direct loops) |
## Layout

```
src/
  root.zig               public API (grad.nn, grad.optim, grad.trainer, ...)
  tensor/                Tensor(T, shape), broadcasting ops
  cpu/kernels.zig        worker pool, tiled matmul, flush-denormals
  autograd/              Tape and Var
  nn/                    Linear, Conv2d, BatchNorm, ConvBn, Gru, Adam, checkpoints, trainer
  onnx/                  ONNX export from a recorded tape
  data/                  MNIST / Fashion-MNIST and CIFAR-10 / CIFAR-100 loaders
  gpu/                   driver loader, runtime, device kernels (PTX)
    math/                per-element math shared by the CPU and the device
tests/                   one test file per source file, plus the finite-difference and GPU-vs-CPU helpers (gradcheck.zig)
examples/
  mnist/  cifar/         image-classification programs built on the trainer
  regression/            a hand-written training loop: MLP, mse, Adam
  charlm/                a character-level GRU unrolled over 64 steps, with text generation
  transformer/           a character-level Transformer with learned positions, 32 x 64 characters per step
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

- The Linux GPU loader compiles but is untested; activations are f32 or (on the GPU, for a subset of ops) bf16, in NHWC, tensors of rank 6 or less on the GPU.
- Grouped and dilated convolutions use a simple one-thread-per-output kernel (correct but not tuned); only the dense stride-1 shapes get the tensor-core paths.
- Batch sizes are compile-time constants; a program lists the sizes it is built for (`--batch=N` picks one).
- ONNX files can be imported for CPU inference only. Exported graphs have the batch size they were traced with.
- Multi-GPU training (`--replicas`) exists but has only been run with replicas sharing one GPU. The CPU is deterministic, and so is the GPU with `--deterministic` (no split-K, ordered reductions, deterministic cuDNN algorithms, somewhat slower); without it GPU runs differ in the last bits because of atomic accumulation.
