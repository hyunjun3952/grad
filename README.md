# grad

An experimental deep-learning engine in Zig 0.17. It trains MLPs, CNNs, a GRU and Transformers on the CPU or an NVIDIA GPU with one code path. The GPU kernels are Zig functions compiled to PTX and launched through the driver API loaded at run time, so no CUDA toolkit is needed.

| Task | Model | Accuracy | Epoch time |
| --- | --- | --- | --- |
| MNIST | MLP 784-128-10 | 98.1% | 0.12 s GPU, 0.30 s CPU |
| MNIST | CNN + BatchNorm | 99.5% | 0.75 s GPU, 7.7 s CPU |
| CIFAR-10 | ResNet-14 64-128-256 | 94.1% (50 epochs), 94.9% (100) | 11.9 s GPU (bf16), 21.9 s (TF32), 29.3 s (f32) |
| CIFAR-10 | ResNet-14 64-192-384 | 95.3% (50 epochs), **96.2%** (100), both with `--tta` | 17.0 s GPU (bf16) |

Measured on an i5-8500 (6 cores) and an RTX 3060 12 GB. Accuracies use flip, shift, CutMix, label smoothing and EMA. Timings drift between sessions, so compare builds with `scripts/compare.ps1` inside one session.

## Highlights

- **Compile-time shapes.** `Tensor(f32, &.{ 64, 784 })` carries its shape in the type, so a wrong `matmul` or `reshape` is a compile error. Batch sizes are compile-time too.
- **Small autograd.** A type-erased `Tape` records backward closures; ops include `matmul`, `conv2d`, pooling, batch and layer norm, softmax, embedding, GRU gates and the losses.
- **One math, two backends.** Per-element math is written once and shared by the CPU worker pool and the GPU kernels.
- **GPU kernels in Zig.** Tiled f32 GEMM, bf16 and TF32 tensor-core GEMMs, implicit-GEMM convolutions with no im2col buffer, fused batch norm and Adam, bf16 activations, optional cuDNN, and training steps replayed as CUDA graphs.
- **ONNX.** Export from the recorded tape, and an inference runtime for imported models on the CPU or the GPU.
- **Tested against references.** GPU against CPU, gradients against finite differences, checkpoint resume bit for bit.

## Requirements

- Zig 0.17.1.
- For the GPU: an NVIDIA GPU with Ampere or newer (`sm_86`). No CUDA toolkit and no Python (Python is only for `scripts/check-onnx.py`).
- Developed on Windows. CI builds and tests on Windows and Linux without a GPU; the Linux GPU path has never run.

## Quick start

```powershell
zig build test
powershell -File scripts/fetch-data.ps1             # datasets into data/

zig build mnist -- 5 --gpu                          # leave out --gpu for the CPU
zig build mnist -Dcnn=true -- 30 --gpu --shift=1 --rotate=8 --zoom=0.05
zig build cifar -Dbf16=true -- 50 --gpu --tensor-cores --flip --shift=4 --cutmix=0.5 --smoothing=0.1 --ema=0.999 --tta
zig build charlm -- 1500 --gpu                      # character GRU on Tiny Shakespeare
zig build transformer -- 1500 --gpu                 # character Transformer
zig build regression                                # library API without the trainer
zig build bench -- tc                               # benchmarks: (none), cpu, step, gpu, tc, cudnn
```

Build options (`-Dcnn -Dhidden -Dbatch -Dwide -Dbf16 -Dcifar100`) go before the `--`, program arguments after it.

### Trainer flags

| Flag | Meaning |
| --- | --- |
| `[epochs]` | Number of epochs; the cosine schedule spans all of them. |
| `--gpu`, `--batch=N`, `--lr=F` | Device, batch size (only sizes listed in the program's Spec.batches exist), learning rate. |
| `--tensor-cores`, `--tf32`, `--cudnn[=DIR]` | bf16 GEMMs, TF32 convolutions, or cuDNN convolutions (`scripts/fetch-cudnn.ps1`). |
| `--deterministic` | Reproducible GPU runs (no atomics), somewhat slower. |
| `--flip`, `--shift=N`, `--rotate=DEG`, `--zoom=F`, `--cutout=N`, `--cutmix=P` | Augmentation. |
| `--smoothing=F`, `--warmup=N`, `--ema=D`, `--tta` | Label smoothing, learning-rate warm-up, weight EMA, mirror-averaged test accuracy. |
| `--replicas=N` | Data-parallel training over N GPU replicas (only checked on one GPU). |
| `--save=PATH`, `--load=PATH`, `--until=N` | Checkpoints; `--until` stops early but keeps the full schedule. |
| `--predict=N`, `--onnx=PATH` | Print predictions, export an ONNX model. |
| `--profile`, `--progress`, `--scratch=N` | Per-kernel timings, progress bar, device arena size in MiB. |

The optimizer is AdamW. Checkpoints hold weights, batch-norm statistics, Adam moments and the EMA shadow, and are portable between CPU and GPU.

## ONNX

`--onnx=PATH` writes the network as an ONNX model (opset 20, no dependencies) with a symbolic batch axis. Check it with `python scripts/check-onnx.py model.onnx cifar <logits>`; the MLP, CNN, ResNet and Transformer agree with the engine to 5e-6. `grad.onnx.Model` (`src/onnx/runtime.zig`) loads an ONNX file for inference (`zig build onnx -- model.onnx mnist 1000`); training an imported graph is not possible. `grad.onnx_gpu.Executor` (`src/onnx/gpu.zig`) runs the same graph on the device with run-time shapes (`zig build onnx -- model.onnx cifar 1000 100 --gpu`, optionally `--tf32`); an op without a device kernel (Slice, Concat, Div, Sqrt) runs on the host, and the example prints how many nodes did. On a CIFAR ResNet-14 at batch 100 the device path reads 4.6k images/s against 0.2k on the CPU with the same logits (6.2k with `--tf32`); the MNIST CNN reads 130k against 8.5k (one session, 1000 to 2000 images, one run each).

## Layout

```
src/
  root.zig       public API
  tensor/        Tensor(T, shape)
  autograd/      Tape, Var, ops
  nn/            layers, Adam, checkpoints, trainer
  cpu/           worker pool, tiled matmul, im2col
  gpu/           driver loader, runtime, device kernels (math/ is shared with the CPU)
  onnx/          export, CPU inference runtime and GPU executor
  data/          MNIST and CIFAR loaders
tests/           one test file per source file
examples/        mnist, cifar, regression, charlm, transformer, onnx
scripts/         dataset and cuDNN download, compare.ps1, check-onnx.py
```

## A training program

`examples/mnist` reduced to one small CNN:

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

A layer is any struct with `init`, `deinit`, `toGpu`, `forward`, `params` and `state`; `grad.trainer.run` handles batching, augmentation, evaluation, checkpoints and the GPU.
