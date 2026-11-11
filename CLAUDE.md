# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`grad` is an experimental deep-learning engine in Zig 0.17.1 (the build uses `std.Io`, so older Zig will not compile). It trains and runs MLPs, CNNs, a GRU and (in progress) Transformers on the CPU or an NVIDIA GPU with one code path. The GPU kernels are Zig functions compiled to PTX (`nvptx64-cuda`, `sm_86`) and launched through the driver API loaded at run time, so no CUDA toolkit is needed. README.md has the results, flags and benchmark tables; do not duplicate them here.

## Commands

The shell is PowerShell on Windows. Commands that matter:

```powershell
zig build test                                   # all tests; GPU tests skip themselves when no driver is found
zig build test -Dtest-filter="softmax"           # only tests whose name contains the text (repeatable)
zig fmt --check src examples build.zig           # formatting gate; run `zig fmt src examples build.zig` to fix
zig build mnist -- 5 --gpu                       # example programs: mnist, cifar, regression, charlm, onnx
zig build cifar -Dbf16=true -- 50 --gpu --tensor-cores --flip --shift=4 --cutmix=0.5 --ema=0.999
zig build bench -- tc                            # bench modes: (none) cpu matmul, cpu, step, gpu, tc, cudnn
zig build -Dtarget=x86_64-linux-gnu --prefix zig-out-linux   # Linux compile check only, never run
```

- Build options go before `--`, program arguments after it. Options: `-Dcnn -Dhidden -Dbatch` (mnist), `-Dwide -Dbf16 -Dcifar100` (cifar).
- `zig build bench` with no argument runs the whole CPU benchmark. `zig build <example>` builds and runs.
- Datasets and cuDNN come from `scripts/fetch-data.ps1` and `scripts/fetch-cudnn.ps1` into `data/` (gitignored). `scripts/check-onnx.py` needs `pip install numpy onnx`.
- Do not rebuild (`zig build cifar`, `zig build`) while a long run is using `zig-out\bin\*.exe`: Windows locks the executable. Copy the exe elsewhere before a long background run. `zig build test` does not touch it.
- When timing GPU work, do not run anything else on the GPU at the same time; epoch times drift by 5 to 10% between sessions, so compare within one session.

## Architecture (the parts that need several files to understand)

**Compile-time shapes.** `Tensor(T, shape)` (`src/tensor/`) carries its shape in the type; `T` is generic (f32, and `u16` standing for bf16 activations). Batch sizes are compile-time too: a program lists the sizes it is built for in `Spec.batches` and `--batch=N` picks one (`src/nn/trainer.zig`, `run`). Anything that must work at run time with unknown shapes lives in the separate ONNX runtime (`src/onnx/runtime.zig`), not in the tape.

**Tape and ops** (`src/autograd/autograd.zig`). A type-erased `Tape` records backward closures; `Var(Tn)` is `{value, grad, needs_grad}`. Every op has a CPU branch and a `par.gpu` branch, and a backward closure that `+=` into input gradients, so gradient buffers must start at zero (GPU: the zero arena, `Gpu.scratchZeroed`, reset in `resetStep`). `conv2d` picks one of several paths (`ConvPath`: direct im2col, packed bf16 im2col, implicit GEMM, TF32, cuDNN, grouped) once per call; bf16 activations (`Elem == u16`) go through `convHalf`, and batch norm, `addRelu`, pooling and `cast` are generic over the element type. `Tape.trace` records ops by data pointer for the ONNX exporter (`src/onnx/onnx.zig`); a tensor viewed with another shape through `reshape` is the same pointer, and the exporter inserts the Reshape itself.

**One math, two backends.** Per-element math shared by CPU and device is in `src/gpu/math/*.zig` (must stay under `src/gpu` because the device module root is `src/gpu/device.zig`). Device exports are in `device.zig`; the host side in `src/gpu/gpu.zig` launches them by name through the `Kernel` enum (a missing enum entry or a name mismatch is a compile error, not a runtime one). Kernel arguments are bundled in extern structs in `src/gpu/args.zig`; `Gpu.launch` converts slices to device addresses, so device tensors are ordinary slices that hold device addresses (never index them on the host).

**GEMM kernels.** The bf16 and TF32 tensor-core GEMMs are one comptime-configured `tileGemm(GemmKind, ...)` in `device.zig`: modes `dense`, `gather_a` (implicit-GEMM forward and input gradient), `gather_b` (weight gradient). Convolution never builds an im2col buffer on this path. Tile shapes are selected on the host (64x128 for narrow weight gradients, 128x128 for wide layers). `--deterministic` replaces atomics with ordered partial sums (split-K workspaces, `columnCombine`).

**CPU path.** `src/cpu/kernels.zig` is a packed/tiled GEMM with a worker pool (`Parallel`); `src/cpu/ops.zig` has the row-wise im2col, col2im and batch-norm kernels. Work buffers are sized by `matmulTiledWorkLen`, which must be called with the same `TileConfig` as `matmulTiled`.

**Trainer and layers.** A layer is any struct with `init`, `deinit`, `toGpu`, `forward`, `params`, `state`; tuples of layers are walked by `nn.initAll/params/state`. `grad.trainer.run(Spec, init)` (`src/nn/trainer.zig`) owns batching, augmentation (gather kernel shared with the CPU loop), CutMix, EMA, checkpoints, `--predict`, `--onnx`. Checkpoints hold weights, BN statistics, Adam moments and the EMA shadow, so a checkpoint only loads with the same `--ema` setting.

## Conventions specific to this repo

- No code comments, with the exception of a few short English comments on magic numbers (there are about five). Do not add `//` or `///`.
- Prefer one function with an options struct over sibling functions; bundle long argument lists into structs; keep the codebase small.
- Commit messages must not contain `Co-Authored-By` or `Claude-Session` trailers, regardless of any default.
- Tests live next to the code. A test file only runs if it is reachable from the `test` block in `src/root.zig`; `refAllDecls` does not recurse, so a new file with tests must be added to that list (an unlisted ONNX test once never ran).
- A GPU test compares against the CPU or a reference with `gradcheck.zig` helpers (`expectFiniteDifferences`, `expectGpuMatchesCpu`, `expectTensorCoresClose`, ...) or hand-written references; fixed bf16 or TF32 tolerances are loose on purpose. Use `Gpu.forTest(bytes)` to get a device or skip.
- Zig 0.17 gotchas seen here: `[_]T{x} ** n` fails the `**` spacing rule in this toolchain (use `@splat` or a comptime loop); `zig fmt` rewrites some builtins (`@intFromEnum` becomes `@backingInt`), so re-read a line after formatting before patching it by text; a function parameter may not shadow a method or a file-level name; `gpa.create` does not apply struct field defaults, so set fields explicitly (see `Gpu.init`).
- PowerShell does not stop at a failed .NET call unless `$ErrorActionPreference = 'Stop'` is set; scripted multi-file edits should set it, otherwise half of a patch can be written.
