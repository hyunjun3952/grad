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
zig build mnist -- 5 --gpu                       # example programs: mnist, cifar, regression, charlm, transformer, onnx
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

**Tape and ops** (`src/autograd/autograd.zig`). A type-erased `Tape` records backward closures; `Var(Tn)` is `{value, grad, needs_grad}`. Every op has a CPU branch and a `par.gpu` branch, and a backward closure that `+=` into input gradients, so gradient buffers must start at zero (GPU: the zero arena, `Gpu.scratchZeroed`, reset in `resetStep`). `conv2d` picks one of several paths (`ConvPath`: direct im2col, packed bf16 im2col, implicit GEMM, TF32, cuDNN, grouped) once per call; bf16 activations (`Elem == u16`) go through `convHalf`, and every other op is generic over the element type (the device kernel has a `...Half` twin; the CPU branch returns `error.NoDevice` for bf16). `matmul` takes bf16 activations with f32 weights, `embedding` takes an output type, the losses widen to f32, and `cast` is traced as an identity for ONNX. `Tape.trace` records ops by data pointer for the ONNX exporter (`src/onnx/onnx.zig`); a tensor viewed with another shape through `reshape` is the same pointer, and the exporter inserts the Reshape itself.

**One math, two backends.** Per-element math shared by CPU and device is in `src/gpu/math/*.zig` (must stay under `src/gpu` because the device module root is `src/gpu/device.zig`). Device exports are in `device.zig`; the host side in `src/gpu/gpu.zig` launches them by name through the `Kernel` enum (a missing enum entry or a name mismatch is a compile error, not a runtime one). Kernel arguments are bundled in extern structs in `src/gpu/args.zig`; `Gpu.launch` converts slices to device addresses, so device tensors are ordinary slices that hold device addresses (never index them on the host).

**GEMM kernels.** The bf16 and TF32 tensor-core GEMMs are one comptime-configured `tileGemm(GemmKind, ...)` in `device.zig`: modes `dense`, `gather_a` (implicit-GEMM forward and input gradient), `gather_b` (weight gradient). Convolution never builds an im2col buffer on this path. Tile shapes are selected on the host (64x128 for narrow weight gradients, 128x128 for wide layers). `--deterministic` replaces atomics with ordered partial sums (split-K workspaces, `columnCombine`).

**CPU path.** `src/cpu/kernels.zig` is a packed/tiled GEMM with a worker pool (`Parallel`); `src/cpu/ops.zig` has the row-wise im2col, col2im and batch-norm kernels. Work buffers are sized by `matmulTiledWorkLen`, which must be called with the same `TileConfig` as `matmulTiled`.

**Trainer and layers.** A layer is any struct with `init`, `deinit`, `toGpu`, `forward`, `params`, `state`; tuples of layers are walked by `nn.initAll/params/state`. `grad.trainer.run(Spec, init)` (`src/nn/trainer.zig`) owns batching, augmentation (gather kernel shared with the CPU loop), CutMix, EMA, checkpoints, `--predict`, `--onnx`. Checkpoints hold weights, BN statistics, Adam moments and the EMA shadow, so a checkpoint only loads with the same `--ema` setting.

## Conventions specific to this repo

- **Never ignore this rule: every sentence the user can see is Korean.** That includes the short status lines written between tool calls, summaries, questions and error explanations, not only the final answer. Code, identifiers, comments, commit messages and repository documents (README.md, CLAUDE.md, TODO.md) stay in English.

- A public item gets a `///` doc comment: a one-line summary in the third person, at most two or three more sentences, parameters described inline rather than in a list, and a short fenced example only where it clarifies the call. Never use backticks inside comments; write names and expressions as plain text. Bodies get `//` comments that say why (a constraint, a device quirk, a numerical reason), tagged `NOTE:` or `TODO:`, never what the next line does. Large files are divided with `// ***** name *****` banners. Private helpers are commented only where the logic is not obvious. Keep a comment true when the code changes.
- Prefer one function with an options struct over sibling functions; bundle long argument lists into structs; keep the codebase small.
- Commit messages must not contain `Co-Authored-By` or `Claude-Session` trailers, regardless of any default.
- Tests live in `tests/`, one file per source file (`tests/nn.zig` tests `src/nn/nn.zig`), and use the library only through `@import("grad")` as `lib`. A test file only runs if `tests/all.zig` imports it (an unlisted ONNX test once never ran). A test can reach only `pub` declarations, so make a function `pub` when a test needs it, and internals that no program uses are exposed from `src/root.zig` (`grad.math`, `grad.cpu_ops`, `grad.gpu_args`, ...).
- A GPU test compares against the CPU or a reference with the `tests/gradcheck.zig` helpers (`expectFiniteDifferences`, `expectGpuMatchesCpu`, `expectTensorCoresClose`, ...) or hand-written references; fixed bf16 or TF32 tolerances are loose on purpose. Use `Gpu.forTest(bytes)` to get a device or skip.
- Zig 0.17 gotchas seen here: `[_]T{x} ** n` fails the `**` spacing rule in this toolchain (use `@splat` or a comptime loop); `zig fmt` rewrites some builtins (`@intFromEnum` becomes `@backingInt`), so re-read a line after formatting before patching it by text; a function parameter may not shadow a method or a file-level name; `gpa.create` does not apply struct field defaults, so set fields explicitly (see `Gpu.init`).
- PowerShell does not stop at a failed .NET call unless `$ErrorActionPreference = 'Stop'` is set; scripted multi-file edits should set it, otherwise half of a patch can be written.
- Keep `TODO.md` current: it lists the next tasks, most useful first. Delete a task when it is done, add new work with a line on why it matters, and update it in the same commit as the change that finishes or creates the task.

## Mistakes that happened repeatedly (do not repeat them)

- **A test counts as passing only if you saw it run.** `zig build test` prints nothing on success, so an empty `Select-String` proves nothing, and a test file missing from `tests/all.zig` is never compiled. After adding a test, run `zig build test --summary all` and check that the pass count rose; break the code once on purpose to see the test fail.
- **Report only what was measured.** Label estimates and anything never run (the Linux loader, multi-GPU) as such. Timings drift between sessions, so compare inside one session against a re-measured baseline of the previous commit.
- **Ask before committing or pushing** unless the message says to ("커밋하고 푸시해").
- **Read the current text before patching it.** `zig fmt` renames builtins, breaks lines and drops trailing commas, so an exact-match patch written from an older copy fails or applies halfway. Use unique anchors, set `$ErrorActionPreference = 'Stop'` in scripted edits, and run `git diff --stat` after a failure.
- **Scripted patch helpers broke on PowerShell array unrolling** ("count 814 ... for: p": the pairs were flattened into strings, so the helper searched for one letter). Prefer the Edit tool for source edits. If a script is needed, build each pair as a hashtable or [pscustomobject] with named fields, never as nested arrays returned from a function, and never write backticks in double-quoted strings (backtick plus a letter is an escape). Test the helper on one pair before sending the full batch.
- **Commit messages: write the text to a file and use git commit -F file.** Piping a here-string into git commit -F - passes the text as a pathspec in this shell and fails. Never put backticks in the message either.
- **Check `git status` before `git add -A`.** The Linux compile check writes to `zig-out-linux/`, and one `git add -A` committed those binaries (they are now ignored and untracked, but still in the history of one pushed commit). Add new output directories to `.gitignore` first.
- **Do not trust hand-computed expected values** (three were wrong: grouped convolution, dilated convolution, a causal limit). Compute them with an independent reference, and when a test fails decide whether the code or the expectation is wrong before editing either.
- **Test inputs and device memory:** list every `Inputs` field in a gradcheck case's `names` (an uninitialised slice segfaults); make `acceptable` accept a reasonable share of random inputs (a strict filter looped for ten minutes); device slices hold device addresses, so never `@memcpy` or index them on the host; large GPU tests need a large `Gpu.forTest` arena.
- **Compile errors that keep recurring:** `**` array repetition, locals that shadow file-level names or parameters (`nn`, `cpu`, `act`, `scores`, `p`, `state`, `shape`), and unused locals left behind by a refactor.
- **No scratch code left behind:** remove debug prints and placeholder lines; write the test cleanly the first time.
- **Do not rebuild an exe that a background run uses** (copy it first), do not start other GPU work while timing, and use `run_in_background` for anything longer than a few minutes.
- **Try a tool behaviour before documenting it.** `--test-filter` is not accepted at run time by this Zig's test runner; it is the compile-time `-Dtest-filter` option here.
