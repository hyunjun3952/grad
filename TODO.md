# TODO

The next tasks, most useful first. Keep this list current: when a task is finished, delete it; when new work shows up, add it with a line on why it matters.

## Next

- **Python wrapper.** A small C ABI over the trainer and the ONNX export (create model, train, predict, save), loaded from Python with ctypes. Decide first which part of the API is worth exposing; the shapes are compile-time, so the wrapper has to list the models it was built for.

## Later

- **Transformer step: small element-wise ops.** Layer norm now uses one warp per row (600 steps took 10.1 s instead of 11.2 to 13.4 s, same losses). What is left in `zig build transformer -- 100 --gpu --profile` (profiling inflates every call): addInPlace and elementwise together take about 19% in roughly 20000 small launches, from residual adds, bias adds and gradient accumulation. Fusing a bias add into the product that feeds it, or accumulating the gradients of a residual branch in place, would remove most of them.
- **Multi-GPU on real hardware.** `--replicas` is checked against single-replica training only on one GPU. On several GPUs, test it, then replace the host round trip with device-to-device copies (peer access or NCCL-style), and average batch-norm statistics across replicas.
- **Skip the gradient zero-fill.** The first writer of a gradient buffer could store instead of accumulate; the zero arena is re-cleared every step.
- **transposeHalf in the bf16 convolution weight gradient.** One pass that could be fused into the producer.
- **Linux GPU loader.** It compiles but has never run; test it on a Linux machine with a driver.
- **Mixed precision loss scaling.** bf16 does not need it, but an f16 path would.