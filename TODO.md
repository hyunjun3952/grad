# TODO

The next tasks, most useful first. Keep this list current: when a task is finished, delete it; when new work shows up, add it with a line on why it matters.

## Next

- **Python wrapper.** A small C ABI over the trainer and the ONNX export (create model, train, predict, save), loaded from Python with ctypes. Decide first which part of the API is worth exposing; the shapes are compile-time, so the wrapper has to list the models it was built for.

## Later

- **Transformer step: layer norm parameter gradients and small element-wise ops.** Profile (`zig build transformer -- 100 --gpu --profile`, profiling inflates every call): layerNormParameters takes 7% at 232 us per call, addInPlace and elementwise together 19% in about 20000 small launches, while the batched attention products are under 10%, so a tensor-core batched GEMM is not worth it.
- **Multi-GPU on real hardware.** `--replicas` is checked against single-replica training only on one GPU. On several GPUs, test it, then replace the host round trip with device-to-device copies (peer access or NCCL-style), and average batch-norm statistics across replicas.
- **Fewer, fatter kernels in the GRU step.** Graph replay (done) saved only about 15%: the step is about 4500 tiny kernels (a 32 x 256 x 256 product on 128 x 128 tiles, one-line element-wise ops) that the profile shows at 12 to 75 us per call (inflated by its per-call synchronisation, so the true device time is lower and unmeasured). Fusing the three input projections and the two recurrent ones of a Gru into one product each, and the gate arithmetic into one element-wise kernel, would cut the count several times over.
- **Skip the gradient zero-fill.** The first writer of a gradient buffer could store instead of accumulate; the zero arena is re-cleared every step.
- **transposeHalf in the bf16 convolution weight gradient.** One pass that could be fused into the producer.
- **Linux GPU loader.** It compiles but has never run; test it on a Linux machine with a driver.
- **Mixed precision loss scaling.** bf16 does not need it, but an f16 path would.