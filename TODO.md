# TODO

The next tasks, most useful first. Keep this list current: when a task is finished, delete it; when new work shows up, add it with a line on why it matters.

## Next

- **Python wrapper.** A small C ABI over the trainer and the ONNX export (create model, train, predict, save), loaded from Python with ctypes. Decide first which part of the API is worth exposing; the shapes are compile-time, so the wrapper has to list the models it was built for.
- **Export a Transformer to ONNX.** Trace layer norm, softmax (with the causal mask), embedding, batched matmul and dropout, and import them in the runtime. Without this only the convolutional and MLP models export.

## Later

- **Tensor-core path for batched matmul.** The f32 kernels take a batch axis, but with --tensor-cores a batch is still one packed product per matrix, and bf16 attention operands are widened to f32. A tileGemm mode with a batch axis and a packing kernel for all matrices would use the tensor cores for attention; measure first whether the Transformer is limited by these products at all.
- **Multi-GPU on real hardware.** `--replicas` is checked against single-replica training only on one GPU. On several GPUs, test it, then replace the host round trip with device-to-device copies (peer access or NCCL-style), and average batch-norm statistics across replicas.
- **Speed of the GRU example.** The character GRU is launch-bound (about 70 ms per step): CUDA graphs or one fused kernel per step.
- **Skip the gradient zero-fill.** The first writer of a gradient buffer could store instead of accumulate; the zero arena is re-cleared every step.
- **transposeHalf in the bf16 convolution weight gradient.** One pass that could be fused into the producer.
- **Linux GPU loader.** It compiles but has never run; test it on a Linux machine with a driver.
- **Mixed precision loss scaling.** bf16 does not need it, but an f16 path would.