# TODO

The next tasks, most useful first. Keep this list current: when a task is finished, delete it; when new work shows up, add it with a line on why it matters.

## Next

- **Multi-GPU data parallel.** One replica per device, gradients all-reduced through the host (copy, sum, copy back), then the same Adam step on every replica. First test it with two replicas on one GPU against single-replica training on the same data; only then think about NCCL-style device-to-device copies. Not measured on real multi-GPU hardware yet.
- **Python wrapper.** A small C ABI over the trainer and the ONNX export (create model, train, predict, save), loaded from Python with ctypes. Decide first which part of the API is worth exposing; the shapes are compile-time, so the wrapper has to list the models it was built for.
- **Batched tensor-core GEMM for attention.** batchedMatmul still launches one product per matrix, and bf16 operands are widened to f32 because packing each small matrix costs more than the product. One launch with a batch axis (grid z) and a packing kernel that handles all matrices would remove thousands of launches per step; the bf16 Transformer is not faster than f32 until then.
- **Export a Transformer to ONNX.** Trace layer norm, softmax (with the causal mask), embedding, batched matmul and dropout, and import them in the runtime. Without this only the convolutional and MLP models export.

## Later

- **Speed of the GRU example.** The character GRU is launch-bound (about 70 ms per step): CUDA graphs or one fused kernel per step.
- **Skip the gradient zero-fill.** The first writer of a gradient buffer could store instead of accumulate; the zero arena is re-cleared every step.
- **transposeHalf in the bf16 convolution weight gradient.** One pass that could be fused into the producer.
- **Linux GPU loader.** It compiles but has never run; test it on a Linux machine with a driver.
- **Mixed precision loss scaling.** bf16 does not need it, but an f16 path would.