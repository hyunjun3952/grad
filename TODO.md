# TODO

The next tasks, most useful first. Keep this list current: when a task is finished, delete it; when new work shows up, add it with a line on why it matters.

## Next

- **Python wrapper.** A small C ABI over the trainer and the ONNX export (create model, train, predict, save), loaded from Python with ctypes. Decide first which part of the API is worth exposing; the shapes are compile-time, so the wrapper has to list the models it was built for.

## Later

- **Transformer step: small element-wise ops.** `zig build transformer -- 100 --gpu --profile` still shows addInPlace and elementwise at about 19% in roughly 20000 small launches, most of them from the 55 Linear layers per step: the bias add after each product and, in the backward pass, the copy of the sum's gradient into the product's gradient. A Tape.linear op with the bias added inside the product (or in place) and a backward pass that writes the product gradient directly would remove about half of them; estimate only, not measured.
- **Multi-GPU on real hardware.** `--replicas` is checked against single-replica training only on one GPU. On several GPUs, test it, then replace the host round trip with device-to-device copies (peer access or NCCL-style), and average batch-norm statistics across replicas.
- **Skip the gradient zero-fill.** The first writer of a gradient buffer could store instead of accumulate; the zero arena is re-cleared every step.
- **transposeHalf in the bf16 convolution weight gradient.** One pass that could be fused into the producer.
- **Linux GPU loader.** It compiles but has never run; test it on a Linux machine with a driver.
- **Mixed precision loss scaling.** bf16 does not need it, but an f16 path would.