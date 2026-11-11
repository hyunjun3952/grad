# TODO

The next tasks, most useful first. Keep this list current: when a task is finished, delete it; when new work shows up, add it with a line on why it matters.

## Next

- **CIFAR bf16: where the epoch goes.** Profile of `cifar -Dbf16=true -- 1 --gpu --tensor-cores --profile` (profiling inflates small kernels): implicit-GEMM convolutions 54.5% (the 64-channel weight gradient runs at 13.4 TFLOPS against 16.2 for cuDNN), column reductions of batch norm 8.7%, batch-norm grad input and apply 7.7%, residual add and ReLU 4.7%, the zero arena memset 3.5%, transposeHalf 3.3%. The reductions and element-wise passes already run near the memory bandwidth, so the gain left is in the GEMM tiles or in fusing passes (batch-norm apply into the residual add, for one).
- **CPU convolution scaling.** The im2col matrix, the weight gradient and the input gradient now work in cache-sized chunks, but the MNIST CNN still takes about 24 s on one thread against 9.4 s on six. Profile what is left per layer (packing of B in the small-n products, pooling, batch norm) before adding a direct loop that skips the column matrix.
- **Multi-GPU on real hardware.** `--replicas` is checked against single-replica training only on one GPU. On several GPUs, test it, then replace the host round trip with device-to-device copies (peer access or NCCL-style), and average batch-norm statistics across replicas.
- **Skip the gradient zero-fill.** The first writer of a gradient buffer could store instead of accumulate; the zero arena is re-cleared every step (3.5% of a profiled bf16 CIFAR epoch). It touches the backward pass of every op.
- **transposeHalf in the bf16 convolution weight gradient.** 3.3% of a profiled bf16 CIFAR epoch, already near the memory bandwidth. Removing it needs a transposed-A load in tileGemm (ldmatrix.trans for A); swapping the operands does not work because the gathered im2col side is not contiguous along K.
- **Linux GPU loader.** It compiles but has never run; test it on a Linux machine with a driver.
- **Mixed precision loss scaling.** bf16 does not need it, but an f16 path would.