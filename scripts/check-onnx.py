"""Check an exported model against the engine: python scripts/check-onnx.py MODEL.onnx mnist|cifar|text LOGITS...

LOGITS are the numbers printed by `--predict=1` (logits of image 0, ten of them), or for the Transformer example
(`zig build transformer -- 20 --gpu --onnx=model.onnx`) the 128 scores of the last position that it prints.
Needs: pip install numpy onnx (the onnx package ships a slow but exact reference evaluator).
"""
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

model_path, dataset, expected = sys.argv[1], sys.argv[2], [float(v) for v in sys.argv[3:]]
model = onnx.load(model_path)
onnx.checker.check_model(model, full_check=True)
print("checker ok;", len(model.graph.node), "nodes:", sorted({n.op_type for n in model.graph.node}))

if dataset == "text":
    context = 64
    text = open("data/text/shakespeare.txt", "rb").read()[:context]
    feeds = {"input0": np.array([c & 0x7F for c in text], dtype=np.int32), "input1": np.arange(context, dtype=np.int32)}
    logits = ReferenceEvaluator(model).run(None, feeds)[0][-1].reshape(-1)
    image = None
elif dataset == "mnist":
    raw = open("data/mnist/t10k-images-idx3-ubyte", "rb").read()
    image = np.frombuffer(raw, dtype=np.uint8, offset=16, count=784).astype(np.float32) / 255.0
else:
    raw = open("data/cifar/test_batch.bin", "rb").read()
    planar = np.frombuffer(raw, dtype=np.uint8, offset=1, count=3072).astype(np.float32).reshape(3, 1024) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None]
    deviation = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)[:, None]
    image = ((planar - mean) / deviation).T.reshape(-1).astype(np.float32)

if image is not None:
    # A symbolic batch dimension has no dim_value (0); one image is fed in that case.
    shape = [d.dim_value or 1 for d in model.graph.input[0].type.tensor_type.shape.dim]
    logits = ReferenceEvaluator(model).run(None, {"input": image.reshape(shape)})[0].reshape(-1)
print("onnx  :", " ".join(f"{v:.5f}" for v in logits))
print("engine:", " ".join(f"{v:.5f}" for v in expected))
error = float(np.max(np.abs(logits - np.array(expected, dtype=np.float32))))
print("max abs difference:", error)
sys.exit(0 if error < 1e-3 else 1)