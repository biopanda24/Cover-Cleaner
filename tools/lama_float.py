"""Write a float32 copy of the LaMa model for GPU inference.

The model zoo file stores weights as block-quantized int8 (DequantizeLinear
with block_size), which ONNX Runtime's CUDA provider cannot run.  Each
DequantizeLinear on constant inputs is evaluated once here and replaced by
its float result.  The batch size is then fixed to 1 and the graph's shape
arithmetic (Shape -> Gather -> Concat -> Pad/Reshape ...) folded into
constants, so no part of the network has to run on the CPU between GPU
layers.

    python tools/lama_float.py [src.onnx] [dst.onnx]
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dequantize(q, scale, zp, axis, block):
    q = q.astype(np.float32)
    zp = np.zeros_like(scale) if zp is None else zp.astype(np.float32)
    if scale.ndim == 0:                                   # per tensor
        return (q - zp) * scale
    axis = axis % q.ndim
    if block:                                             # blocked along axis
        scale = np.repeat(scale, block, axis=axis)
        zp = np.repeat(zp, block, axis=axis)
        sl = [slice(None)] * q.ndim
        sl[axis] = slice(0, q.shape[axis])
        return (q - zp[tuple(sl)]) * scale[tuple(sl)]
    shape = [1] * q.ndim                                  # per channel
    shape[axis] = -1
    return (q - zp.reshape(shape)) * scale.reshape(shape)


def fold_constants(m, rounds=20, max_elems=1 << 22):
    """Evaluate every node whose inputs are known constants (or, for Shape,
    whose input shape is static) and replace it by an initializer."""
    from onnx import shape_inference
    from onnx.reference import ReferenceEvaluator
    g = m.graph
    total = 0
    for _ in range(rounds):
        try:                                     # follows Einsum, Range, Reshape ...
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
            inferred = SymbolicShapeInference.infer_shapes(m, auto_merge=True)
        except Exception:
            inferred = shape_inference.infer_shapes(m)
        shapes = {}
        for v in list(inferred.graph.value_info) + list(inferred.graph.input):
            dims = v.type.tensor_type.shape.dim
            if v.type.tensor_type.HasField("shape") and all(d.HasField("dim_value") for d in dims):
                shapes[v.name] = [d.dim_value for d in dims]
        known = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        keep, folded = [], 0
        for n in g.node:
            out = None
            if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
                out = [numpy_helper.to_array(n.attribute[0].t)]
            elif n.op_type == "Shape" and n.input[0] in shapes and not n.attribute:
                out = [np.array(shapes[n.input[0]], np.int64)]
            elif n.input and all((not x) or x in known for x in n.input)                     and n.op_type not in ("RandomNormal", "RandomUniform", "RandomNormalLike",
                                          "RandomUniformLike", "Multinomial"):
                try:
                    ev = ReferenceEvaluator(onnx.helper.make_model(
                        onnx.helper.make_graph(
                            [n], "one",
                            [onnx.helper.make_tensor_value_info(x, onnx.helper.np_dtype_to_tensor_dtype(known[x].dtype), known[x].shape) for x in n.input if x],
                            [onnx.helper.make_tensor_value_info(o, onnx.TensorProto.UNDEFINED, None) for o in n.output]),
                        opset_imports=m.opset_import))
                    out = ev.run(None, {x: known[x] for x in n.input if x})
                except Exception:
                    out = None
            if out is not None and all(np.asarray(v).size <= max_elems for v in out):
                for name, v in zip(n.output, out):
                    v = np.asarray(v)
                    known[name] = v
                    g.initializer.append(numpy_helper.from_array(v, name))
                folded += 1
                continue
            keep.append(n)
        del g.node[:]
        g.node.extend(keep)
        used = {x for n in g.node for x in n.input} | {o.name for o in g.output}
        live = [t for t in g.initializer if t.name in used]
        del g.initializer[:]
        g.initializer.extend(live)
        del g.value_info[:]
        total += folded
        if not folded:
            break
    return total


def convert(src, dst):
    m = onnx.load(src)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    consts = {}
    for n in g.node:                                      # Constant nodes count too
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)

    def value(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        return consts.get(name)

    keep, folded = [], 0
    for n in g.node:
        if n.op_type == "DequantizeLinear":
            vals = [value(x) if x else None for x in n.input]
            if vals[0] is not None and vals[1] is not None:
                attrs = {a.name: a.i for a in n.attribute}
                w = dequantize(vals[0], vals[1], vals[2] if len(vals) > 2 else None,
                               attrs.get("axis", 1), attrs.get("block_size", 0))
                g.initializer.append(numpy_helper.from_array(w.astype(np.float32), n.output[0]))
                folded += 1
                continue
        keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    used = {x for n in g.node for x in n.input} | {o.name for o in g.output}
    live = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(live)
    for v in list(g.input) + list(g.output):          # static batch of one
        d0 = v.type.tensor_type.shape.dim[0]
        d0.ClearField("dim_param")
        d0.dim_value = 1
    fold_constants(m)
    # Opsets 19-21 only added int4/float8 types (and Pad's "wrap" mode) to the
    # ops this model uses; the CUDA provider registers some of them only up to
    # opset 18 and would run them on the CPU.  Nothing here needs more than 18.
    for o in m.opset_import:
        if o.domain in ("", "ai.onnx") and o.version > 18:
            o.version = 18
    onnx.save(m, dst)
    return folded


if __name__ == "__main__":
    from covercleaner import ml
    src = sys.argv[1] if len(sys.argv) > 1 else ml.model_path()
    dst = sys.argv[2] if len(sys.argv) > 2 else ml.float_model_path()
    print("folded", convert(src, dst), "DequantizeLinear nodes ->", dst,
          f"({os.path.getsize(dst) / 1e6:.0f} MB)")
