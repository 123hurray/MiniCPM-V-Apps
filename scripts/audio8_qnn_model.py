"""Prepare Audio8 INT4 single-token graphs for SM8550 QNN.

Weights stay INT4 except the embedding tables: QNN HTP does not implement
GatherBlockQuantized. Dequantizing that lookup table is not requantization.
The dynamic CPU prefill graph is intentionally kept separate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper


def make_dim_param_fixed(graph, name: str, value: int) -> None:
    # Pure protobuf operation: preparing models must not import an inference
    # runtime or initialize its telemetry/provider libraries.
    for tensor in (*graph.input, *graph.output, *graph.value_info):
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_param == name:
                dim.ClearField('dim_param')
                dim.dim_value = value


def convert(model: onnx.ModelProto, kind: str) -> onnx.ModelProto:
    if kind not in ('slow', 'fast'): raise ValueError('Unknown graph kind')
    capacity, layers = (2048, 24) if kind == 'slow' else (10, 4)
    make_dim_param_fixed(model.graph, 'batch', 1)
    if kind == 'slow': make_dim_param_fixed(model.graph, 'sequence', 1)
    index = {t.name: t for t in model.graph.initializer}
    remove, converted = set(), {}
    for node in model.graph.node:
        if node.op_type != 'GatherBlockQuantized': continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (attrs.get('quantize_axis'), attrs.get('gather_axis'), attrs.get('block_size')) != (1, 0, 32):
            raise ValueError('Unsupported embedding quantization contract')
        old = node.input[0]
        if old not in converted:
            weights = numpy_helper.to_array(index[old])
            scales = numpy_helper.to_array(index[node.input[2]]).astype(np.float32)
            zeros = numpy_helper.to_array(index[node.input[3]]).astype(np.float32)
            values = np.empty(weights.shape, dtype=np.float16)
            for start in range(0, weights.shape[0], 2048):
                end = min(start + 2048, weights.shape[0])
                rows = weights[start:end].astype(np.float32).reshape(end-start, -1, 32)
                values[start:end] = ((rows - zeros[start:end, :, None]) * scales[start:end, :, None]).reshape(end-start, -1)
            name = old + '_dequant_fp16'
            model.graph.initializer.append(numpy_helper.from_array(values, name))
            converted[old] = name
            del weights, scales, zeros, values
        remove.update([old, node.input[2], node.input[3]])
        indices = node.input[1]
        node.op_type, node.domain = 'Gather', ''
        del node.input[:]; node.input.extend([converted[old], indices])
        del node.attribute[:]; node.attribute.extend([helper.make_attribute('axis', 0)])
    kept = [t for t in model.graph.initializer if t.name not in remove]
    del model.graph.initializer[:]; model.graph.initializer.extend(kept)
    del kept, index
    producers = {v: n for n in model.graph.node for v in n.output}
    position_name = 'mobile_full_positions'
    model.graph.initializer.append(numpy_helper.from_array(
        np.arange(capacity, dtype=np.int64).reshape(1, 1, capacity, 1), position_name))
    replacement = [helper.make_node('Equal', [position_name, 'input_pos'], ['mobile_cache_slot'])]
    count = 0
    for node in model.graph.node:
        if node.name == '/Range':
            replacement.append(helper.make_node('Constant', [], list(node.output),
                value=numpy_helper.from_array(np.arange(capacity, dtype=np.int64))))
        elif node.op_type == 'Concat' and len(node.input) == 2 and node.input[0] in producers:
            first = producers[node.input[0]]
            if first.op_type == 'Gather' and first.input[0].startswith(('cache_key_', 'cache_value_')):
                replacement.append(helper.make_node('Where',
                    ['mobile_cache_slot', node.input[1], first.input[0]], list(node.output)))
                count += 1
            else: replacement.append(node)
        else: replacement.append(node)
    if count != layers * 2: raise ValueError(f'KV contract mismatch: {count}')
    del model.graph.node[:]; model.graph.node.extend(replacement)
    del replacement, producers
    del model.graph.value_info[:]
    model = onnx.utils.Extractor(model).extract_model(
        [v.name for v in model.graph.input], [v.name for v in model.graph.output])
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    onnx.checker.check_model(model)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['slow','fast'], required=True)
    args = p.parse_args()
    if args.output.exists() or Path(str(args.output)+'.data').exists():
        raise SystemExit('Refusing to overwrite prepared model')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = convert(onnx.load(args.source), args.kind)
    onnx.save_model(model, args.output, save_as_external_data=True,
        all_tensors_to_one_file=True, location=args.output.name+'.data', size_threshold=1024)
    print(json.dumps({'kind':args.kind, 'output':str(args.output),
        'decode_steps':1, 'cache_capacity':2048 if args.kind=='slow' else 10}))

if __name__ == '__main__': main()
