"""Structural equivalence/dataflow checks, not a QNN or audio correctness test."""
import argparse,bisect,collections,hashlib,json,math,shutil,tempfile
from pathlib import Path
import onnx
from onnx import numpy_helper
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('parts');a=p.parse_args()
source=onnx.load(a.source,load_external_data=False)
original={n.name:n.SerializeToString() for n in source.graph.node}
original_index={n.name:i for i,n in enumerate(source.graph.node)}
weights={t.name:t.SerializeToString() for t in source.graph.initializer}
root=Path(a.parts);manifest=json.loads((root/'codec-parts.json').read_text())
for cap in manifest['capacities']:
    files=[e for e in manifest['files'] if e['name'].startswith(f'codec-{cap}-')]
    assert len(files)==14
    live={'codes'};metadata={'codes':source.graph.input[0].type.tensor_type.elem_type};prior=-1
    seen_nodes=set();carried_sizes={};use_stages=collections.defaultdict(set)
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        # Sparse placeholder only for external reference bounds checking; never execute it.
        with (tmp/'codec_decoder_fp16.onnx.data').open('wb') as f:f.truncate(260741440)
        for stage,e in enumerate(files):
            path=root/e['name'];assert path.stat().st_size==e['size']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']
            shutil.copyfile(path,tmp/path.name);onnx.checker.check_model(str(tmp/path.name))
            m=onnx.load(path,load_external_data=False)
            assert [v.name for v in m.graph.input]==e['inputs'];assert [v.name for v in m.graph.output]==e['outputs']
            for v in m.graph.input:
                assert v.name in live,('missing input',v.name)
                assert v.type.tensor_type.elem_type==metadata[v.name]
                assert all(d.HasField('dim_value') and d.dim_value>0 for d in v.type.tensor_type.shape.dim)
                use_stages[v.name].add(stage)
            for n in m.graph.node:
                assert n.name not in seen_nodes
                assert n.SerializeToString()==original[n.name],('modified operator',n.name)
                assert original_index[n.name]>prior
                prior=original_index[n.name];seen_nodes.add(n.name)
            for t in m.graph.initializer:
                if t.external_data:assert t.SerializeToString()==weights[t.name],('changed weight reference',t.name)
            for v in m.graph.output:
                shape=v.type.tensor_type.shape
                assert all(d.HasField('dim_value') and d.dim_value>0 for d in shape.dim)
                metadata[v.name]=v.type.tensor_type.elem_type;live.add(v.name)
                carried_sizes[v.name]=math.prod(d.dim_value for d in shape.dim)*onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type).itemsize
    for name,size in carried_sizes.items():
        if size>4096:assert len(use_stages[name])<=1,('buffer has multiple consumers',name)
    assert 'audio' in live
    print(f'capacity={cap}: 14 graph parts, operator/weight identity, complete boundary dataflow and bounded-buffer rules verified; no inference executed')
