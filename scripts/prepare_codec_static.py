"""Freeze codec capacities and fold bounded constants using ONNX's reference ops only.
No ORT/QNN import, provider initialization or inference session.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,shape_inference
from onnx.reference import ReferenceEvaluator
SOURCE_SHA='25379b866ad555b9a55226c46325344d2ebfafea1b474c29de5223a9d01ea533'
def freeze(source,capacity,destination):
    raw=Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Unexpected source graph')
    m=onnx.load_model_from_string(raw)
    inp=m.graph.input[0]
    if inp.name!='codes' or inp.type.tensor_type.elem_type!=7:raise ValueError('Unexpected codec input')
    for dim,value in zip(inp.type.tensor_type.shape.dim,[1,10,capacity]):dim.ClearField('dim_param');dim.dim_value=value
    for dim,value in zip(m.graph.output[0].type.tensor_type.shape.dim,[1,1,capacity*2048]):dim.ClearField('dim_param');dim.dim_value=value
    original=len(m.graph.node)
    for iteration in range(12):
        m=shape_inference.infer_shapes(m,data_prop=True)
        shapes={v.name:[d.dim_value if d.HasField('dim_value') else None for d in v.type.tensor_type.shape.dim] for v in [*m.graph.input,*m.graph.value_info,*m.graph.output]}
        values={t.name:numpy_helper.to_array(t) for t in m.graph.initializer if not t.external_data and int(np.prod(t.dims))<=65536}
        nodes=[];folded=0
        for n in m.graph.node:
            result=None
            if n.op_type=='Shape' and n.input[0] in shapes and all(d is not None for d in shapes[n.input[0]]):
                attrs={a.name:helper.get_attribute_value(a) for a in n.attribute};dims=shapes[n.input[0]]
                result=[np.asarray(dims,dtype=np.int64)[attrs.get('start',0):attrs.get('end',len(dims))]]
            elif n.op_type=='Constant':
                try:result=ReferenceEvaluator(n,opsets={x.domain:x.version for x in m.opset_import}).run(None,{})
                except Exception:pass
            elif all(not name or name in values for name in n.input) and n.op_type not in ['RandomNormal','RandomUniform']:
                # Avoid evaluating data/weight-heavy operators; bound broadcast output sizes.
                output_shape=shapes.get(n.output[0])
                safe=output_shape and all(d is not None for d in output_shape) and np.prod(output_shape)<=65536
                if safe and n.op_type not in ['Conv','ConvTranspose','MatMul','Einsum','Tile','Expand']:
                    try:result=ReferenceEvaluator(n,opsets={x.domain:x.version for x in m.opset_import}).run(None,{name:values[name] for name in n.input if name})
                    except Exception:pass
            if result is not None and all(isinstance(v,np.ndarray) and v.size<=65536 for v in result):
                for name,value in zip(n.output,result):
                    values[name]=value;m.graph.initializer.append(numpy_helper.from_array(value,name))
                folded+=1
            else:nodes.append(n)
        del m.graph.node[:];m.graph.node.extend(nodes)
        if not folded:break
    # Drop dead branches and constants, keeping external tensor references unchanged.
    needed={x.name for x in m.graph.output};live=[]
    for n in reversed(m.graph.node):
        if needed.intersection(n.output):live.append(n);needed.update(n.input)
    del m.graph.node[:];m.graph.node.extend(reversed(live))
    weights=[t for t in m.graph.initializer if t.name in needed]
    del m.graph.initializer[:];m.graph.initializer.extend(weights)
    # Remove stale intermediate metadata; infer clean metadata from the final graph.
    del m.graph.value_info[:];m=shape_inference.infer_shapes(m,data_prop=True)
    onnx.save(m,destination)
    return {'capacity':capacity,'original_nodes':original,'nodes':len(m.graph.node),'file':Path(destination).name,'size':Path(destination).stat().st_size,'sha256':hashlib.sha256(Path(destination).read_bytes()).hexdigest()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    print(json.dumps([freeze(a.source,n,out/f'codec-static-{n}.onnx') for n in [128,256]],indent=2))
