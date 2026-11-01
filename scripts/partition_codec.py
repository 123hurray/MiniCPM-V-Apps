"""Split fixed codec graphs at residual boundaries; preserve all live tensors exactly."""
from pathlib import Path
import argparse,copy,hashlib,json,bisect,math
import onnx
from onnx import helper,shape_inference
from prepare_codec_static import freeze,optimize

def partition(source,capacity,out):
    full=out/f'full-{capacity}.onnx';freeze(source,capacity,full)
    m=onnx.load(full,load_external_data=False)
    # Causal decoder residual endpoints have exact channel/time dimensions.
    known={}
    for block,(channels,stride) in enumerate([(768,32),(384,256),(192,1024),(96,2048)],1):
        for residual in [2,3,4]:known[f'/decoder/model/model.{block}/block/block.{residual}/Add_output_0']=[1,channels,capacity*stride]
    infos={v.name:v for v in m.graph.value_info}
    for name,shape in known.items():
        if name not in infos:raise ValueError('Unknown residual endpoint '+name)
        info=infos[name];typ=info.type.tensor_type.elem_type
        info.CopyFrom(helper.make_tensor_value_info(name,typ,shape))
    optimize(m,capacity,full)
    m=onnx.load(full,load_external_data=False)
    for v in m.graph.value_info:
        if v.name in known:v.CopyFrom(helper.make_tensor_value_info(v.name,v.type.tensor_type.elem_type,known[v.name]))
    cuts=[]
    # Small memory domains: quantizer+transformer, decoder initial conv, and each residual unit.
    for index,node in enumerate(m.graph.node):
        if node.op_type=='Conv' and node.name=='/decoder/model/model.0/conv/Conv':cuts.append(index)
        if node.op_type=='Add' and node.output[0] in known:cuts.append(index)
    cuts.append(len(m.graph.node)-1)
    if len(cuts)!=14:raise ValueError(f'Unexpected stage count {len(cuts)}')
    info={v.name:v for v in [*m.graph.input,*m.graph.value_info,*m.graph.output]}
    weights={t.name:t for t in m.graph.initializer};producer={name:i for i,n in enumerate(m.graph.node) for name in n.output}
    consumers={}
    for i,n in enumerate(m.graph.node):
        for name in n.input:consumers.setdefault(name,[]).append(i)
    final={v.name for v in m.graph.output};start=0;entries=[]
    for part,end in enumerate(cuts):
        nodes=list(m.graph.node[start:end+1]);inputs=[];outputs=[];needed=set()
        for n in nodes:
            for name in n.input:
                if not name:continue
                if name in weights:needed.add(name)
                elif producer.get(name,-1)<start and name not in inputs:inputs.append(name)
            for name in n.output:
                if name in final or any(i>end for i in consumers.get(name,[])):outputs.append(name)
        for name in inputs+outputs:
            if name not in info:raise ValueError('Missing tensor metadata '+name)
            shape=info[name].type.tensor_type.shape
            if any(not d.HasField('dim_value') or d.dim_value<1 for d in shape.dim):raise ValueError('Non-static boundary '+name+' '+str(shape))
        for name in outputs:
            typ=info[name].type.tensor_type
            size=math.prod(d.dim_value for d in typ.shape.dim)*helper.tensor_dtype_to_np_dtype(typ.elem_type).itemsize
            downstream={bisect.bisect_left(cuts,index) for index in consumers.get(name,[]) if index>end}
            if size>4096 and len(downstream)>1:raise ValueError("Large tensor has multiple consuming stages: "+name)
        graph=helper.make_graph(nodes,f'codec-{capacity}-stage-{part}',[info[n] for n in inputs],[info[n] for n in outputs],[weights[n] for n in weights if n in needed])
        model=helper.make_model(graph,opset_imports=list(m.opset_import),ir_version=m.ir_version)
        name=f'codec-{capacity}-part{part:02d}.onnx';path=out/name;onnx.save(model,path)
        entries.append({'name':name,'size':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'inputs':inputs,'outputs':outputs})
        start=end+1
    full.unlink()
    return entries
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest={'schema':1,'capacities':[128,256],'parts':14,'files':sum([partition(a.source,c,out) for c in [128,256]],[])}
    (out/'codec-parts.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(hashlib.sha256((out/'codec-parts.json').read_bytes()).hexdigest())
