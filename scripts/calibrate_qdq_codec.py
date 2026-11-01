"""Experimental offline QNN QDQ calibration from opt-in real codec fixtures.
Produces research artifacts only; does not enable a backend or execute QNN.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
import onnx
from export_optimized_codec import read_codes

def collect(path):
    arrays=[];seen=set()
    if path.is_dir():items=[(p.name,p.read_bytes()) for p in sorted(path.glob('*.a8c'))]
    else:
        with zipfile.ZipFile(path) as z:
            if len(z.infolist())>100 or sum(i.file_size for i in z.infolist())>20*1024*1024:raise ValueError('Calibration archive exceeds bounds')
            items=[(i.filename,z.read(i)) for i in z.infolist() if i.filename.endswith('.a8c')]
    for name,raw in items:
        digest=hashlib.sha256(raw).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        if len(raw)<12 or raw[:4]!=b'A8C1':raise ValueError('Invalid real-code fixture '+name)
        books,n=np.frombuffer(raw[4:12],dtype='<i4')
        if books!=10 or n<1 or n>128 or len(raw)!=12+40*n:continue
        codes=np.frombuffer(raw[12:],dtype='<i4').reshape(10,int(n))
        if np.any(codes<0) or np.any(codes>=4096):raise ValueError('Invalid code indices')
        padded=np.zeros((10,128),np.int32);padded[:,:n]=codes;padded[1:]=np.minimum(padded[1:],1023);arrays.append(padded)
    if len(arrays)<16:raise ValueError('At least 16 distinct real utterances of <=128 frames are required; random inputs are not a substitute')
    return arrays

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    arrays=collect(a.calibration)
    import onnxruntime as ort
    ort.disable_telemetry_events()
    from onnxruntime.quantization import CalibrationDataReader,QuantType,quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config,qnn_preprocess_model
    model=onnx.load(a.model,load_external_data=False)
    if len(model.graph.input)!=1:raise ValueError('Expected one codes input')
    inp=model.graph.input[0];shape=[d.dim_value for d in inp.type.tensor_type.shape.dim]
    if shape not in [[1,1280],[1,10,128]]:raise ValueError('Expected static T128 codec')
    if any(t.data_type==onnx.TensorProto.FLOAT16 for t in model.graph.initializer):raise ValueError('Use the F32 optimized export, not a weight-only FP16 graph')
    name=inp.name;dtype=np.int32 if inp.type.tensor_type.elem_type==6 else np.int64
    class Reader(CalibrationDataReader):
        def __init__(self):self.rewind()
        def rewind(self):self.iterator=iter(arrays)
        def get_next(self):
            x=next(self.iterator,None)
            if x is None:return None
            return {name:(x.T.reshape(1,-1) if shape==[1,1280] else x[None]).astype(dtype)}
    a.output.mkdir(parents=True,exist_ok=True);pre=a.output/'codec.preprocessed.onnx';out=a.output/'codec.qdq-w8a16.onnx'
    changed=qnn_preprocess_model(str(a.model),str(pre));source=str(pre if changed else a.model)
    config=get_qnn_qdq_config(source,Reader(),activation_type=QuantType.QUInt16,weight_type=QuantType.QUInt8)
    quantize(source,str(out),config)
    (a.output/'qdq-status.json').write_text(json.dumps({'real_calibration_utterances':len(arrays),'weight_bits':8,'activation_integer_bits':16,'qnn_executed':False,'phone_validated':False,'production_enabled':False},indent=2))
if __name__=='__main__':main()
