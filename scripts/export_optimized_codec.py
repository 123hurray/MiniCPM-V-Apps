"""Offline codec-only export with source-level folding and a required real-code parity gate.
Requires a trusted local official checkpoint, PyTorch, Transformers, ONNX and NumPy.
No AR model is loaded; never run as part of app startup.
"""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
import numpy as np

def read_codes(path):
    raw=Path(path).read_bytes()
    if len(raw)<12 or raw[:4]!=b'A8C1':raise ValueError('Expected a real exported A8C1 fixture')
    books,frames=np.frombuffer(raw[4:12],dtype='<i4')
    if books!=10 or not 1<=frames<=128 or len(raw)!=12+40*frames:raise ValueError('Invalid fixture shape')
    codes=np.frombuffer(raw[12:],dtype='<i4').copy().reshape(10,int(frames))
    if np.any(codes<0) or np.any(codes>=4096):raise ValueError('Invalid code indices')
    return codes

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--codes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=a.checkpoint/'codec.pth'
    if hashlib.file_digest(data.open('rb'),'sha256').hexdigest()!='c310505aa11fe2f6cc63b8d3130dc7e77e73227774f5c62575769b1f47a8d048':raise ValueError('Unexpected source codec')
    # Local checkpoint Python is trusted code supplied explicitly by the caller.
    sys.path.insert(0,str(a.checkpoint.resolve()));sys.path.insert(0,str(Path(__file__).parent/'vendor'))
    import torch,copy,types
    package=types.ModuleType("audio8_codec_checkpoint");package.__path__=[str(a.checkpoint.resolve())];sys.modules[package.__name__]=package
    config_module=importlib.import_module("audio8_codec_checkpoint.configuration_arktts")
    codec_module=importlib.import_module("audio8_codec_checkpoint.modeling_arktts_codec")
    sys.modules["modeling_arktts_codec"]=codec_module
    ArkttsConfig=config_module.ArkttsConfig
    from dakeqq_codec import load_audio8_codec,prepare_codec_for_export,A8_DECODER
    config=ArkttsConfig(**json.loads((a.checkpoint/'config.json').read_text()))
    original=load_audio8_codec(a.checkpoint,config);optimized=copy.deepcopy(original)
    stats=prepare_codec_for_export(optimized)
    wrapper=A8_DECODER(optimized,44100,44100,128).float().eval()
    codes=read_codes(a.codes);tensor=torch.from_numpy(codes.astype(np.int64))[None]
    with torch.no_grad():
        reference=original.decode(tensor)
        if isinstance(reference,(tuple,list)):reference=reference[0]
        actual,_=wrapper(torch.from_numpy(codes.T.reshape(1,-1)))
        x=reference.detach().cpu().float().numpy().reshape(-1);y=actual.detach().cpu().float().numpy().reshape(-1)
        if x.shape!=y.shape:raise ValueError('Decoder parity shape mismatch')
        error=float(np.max(np.abs(x-y)));rms=float(np.sqrt(np.mean((x-y)**2)))
        if not np.all(np.isfinite(y)) or error>1e-3:raise ValueError(f'Codec parity failed max_abs={error} rms={rms}')
    a.output.mkdir(parents=True,exist_ok=True)
    dummy=torch.zeros((1,1280),dtype=torch.int32)
    torch.onnx.export(wrapper,dummy,str(a.output/'codec-optimized-f32-T128.onnx'),input_names=['codes_flat'],output_names=['audio','samples'],opset_version=18,dynamo=False)
    (a.output/'export-validation.json').write_text(json.dumps({'max_abs':error,'rms_error':rms,'source_codec_sha256':'c310505aa11fe2f6cc63b8d3130dc7e77e73227774f5c62575769b1f47a8d048','folding':stats,'phone_validated':False},indent=2))
if __name__=='__main__':main()
