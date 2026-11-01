"""Real pinned codec CPU smoke test; host timings are not phone benchmarks."""
import argparse, hashlib, json, urllib.request
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter
REV = '56fcb491ebf03eb3a1a66615fedf6744015dd4c7'
FILES = {
    'codec_decoder_int8_T128.tflite': ('codec_decoder_int8_T128.tflite', '8f7bb7d518bf77e33c3a729aeae55d96b79c592b465b60bd9f78f0143619d80b'),
    'codes.npy': ('voices/en_librispeech_1272/codes.npy', 'fb61a206bd78ee559702d9b1bab8f6f15d1a1e4ce60985a2e6922e56b6b9ffb7'),
}
def main():
    p=argparse.ArgumentParser();p.add_argument('--download',type=Path,required=True);a=p.parse_args();a.download.mkdir(parents=True,exist_ok=True)
    for name,(remote,sha) in FILES.items():
        path=a.download/name
        if not path.exists():
            urllib.request.urlretrieve(f'https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b/resolve/{REV}/{remote}',path)
        with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sha,name
    model=Interpreter(model_path=str(a.download/'codec_decoder_int8_T128.tflite'),num_threads=4);model.allocate_tensors()
    assert model.get_signature_list()=={'decode':{'inputs':['codes'],'outputs':['wav']}}
    assert model.get_input_details()[0]['shape'].tolist()==[1,10,128]
    assert model.get_input_details()[0]['dtype']==np.int32
    runner=model.get_signature_runner('decode');codes=np.load(a.download/'codes.npy')[:,:128].astype(np.int32)[None]
    first=runner(codes=codes)['wav'];second=runner(codes=codes)['wav']
    altered=codes.copy();altered[:,0]=(altered[:,0]+1)%4096;altered[:,1:]=(altered[:,1:]+1)%1024
    other=runner(codes=altered)['wav']
    assert first.shape==(1,1,128*2048) and np.isfinite(first).all()
    rms=float(np.sqrt(np.mean(first**2)));mse=float(np.mean((first-other)**2))
    assert 1e-7<rms<2 and mse>1e-10
    np.testing.assert_allclose(first,second,rtol=1e-5,atol=1e-6)
    print(json.dumps({'host_cpu_smoke':'passed','pcm_rms':rms,'input_sensitivity_mse':mse,'phone_validated':False}))
if __name__=='__main__':main()
