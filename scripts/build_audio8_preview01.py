"""Pinned Preview 0.1B GGUF conversion and a native ggml CPU smoke test.

No ONNX Runtime/QNN imports or execution. Source checkpoint code is not imported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import urllib.request
import wave

REPO='Edge0/Audio8-TTS-Preview-0.1b'
REVISION='8c837020ab800498834038363dea06757872afda'
FILES={
    'model.safetensors':('sha256','69b162eb71da66b3a3f6ed7f9aa5e623853526717fd798d347b34919d58a4734',339605208),
    'codec.pth':('sha256','c310505aa11fe2f6cc63b8d3130dc7e77e73227774f5c62575769b1f47a8d048',1349857559),
    'config.json':('git','82753e3275c9eb6284a707b41dbe10db091d1629',2651),
    'tokenizer.json':('git','f6fb8c3a6fb6dd9cecebd42f0fdc16e60b7eb264',5852397),
    'tokenizer_config.json':('git','03b4c7969d2de1b2905645cb85a9ef42a0b77acb',848277),
    'LICENSE':('git','554ce8eb0bb41a1e0610cd65904ce07516c219d7',6973),
}
NAME='audio8-preview-0.1b-q8_0.gguf'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def fetch(folder,name,spec):
    kind,expected,size=spec
    target=folder/name
    if target.exists():raise ValueError('Input already exists: '+name)
    part=target.with_suffix(target.suffix+'.part')
    digest=hashlib.sha256() if kind=='sha256' else hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    print('Downloading pinned source',name,size,flush=True)
    count=0
    with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',timeout=120) as response,part.open('xb') as out:
        while block:=response.read(1024*1024):
            count+=len(block)
            if count>size:raise ValueError('Oversize input '+name)
            digest.update(block);out.write(block)
    if count!=size or digest.hexdigest()!=expected:raise ValueError('Pinned source verification failed: '+name)
    part.rename(target)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.inputs.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Output must be empty')
    for name,spec in FILES.items():fetch(a.inputs,name,spec)
    config=json.loads((a.inputs/'config.json').read_text())
    if config.get('slow_backbone')!='falcon_h1' or config.get('dim')!=512:raise ValueError('Wrong small-model architecture')
    subprocess.run([sys.executable,str(a.runtime/'tools/community_models/convert_audio8_tts_codec.py'),
        str(a.inputs/'codec.pth'),str(a.inputs/'codec.safetensors')],check=True)
    subprocess.run([sys.executable,str(a.runtime/'tools/community_models/convert_audio8_tts.py'),
        '--model-dir',str(a.inputs),'--converter',str(a.build/'bin/audiocpp_gguf'),
        '--output-dir',str(a.output),'--name','audio8-preview-0.1b','--type','q8_0'],check=True)
    model=a.output/NAME
    with model.open('rb') as stream:
        if stream.read(4)!=b'GGUF':raise ValueError('Not a GGUF')
    # Run only native audio.cpp/ggml CPU. No provider library, Python remote code,
    # ORT/QNN environment or previously blocked host inference test is loaded.
    sample=a.inputs/'smoke.wav'
    subprocess.run([str(a.build/'bin/audiocpp_cli'),'--family','audio8_tts','--task','tts',
        '--model',str(model),'--text','你好。','--seed','42','--threads','2',
        '--max-tokens','96','--metrics','--out',str(sample)],check=True,timeout=240)
    with wave.open(str(sample),'rb') as wav:
        if wav.getframerate()!=44100 or wav.getnchannels()!=1 or wav.getsampwidth()!=2 or wav.getnframes()<2048:
            raise ValueError('Unexpected native smoke-test audio format')
        raw=wav.readframes(wav.getnframes())
    samples=struct.unpack('<'+'h'*(len(raw)//2),raw)
    rms=(sum(float(v)*v for v in samples)/len(samples))**0.5/32768
    if not 0.00001<rms<0.9:raise ValueError('Silent or invalid native smoke-test audio')
    shutil.copyfile(a.inputs/'LICENSE',a.output/'LICENSE.txt')
    (a.output/'NOTICE.txt').write_text('Copyright © 2026 Audio8. Source: '+REPO+' at '+REVISION+'\nModified: converted to audio.cpp GGUF Q8_0 using pinned runtime 3a0dd44293fc706f5c375ec96d3e6fd9c54b6c45.\nNo endorsement is implied. Audio8 Community License v1.0 retained.\n')
    info={'schema':1,'source':REPO,'revision':REVISION,'file':NAME,'size':model.stat().st_size,'sha256':sha(model),
        'license':'Audio8 Community License v1.0','native_cpu_smoke_passed':True,'smoke_rms':rms,
        'phone_validated':False,'npu_supported':False,'source_files':FILES}
    (a.output/'model-info.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
