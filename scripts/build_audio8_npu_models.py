"""Download pinned public weights and build static ONNX graphs, without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import onnx
from audio8_qnn_model import convert

REPO='stephvax/audio8-tts-realistic-0.6b-onnx-int4'
REVISION='04313606782af6b9a1aa93724555bc7b64ba9197'
FILES={
    'slow_ar_int4.onnx':'eb0d4925019ce6d83e19e56a4bc73d62e64eec0c1c825a51c9b56871d5f10f70',
    'slow_ar_int4.onnx.data':'27fe1e3d74e4a660f02bdf4c3a922f522a26c99ca9c1f78b268eed43f5fd6501',
    'fast_ar_int4.onnx':'41cfbf981e80f0f9c178f5439fd07d36090c8783280c6a82d9df868296aa09fd',
    'fast_ar_int4.onnx.data':'054dc7c25d297ada0e6753f45e90fca5068f989528f33db6ec63c1085050af49',
}

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def download(folder,name,expected=None):
    target=folder/name
    if target.exists():
        if expected and sha(target)==expected: return target
        raise ValueError('Refusing to overwrite existing or unverified input '+str(target))
    partial=target.with_name(target.name+'.part')
    url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
    print('Downloading',name,flush=True)
    try:
        with urllib.request.urlopen(url,timeout=60) as response,partial.open('xb') as out:
            shutil.copyfileobj(response,out,1024*1024)
        if expected and sha(partial)!=expected: raise ValueError('SHA-256 mismatch for '+name)
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.inputs.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()): raise ValueError('Output directory must be empty')
    for name,digest in FILES.items(): download(args.inputs,name,digest)
    for name in ('LICENSE','NOTICE','provenance.json','README.md'):
        download(args.inputs,name)
        shutil.copyfile(args.inputs/name,args.output/('SOURCE-'+name))
    for kind in ('fast','slow'):
        model=convert(onnx.load(args.inputs/(kind+'_ar_int4.onnx')),kind)
        filename=kind+'_static.onnx'
        onnx.save_model(model,args.output/filename,save_as_external_data=True,
            all_tensors_to_one_file=True,location=filename+'.data',size_threshold=1024)
        del model
        print('Prepared',filename,flush=True)
    manifest={
        'source_repository':REPO,'source_revision':REVISION,'source_sha256':FILES,
        'target_soc':'SM8550','target_htp_arch':'v73','decode_sequence_length':1,
        'slow_cache_capacity':2048,'fast_cache_capacity':10,
        'matmul_weights':'unchanged INT4','embeddings':'FP16 dequantization of source INT4 values',
        'prefill':'separate original CPU graph required',
        'codec':'not included in this build',
        'build_status':'static_model_conversion_only',
        'qnn_compilation_run_in_ci':False,'android_backend_integrated':False,
        'on_device_validated':False,
        'validation_note':'Inference execution remains paused after a telemetry safety rejection; this build does not import ONNX Runtime or QNN.',
    }
    (args.output/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files=sorted(p for p in args.output.iterdir() if p.is_file())
    (args.output/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
    print('Static model preparation complete. Not an Android NPU TTS APK.',flush=True)

if __name__=='__main__': main()
