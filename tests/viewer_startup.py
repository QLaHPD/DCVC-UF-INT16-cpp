#!/usr/bin/env python3
"""Viewer validation/headless startup. Tk failure precedes codec worker creation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

parser=argparse.ArgumentParser();parser.add_argument('--binary',type=Path,required=True);args=parser.parse_args()
binary=str(args.binary.resolve());passed=0

def check(name,command,marker,display_failure=False):
    global passed
    env=os.environ.copy();env['DISPLAY']=''
    result=subprocess.run([binary,'view',*map(str,command)],capture_output=True,text=True,timeout=15,env=env)
    output=result.stdout+result.stderr
    assert result.returncode!=0 and marker in output,(name,result.returncode,output)
    if display_failure:
        assert 'no display name' in output or "couldn't connect to display" in output,(name,'unexpected Tcl/Tk failure',output)
        assert 'checkpoint' not in output.lower() and 'prepared' not in output.lower(),(name,'unexpected codec loading',output)
    passed+=1;print('PASS '+name,flush=True)

with tempfile.TemporaryDirectory(prefix='uf-viewer-startup-') as temporary:
    root=Path(temporary);empty=root/'empty';empty.mkdir()
    check('empty gallery refuses startup',['--input',empty],'No .dcvci images found')
    (empty/'ordinary.png').write_bytes(b'not a UF container')
    check('ordinary image is excluded from UF gallery',['--input',empty],'No .dcvci images found')
    for value in ['0','-1','nan','inf']:
        check('invalid FPS '+value,['--input',empty,'--fps',value],'must be')
    check('runtime selector validation',['--input',empty,'--runtime','other'],'Invalid value for --runtime')
    check('input aliases select the same gallery',['--input',empty,'--input_file',empty],'No .dcvci images found')
    nested=empty/'nested';nested.mkdir();image=nested/'image.DCVCI'
    payload=b'opaque viewer payload';metadata={'format':'dcvc-uf-image','pipeline':{'runtime':'fp16','variant':'image','image_only':True},
        'video':{'width':8,'height':4,'frames':1,'packets':1},'payload_size':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),
        'validation':{'mode':'artifact-hashes','decode_performed':False}}
    header=json.dumps(metadata).encode();image.write_bytes(b'UFIMAGE\x01'+struct.pack('>I',len(header))+header+payload)
    check('nonrecursive gallery excludes nested input',['--input',empty],'No .dcvci images found')
    check('recursive gallery reaches Tk display check',['--input',empty,'--recursive'],'Cannot initialize the UF viewer',True)
    check('single UF image reaches Tk display check',['--input',image],'Cannot initialize the UF viewer',True)
    check('last input alias selects the UF image',['--input',empty,'--input_file',image],'Cannot initialize the UF viewer',True)
    bundle=root/'bundle';bundle.mkdir();stream=b'\0\4\10'+bytes([0x10,30,2,len(payload)])+payload
    (bundle/'video.bin').write_bytes(stream)
    data={'format':'dcvc-uf-archive','version':1,'pipeline':{'runtime':'fp16','variant':'ld'},
        'video':{'width':8,'height':4,'frames':1,'packets':1,'fps':25},'artifacts':{'video.bin':{'size':len(stream),'sha256':hashlib.sha256(stream).hexdigest()}},
        'validation':{'mode':'artifact-hashes','decode_performed':False}}
    (bundle/'manifest.json').write_text(json.dumps(data))
    check('indexed video reaches Tk display check',['--input',bundle],'Cannot initialize the UF viewer',True)
print(str(passed)+' viewer startup checks passed; no GUI, models or checkpoints',flush=True)
