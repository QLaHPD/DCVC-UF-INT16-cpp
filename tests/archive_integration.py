#!/usr/bin/env python3
"""Synthetic archive/packet CLI checks. Opaque payloads are never neurally decoded."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,required=True)
args=parser.parse_args()
binary=str(args.binary.resolve())
passed=0

def adaptive(value):
    if value<128:return bytes([value])
    if value<16384:return bytes([0x80|(value>>8),value&255])
    return struct.pack('>I',value|0xc0000000)

def packet(kind=1,identity=0,qp=30,parallel=1,reset=0,payload=b'opaque'):
    return bytes([(kind<<4)|identity,qp,(parallel<<1)|reset])+adaptive(len(payload))+payload

def check(name,command,expected=0,marker=None):
    global passed
    result=subprocess.run([binary,*command],capture_output=True,text=True,timeout=30)
    assert (result.returncode==0 if expected==0 else result.returncode!=0),(name,result.returncode,result.stdout,result.stderr)
    if marker:assert marker in result.stdout+result.stderr,(name,result.stdout,result.stderr)
    passed+=1;print('PASS '+name,flush=True)
    return result

with tempfile.TemporaryDirectory(prefix='uf-archive-cli-') as temporary:
    root=Path(temporary)
    for runtime in ['fp16','int16']:
        for variant in ['hts','htl','ld']:
            folder=root/(runtime+'-'+variant);folder.mkdir()
            preamble=b'' if runtime=='fp16' else b'UF16\x01\0\0\0'+bytes(32)+bytes([1])*32
            stream=preamble+bytes([0])+adaptive(4)+adaptive(8)+packet()+packet(kind=2,reset=1)
            payload=folder/'clip.bin';payload.write_bytes(stream)
            pipeline={'runtime':runtime,'variant':variant}
            if runtime=='int16':pipeline['integer']={'arithmetic':'uf-int16-q9-w13-acc64-rna-sat-v1','image':'00'*32,'video':'01'*32}
            metadata={'format':'dcvc-uf-archive','version':1,'layout':'flat','archive_stem':'clip',
                'pipeline':pipeline,'video':{'width':8,'height':4,'frames':2 if variant=='ld' else 9,'packets':2},
                'files':{'video.bin':'clip.bin'},'artifacts':{'video.bin':{'size':len(stream),'sha256':hashlib.sha256(stream).hexdigest()}},
                'validation':{'mode':'artifact-hashes','decode_performed':False}}
            manifest=folder/'clip.uf.json';manifest.write_text(json.dumps(metadata))
            output=check(runtime+' '+variant+' inspect',['inspect',str(manifest)])
            assert json.loads(output.stdout)['pipeline']['runtime']==runtime
            check(runtime+' '+variant+' artifact/structure',['check-archive',str(payload)],marker='storage_and_structure=ok')
            check(runtime+' '+variant+' raw scan',['scan-bitstream',str(payload),'--variant',variant],marker='packets=2')
            check(runtime+' '+variant+' manifest scan',['scan-bitstream',str(manifest)],marker='integer='+str(int(runtime=='int16')))
            payload.write_bytes(stream+b'x')
            check(runtime+' '+variant+' modified artifact',['check-archive',str(manifest)],expected=1)
            payload.write_bytes(stream)
            metadata['video']['width']=10;manifest.write_text(json.dumps(metadata))
            check(runtime+' '+variant+' metadata dimensions',['check-archive',str(manifest)],expected=1)
            metadata['video']['width']=8;metadata['video']['packets']=3;manifest.write_text(json.dumps(metadata))
            check(runtime+' '+variant+' metadata packet count',['check-archive',str(manifest)],expected=1)
            metadata['video']['packets']=2;metadata['validation']['decode_performed']=True;manifest.write_text(json.dumps(metadata))
            check(runtime+' '+variant+' missing completion evidence',['check-archive',str(manifest)],expected=1)
            metadata['validation']['decode_performed']=False
            if runtime=='int16':
                metadata['pipeline']['integer']['image']='02'*32;manifest.write_text(json.dumps(metadata))
                check(runtime+' '+variant+' prepared identity mismatch',['check-archive',str(manifest)],expected=1)

    for runtime in ['fp16','int16']:
        preamble=b'' if runtime=='fp16' else b'UF16\x01\0\0\0'+bytes(64)
        stream=preamble+bytes([0])+adaptive(4)+adaptive(8)+packet()
        metadata={'format':'dcvc-uf-image','pipeline':{'runtime':runtime,'image_only':True},
            'video':{'width':8,'height':4,'frames':1,'packets':1},
            'validation':{'mode':'artifact-hashes','decode_performed':False},
            'payload_size':len(stream),'payload_sha256':hashlib.sha256(stream).hexdigest()}
        header=json.dumps(metadata).encode();path=root/(runtime+'.dcvci')
        path.write_bytes(b'UFIMAGE\x01'+struct.pack('>I',len(header))+header+stream)
        check(runtime+' intra inspect',['inspect',str(path)],marker='dcvc-uf-image')
        check(runtime+' intra check',['check-archive',str(path)],marker='frames=1')
        check(runtime+' intra scan',['scan-bitstream',str(path)],marker='intra=1')
        path.write_bytes(path.read_bytes()+b'x');check(runtime+' intra corruption',['check-archive',str(path)],expected=1)
    malformed={'empty':b'', 'unknown NAL':b'\x30','unknown SPS':packet(identity=1),
        'P first':bytes([0,4,8])+packet(kind=2),'zero entropy':bytes([0,4,8])+packet(parallel=0),
        'truncated payload':bytes([0,4,8])+packet()[:-2],'bad preamble':b'UF16\x02\0\0\0'}
    for name,data in malformed.items():
        path=root/(name.replace(' ','-')+'.bin');path.write_bytes(data)
        check('reject '+name,['scan-bitstream',str(path)],expected=1)
print(str(passed)+' archive CLI checks passed; no models/checkpoints loaded',flush=True)
