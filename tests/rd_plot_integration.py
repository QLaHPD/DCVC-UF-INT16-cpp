#!/usr/bin/env python3
"""Native Cairo RD plots from synthetic curves; no models or checkpoints."""
import argparse
import binascii
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import zlib

parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--evidence-dir',type=Path)
args=parser.parse_args();binary=str(args.binary.resolve());passed=0

def png(path,size):
    data=path.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n'
    position=8;compressed=bytearray();density=None;header=None;ended=False
    while position<len(data):
        length=struct.unpack_from('>I',data,position)[0];kind=data[position+4:position+8]
        value=data[position+8:position+8+length]
        assert len(value)==length and position+length+12<=len(data),'truncated plot PNG'
        assert binascii.crc32(kind+value)&0xffffffff==struct.unpack_from('>I',data,position+8+length)[0],'plot PNG CRC'
        if kind==b'IHDR':header=struct.unpack('>IIBBBBB',value)
        if kind==b'pHYs':density=struct.unpack('>IIB',value)
        if kind==b'IDAT':compressed.extend(value)
        position+=length+12
        if kind==b'IEND':ended=True;break
    assert ended and position==len(data) and header[:2]==size,(path,header,size)
    width,height,depth,color,compression,filtering,interlace=header
    assert depth==8 and color in [2,6] and (compression,filtering,interlace)==(0,0,0),header
    assert density==(11811,11811,1),density
    pixels=zlib.decompress(compressed);stride=width*(3 if color==2 else 4)+1
    assert len(pixels)==height*stride and all(pixels[row*stride] in range(5) for row in range(height))

def logs(root,count=2,labels=('anchor','candidate')):
    root.mkdir();paths=[]
    for method,factor in zip(labels,[1.0,0.9]):
        data={}
        for dataset in range(count):
            points={}
            for index,(rate,quality) in enumerate(zip([0.1,0.2,0.4,0.8],[30.0,32.0,34.0,36.0])):
                record={'i_frame_num':1,'p_frame_num':7}
                for kind in ['i','p','all']:
                    record['ave_'+kind+'_frame_bpp']=rate*factor
                    record['ave_'+kind+'_frame_psnr']=quality+dataset/4
                points[str(index)]=record
            data['Class'+str(dataset)]={'Sequence'+str(dataset):points}
        path=root/(method+'.json');path.write_text(json.dumps(data));paths+=[method,str(path)]
    return labels[0],paths

def run(name,root,comparison='class',scheme='combined',count=2,enabled=True,labels=('anchor','candidate'),failure=False):
    global passed
    anchor,paths=logs(root,count,labels);output=root/'plots'
    if failure:output.write_bytes(b'not a directory')
    command=[binary,'compare-bd-rate','--base_method',anchor,'--log_paths',*paths,
        '--compare_between',comparison,'--plot_scheme',scheme,'--compare_frame_type','all',
        '--plot_rd_curve',str(int(enabled)),'--plot_path',str(output)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=45)
    if failure:
        assert result.returncode!=0 and 'Cannot create RD curve PNG' in result.stderr,(name,result)
    else:
        assert result.returncode==0 and '-10.0' in result.stdout,(name,result.returncode,result.stdout,result.stderr)
        images=sorted(output.glob('*.png'))
        expected=0 if not enabled or (scheme=='combined' and count>7) else (1 if scheme=='combined' else count)
        assert len(images)==expected,(name,images,expected)
        size=(count*1500,1200) if scheme=='combined' else (1920,1440)
        for image in images:png(image,size)
        if count>7:assert 'more than 7 datasets/sequences' in result.stderr,result.stderr
        if labels[0].startswith('_'):assert 'No artists with labels' in result.stderr,result.stderr
        if args.evidence_dir and name=='class combined':
            args.evidence_dir.mkdir(parents=True,exist_ok=True)
            for image in images:shutil.copyfile(image,args.evidence_dir/image.name)
    passed+=1;print('PASS '+name,flush=True)

with tempfile.TemporaryDirectory(prefix='uf-rd-plot-') as directory:
    root=Path(directory)
    run('class combined',root/'class-combined')
    run('class separate',root/'class-separate',scheme='separate')
    run('sequence combined',root/'sequence-combined',comparison='sequence')
    run('sequence separate',root/'sequence-separate',comparison='sequence',scheme='separate')
    run('single combined panel',root/'single',count=1)
    run('disabled plotting writes no PNG',root/'disabled',enabled=False)
    run('combined panel limit warns and skips',root/'limit',count=8)
    run('hidden legend labels warn',root/'legend',labels=('_anchor','_candidate'))
    run('output file cannot be plot directory',root/'bad-output',failure=True)
print(str(passed)+' RD plot cases passed; no models/checkpoints loaded',flush=True)
