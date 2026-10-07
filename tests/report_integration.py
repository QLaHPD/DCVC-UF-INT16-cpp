#!/usr/bin/env python3
"""Exercise report commands with synthetic rate points; never touch models."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,required=True)
args=parser.parse_args()
with tempfile.TemporaryDirectory(prefix='uf-report-test-') as temporary:
    root=Path(temporary)
    for name,factor in [('anchor',1.0),('candidate',0.9)]:
        log={'ClassB':{'Sequence':{}}}
        for index,(rate,quality) in enumerate(zip([0.1,0.2,0.4,0.8],[30.0,32.0,34.0,36.0])):
            record={'i_frame_num':1,'p_frame_num':7}
            for kind in ['i','p','all']:
                record['ave_'+kind+'_frame_bpp']=rate*factor
                record['ave_'+kind+'_frame_psnr']=quality
            log['ClassB']['Sequence'][str(index)]=record
        (root/(name+'.json')).write_text(json.dumps(log))
    base=[str(args.binary.resolve()),'compare-bd-rate','--base_method','anchor',
          '--log_paths','anchor',str(root/'anchor.json'),'candidate',str(root/'candidate.json'),
          '--plot_rd_curve','0']
    for comparison in ['class','sequence']:
        for extension in ['txt','csv']:
            output=root/(comparison+'.'+extension)
            result=subprocess.run(base+['--compare_between',comparison,'--output_path',str(output)],capture_output=True,text=True,timeout=45)
            assert result.returncode==0,(result.returncode,result.stdout,result.stderr)
            text=output.read_text() if extension=='txt' else result.stdout
            assert 'candidate' in text and '-10.0' in text,text
            if extension=='csv':
                assert not output.exists(),'Python-compatible non-TXT output should remain on stdout'
            print('PASS comparison '+comparison+' '+extension,flush=True)
    missing=subprocess.run([str(args.binary.resolve()),'compare-bd-rate','--base_method','missing','--log_paths','anchor',str(root/'anchor.json'),'--plot_rd_curve','0'],capture_output=True,text=True,timeout=45)
    assert missing.returncode!=0,'Missing anchor silently accepted'
    print('PASS missing anchor diagnostic',flush=True)
print('5/5 report integration cases passed; no models/checkpoints loaded',flush=True)
