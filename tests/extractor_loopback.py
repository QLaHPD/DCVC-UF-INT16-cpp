#!/usr/bin/env python3
"""Real installed yt-dlp/FFmpeg with owned loopback media; no external site."""
import argparse
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading

parser=argparse.ArgumentParser();parser.add_argument('--native-tests',type=Path,required=True)
parser.add_argument('--ytdlp',type=Path,required=True);args=parser.parse_args()
passed=0
def run(command,timeout=30):
    env=os.environ.copy()
    for name in ['HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy','ALL_PROXY','all_proxy']:env.pop(name,None)
    env['NO_PROXY']='127.0.0.1';env['no_proxy']='127.0.0.1'
    return subprocess.run(list(map(str,command)),capture_output=True,timeout=timeout,env=env)
def done(name):
    global passed
    passed+=1;print('PASS '+name,flush=True)

with tempfile.TemporaryDirectory(prefix='uf-extractor-loopback-') as directory:
    root=Path(directory);source=root/'clip.mp4'
    generated=run(['ffmpeg','-v','error','-nostdin','-f','lavfi','-i','testsrc=size=8x4:rate=5','-frames:v','6',
        '-threads','1','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',source])
    assert generated.returncode==0,generated.stderr
    requests=[]
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self,*values,**keywords):super().__init__(*values,directory=str(root),**keywords)
        def log_message(self,*values):pass
        def do_HEAD(self):requests.append(('HEAD',self.path));super().do_HEAD()
        def do_GET(self):requests.append(('GET',self.path));super().do_GET()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url='http://127.0.0.1:'+str(server.server_port)+'/clip.mp4'
    try:
        version=run([args.ytdlp,'--version']);assert version.returncode==0,version.stderr
        print('Installed yt-dlp '+version.stdout.decode().strip(),flush=True)
        metadata=run([args.ytdlp,'--ignore-config','--skip-download','--dump-single-json','--',url])
        assert metadata.returncode==0,metadata.stderr
        info=json.loads(metadata.stdout);assert info.get('url')==url or any(f.get('url')==url for f in info.get('formats',[])),info
        done('real generic extractor metadata over loopback')
        target=root/'copied.mp4';result=run([args.native_tests,'_prepared_transfer',url,args.ytdlp,target])
        assert result.returncode==0,(result.stdout,result.stderr)
        expected=hashlib.sha256(source.read_bytes()).hexdigest()
        assert result.stdout.decode().strip()==expected and target.read_bytes()==source.read_bytes()
        done('native PreparedDownload delivers exact real extractor bytes')
        original=run(['ffmpeg','-v','error','-nostdin','-i',source,'-threads','1','-f','rawvideo','-pix_fmt','yuv420p','-'])
        copied=run(['ffmpeg','-v','error','-nostdin','-i',target,'-threads','1','-f','rawvideo','-pix_fmt','yuv420p','-'])
        assert original.returncode==copied.returncode==0 and original.stdout==copied.stdout and len(copied.stdout)==6*8*4*3//2
        done('FFmpeg decodes transferred clip to identical frames')
        failed=run([args.native_tests,'_prepared_transfer',url.replace('clip.mp4','missing.mp4'),args.ytdlp,root/'missing-output'])
        assert failed.returncode!=0 and b'yt-dlp stream failed:' in failed.stderr and b'404' in failed.stderr,failed.stderr
        done('real extractor 404 reaches native failure diagnostic')
        assert requests and all(path in ['/clip.mp4','/missing.mp4'] for _,path in requests),requests
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3);assert not thread.is_alive()
print(str(passed)+' loopback extractor cases passed; no external site/model/checkpoint',flush=True)
