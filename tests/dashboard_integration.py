#!/usr/bin/env python3
"""Owned PTY manager interactions without deletion approval or model loading."""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import tempfile
import termios
import time

parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,required=True)
args=parser.parse_args()
binary=str(args.binary.resolve())
passed=0

def fixture(root,channels=1,metadata=True):
    output=root/'output';sources=[]
    for index in range(channels):
        channel=output/('channel-'+str(index));channel.mkdir(parents=True)
        source_root=root/'sources'/channel.name;source_root.mkdir(parents=True)
        source=source_root/'original.mp4';source.write_bytes(b'synthetic dashboard original')
        if metadata:source.with_suffix('.info.json').write_text('{}')
        payload=channel/'clip.bin';payload.write_bytes(b'opaque dashboard payload')
        data={'format':'dcvc-uf-archive','version':1,'layout':'flat','archive_stem':'clip',
            'pipeline':{'runtime':'fp16','variant':'ld'},
            'source':{'path':str(source),'relative_path':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'media':{'audio':False}},
            'video':{'width':8,'height':4,'frames':1,'packets':1},
            'files':{'video.bin':'clip.bin'},'artifacts':{'video.bin':{'size':payload.stat().st_size,'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}},
            'validation':{'mode':'artifact-hashes','decode_performed':False}}
        (channel/'clip.uf.json').write_text(json.dumps(data));sources.append(source)
    return output,sources

class Session:
    def __init__(self,root,channels=1,metadata=True):
        self.output,self.sources=fixture(root,channels,metadata)
        self.originals={path:path.read_bytes() for path in self.sources}
        self.log=root/'events.jsonl';self.screen=bytearray()
        self.master,self.slave=os.openpty()
        fcntl.ioctl(self.slave,termios.TIOCSWINSZ,struct.pack('HHHH',24,200,0,0))
        self.before=termios.tcgetattr(self.slave)
        env=os.environ.copy();env['TERM']='xterm-256color';env['UF_PROGRESS_LOG']=str(self.log);env.pop('UF_JOB_KEY',None)
        self.process=subprocess.Popen([binary,'manage','--output_root',str(self.output),'--ui','tui'],
            stdin=self.slave,stdout=self.slave,stderr=self.slave,env=env,close_fds=True,start_new_session=True)

    def pump(self,seconds=0.05):
        if select.select([self.master],[],[],seconds)[0]:
            try:self.screen.extend(os.read(self.master,65536))
            except OSError as error:
                if error.errno!=errno.EIO:raise

    def events(self):
        if not self.log.exists():return []
        return [json.loads(line) for line in self.log.read_text().splitlines(keepends=True) if line.endswith('\n')]

    def wait(self,predicate,label,timeout=8):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.pump()
            if predicate():return
            if self.process.poll() is not None:raise AssertionError((label,'unexpected exit',self.process.returncode,bytes(self.screen[-3000:])))
        raise AssertionError((label,'timeout',self.events(),bytes(self.screen[-3000:])))

    def ready(self):
        self.wait(lambda:any(event['event']=='lifecycle_waiting' for event in self.events()),'interactive lifecycle ready')
        self.wait(lambda:b'DCVC managed encode' in self.screen,'dashboard rendered')
        assert b'\x1b[?1049h' in self.screen
        current=termios.tcgetattr(self.slave)
        assert not current[3]&(termios.ICANON|termios.ECHO),'dashboard did not enter raw mode'
        return next(event['channels'] for event in self.events() if event['event']=='lifecycle_inventory')

    def keys(self,value):
        # Confirming with y/Y could invoke model-backed verification. Never
        # send those keys or automatic-delete flags in this test.
        assert b'y' not in value and b'Y' not in value
        os.write(self.master,value)

    def finish(self,expected=0,quit=True):
        if quit:self.keys(b'x')
        self.wait(lambda:self.process.poll() is not None,'manager exit')
        for _ in range(3):self.pump(0.01)
        assert self.process.returncode==expected,(self.process.returncode,bytes(self.screen[-3000:]))
        assert termios.tcgetattr(self.slave)==self.before,'terminal settings were not restored'
        assert b'\x1b[?1049l' in self.screen,'alternate screen was not restored'
        assert all(path.exists() and path.read_bytes()==value for path,value in self.originals.items()),'original changed'
        assert not any(event['event'] in ['original_deleted','original_validation_started','channel_deletion_started'] for event in self.events()),'test entered deletion/validation workflow'
        assert not list(self.output.rglob('*.cleanup*.json'))

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        os.close(self.master);os.close(self.slave)

def run(name,root,body,**options):
    global passed
    session=Session(root,**options)
    try:body(session)
    finally:session.close()
    passed+=1;print('PASS '+name,flush=True)

with tempfile.TemporaryDirectory(prefix='uf-dashboard-pty-') as temporary:
    root=Path(temporary)
    def exit_only(session):
        session.ready();session.finish()
    run('interactive quit restores terminal and preserves originals',root/'quit',exit_only)

    def keep(session):
        rows=session.ready();channel=rows[0]['channel'];session.keys(b'ak')
        session.wait(lambda:any(event['event']=='lifecycle_inventory' and any(row['channel']==channel and row['originals']==0 and row['ineligible']==1 for row in event['channels']) for event in session.events()),'keep action reflected')
        session.finish()
    run('keep action changes inventory without deleting',root/'keep',keep)

    def cancel_prompt(session):
        session.ready();offset=len(session.screen);session.keys(b'ad')
        session.wait(lambda:b'Press y to confirm, n/Esc to cancel' in session.screen[offset:],'confirmation prompt')
        session.keys(b'n');session.finish()
    run('deletion prompt canceled with n',root/'cancel-n',cancel_prompt)

    def cancel_escape(session):
        session.ready();offset=len(session.screen);session.keys(b'd')
        session.wait(lambda:b'Press y to confirm, n/Esc to cancel' in session.screen[offset:],'confirmation prompt')
        offset=len(session.screen)
        session.keys(b'\x1b')
        # Require a complete new frame with the prompt gone before sending
        # any following key, proving the standalone Escape timeout path.
        session.wait(lambda:any(b'Ctrl-C safely stops encoding' in frame and b'Press y to confirm' not in frame
            for frame in session.screen[offset:].split(b'\x1b[H')[1:]),'standalone Escape canceled prompt')
        session.finish()
    run('standalone Escape cancels confirmation',root/'cancel-escape',cancel_escape)

    def selection(session):
        rows=session.ready();assert len(rows)==2;target=rows[1]['channel'];other=rows[0]['channel']
        session.keys(b'ca\t\x1b[D\x1b[Bk')
        session.wait(lambda:any(event['event']=='lifecycle_inventory' and
            any(row['channel']==target and row['originals']==0 for row in event['channels']) and
            any(row['channel']==other and row['originals']==1 for row in event['channels']) for event in session.events()),'arrow-selected keep target')
        session.finish()
    run('view and arrow keys keep selected channel only',root/'selection',selection,channels=2)

    def retry(session):
        rows=session.ready();assert rows[0]['blocked'];session.keys(b'ar')
        session.wait(lambda:any(event['event']=='validation_failed' and 'No completed-job evidence' in event['message'] for event in session.events()),'retry evidence refusal')
        session.finish()
    run('retry refuses missing completed-job evidence',root/'retry',retry,metadata=False)

    def refresh(session):
        session.ready();session.sources[0].unlink()
        session.wait(lambda:any(event['event']=='lifecycle_inventory' and event['channels'][0]['absent']==1 for event in session.events()),'periodic absent-source refresh',timeout=9)
        # Restore the controller-owned fixture before checking preserved data.
        session.sources[0].write_bytes(session.originals[session.sources[0]]);session.finish()
    run('periodic refresh observes fixture source disappearance',root/'refresh',refresh)

    def interrupt(session):
        session.ready();session.process.send_signal(signal.SIGINT);session.finish(expected=130,quit=False)
    run('SIGINT restores terminal without deletion',root/'interrupt',interrupt)
print(str(passed)+' dashboard PTY checks passed; no approval, models or checkpoints',flush=True)
