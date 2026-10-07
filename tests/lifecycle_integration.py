#!/usr/bin/env python3
"""Temporary inventory/cleanup previews. Never authorize deletion or decoding."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace

parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--reference-root',type=Path,default=Path(__file__).resolve().parents[2]/'DCVC-UF-INT16')
args=parser.parse_args()
binary=str(args.binary.resolve())
reference=args.reference_root
passed=0
errors=[]
namespace={'Path':Path,'json':json,'hashlib':hashlib,'struct':struct,'MAGIC':b'UFIMAGE\x01',
    'event':lambda kind,**fields:errors.append((kind,fields))}

class StatelessImports(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module in ['src.archive.thumbnails','src.archive.media']:
            return None
        return node

def functions(path,names):
    nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==set(names)
    module=StatelessImports().visit(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)

functions(reference/'src/archive/storage.py',['has_completion_record'])
functions(reference/'src/archive/thumbnails.py',['read_image'])
functions(reference/'src/archive/lifecycle.py',['inventory','channel_blockers'])
media=ast.parse((reference/'src/archive/media.py').read_text())
namespace['VIDEO_EXTENSIONS']=next(ast.literal_eval(node.value) for node in media.body
    if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='VIDEO_EXTENSIONS' for target in node.targets))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fixture(folder,stem='clip',kind='video',source_name=None,present=True,info=True):
    output=folder/'output';channel=output/'channel';sources=folder/'sources'
    channel.mkdir(parents=True,exist_ok=True);sources.mkdir(exist_ok=True)
    source=sources/(source_name or (stem+'.jpg' if kind=='image' else stem+'.mp4'))
    if present:source.write_bytes(('synthetic original '+stem).encode())
    if info and kind=='video' and present:source.with_suffix('.info.json').write_text('{}')
    data={'format':'dcvc-uf-image' if kind=='image' else 'dcvc-uf-archive','version':1,
        'pipeline':{'runtime':'fp16',**({'image_only':True} if kind=='image' else {'variant':'ld'})},
        'source':{'path':str(source),'relative_path':source.name,'sha256':digest(source) if present else '00'*32,'media':{'audio':False}},
        'video':{'width':8,'height':4,'frames':1,'packets':1},
        'validation':{'mode':'artifact-hashes','decode_performed':False}}
    payload=b'opaque lifecycle fixture'
    if kind=='image':
        archive=channel/(stem+'.dcvci');artifact=None
        data.update(payload_size=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest())
    else:
        archive=channel/(stem+'.uf.json');artifact=channel/(stem+'.bin');artifact.write_bytes(payload)
        data.update(layout='flat',archive_stem=stem,files={'video.bin':artifact.name},
            artifacts={'video.bin':{'size':len(payload),'sha256':digest(artifact)}})
    item=SimpleNamespace(output=output,channel=channel,sources=sources,source=source,archive=archive,artifact=artifact,data=data,payload=payload)
    write(item);return item

def write(item):
    header=json.dumps(item.data).encode()
    item.archive.write_bytes(b'UFIMAGE\x01'+struct.pack('>I',len(header))+header+item.payload if item.artifact is None else header)

def check(name,command,expected=0,marker=None):
    global passed
    assert '--yes' not in command and '--auto-delete' not in command
    log=logs/(str(passed)+'.jsonl');env=os.environ.copy();env['UF_PROGRESS_LOG']=str(log);env.pop('UF_JOB_KEY',None)
    result=subprocess.run([binary,*map(str,command)],capture_output=True,text=True,timeout=30,env=env)
    assert (result.returncode==0 if expected==0 else result.returncode!=0),(name,result.returncode,result.stdout,result.stderr)
    if marker:assert marker in result.stdout+result.stderr,(name,result.stdout,result.stderr)
    events=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    passed+=1;print('PASS '+name,flush=True)
    return events

def preview(name,item,extra=(),expected=0,marker='cleanup_preview'):
    before=item.source.read_bytes() if item.source.exists() else None
    events=check(name,['cleanup','--input',item.archive,*extra],expected,marker if expected else None)
    if expected==0 and marker:assert any(row['event']==marker for row in events),(name,events)
    assert (item.source.read_bytes() if item.source.exists() else None)==before,name+' altered original'
    assert not list(item.channel.glob('*.cleanup*.json')),name+' wrote deletion records'
    return events

def canonical(rows):
    result=[]
    for row in rows:
        entries=[]
        for item in row['archives']:
            fields=['input','source','eligible','remote','exists','kind']
            if item['kind']=='video':fields.append('metadata')
            entries.append({field:item[field] for field in fields})
        result.append({**{field:row[field] for field in ['channel','originals','bytes','streamed','absent','ineligible']},
            'archives':sorted(entries,key=lambda item:(item['kind'],item['input'],item['source'])),
            'blockers':sorted(row['blockers']),'blocked':row['blocked']})
    return sorted(result,key=lambda row:row['channel'])

def manage(name,item,extra=(),channels=None,allow_missing=False,base=None,dry=False):
    command=['manage','--output_root',item.output,'--ui','plain','--cleanup-dry-run' if dry else '--keep-originals',*extra]
    if channels:command+=['--channel_ids',*channels]
    if allow_missing:command+=['--allow-missing-metadata']
    if base:command+=['--base_root',base]
    events=check(name,command)
    errors.clear();roots=[item.output/name for name in channels] if channels else [item.output]
    rows=list(namespace['inventory'](roots,base_root=base).values())
    for row in rows:
        row['blockers']=namespace['channel_blockers'](row,SimpleNamespace(allow_missing_metadata=allow_missing))
        row['blocked']=bool(row['blockers'])
    actual=[event['channels'] for event in events if event['event']=='lifecycle_inventory']
    assert actual and all(canonical(value)==canonical(rows) for value in actual),(name,actual,rows)
    assert sum(event['event']=='inventory_error' for event in events)==len(errors)*(2 if dry else 1),(name,events,errors)
    assert not any(event['event']=='original_deleted' for event in events),name+' deleted an original'
    return events,rows

with tempfile.TemporaryDirectory(prefix='uf-lifecycle-cli-') as temporary:
    root=Path(temporary);logs=root/'logs';logs.mkdir()
    item=fixture(root/'preview');preview('video cleanup preview preserves source',item)
    item=fixture(root/'absent',present=False);preview('absent original returns normally',item,marker='original_already_absent')
    item=fixture(root/'changed');item.source.write_bytes(b'changed original')
    preview('changed source refuses cleanup',item,expected=1,marker='Original differs')
    item=fixture(root/'symlink');target=item.sources/'target.bin';target.write_bytes(item.source.read_bytes());item.source.unlink();item.source.symlink_to(target)
    preview('symlink original refuses cleanup',item,expected=1,marker='Original differs')
    item=fixture(root/'partial');item.data['pipeline']['max_frames']=3;write(item)
    preview('partial archive refuses cleanup',item,expected=1,marker='Partial archives')
    item=fixture(root/'audio');item.data['source']['media']['audio']=True;write(item)
    preview('omitted audio refuses cleanup',item,expected=1,marker='omits source audio')
    item=fixture(root/'remote');item.data['source']['kind']='youtube';item.data['source']['url']='https://synthetic.invalid/video';write(item)
    preview('streamed source refuses cleanup',item,expected=1,marker='Streamed archives')
    item=fixture(root/'artifact');item.artifact.write_bytes(b'corrupt')
    preview('modified artifact refuses cleanup',item,expected=1,marker=None)
    item=fixture(root/'completion');item.data['validation']['decode_performed']=True;write(item)
    preview('missing completion refuses cleanup',item,expected=1,marker=None)
    item=fixture(root/'remap');item.data['source']['path']=str(root/'nonexistent-original');write(item)
    preview('base root remaps source for preview',item,['--base_root',item.sources])
    item=fixture(root/'escape');item.data['source']['relative_path']='../outside.mp4';write(item)
    preview('base root escape refuses cleanup',item,['--base_root',item.sources],expected=1,marker='escapes base_root')
    for mode in ['preview','changed','symlink','absent','artifact','escape']:
        item=fixture(root/('image-'+mode),kind='image',present=mode!='absent')
        if mode=='changed':item.source.write_bytes(b'changed image')
        if mode=='symlink':
            target=item.sources/'target.bin';target.write_bytes(item.source.read_bytes());item.source.unlink();item.source.symlink_to(target)
        if mode=='artifact':item.archive.write_bytes(item.archive.read_bytes()+b'corrupt')
        if mode=='escape':item.data['source']['relative_path']='../outside.jpg';write(item)
        extra=['--base_root',item.sources] if mode=='escape' else []
        preview('image cleanup '+mode,item,extra,expected=0 if mode in ['preview','absent'] else 1,
            marker='cleanup_preview' if mode=='preview' else None)

    item=fixture(root/'inventory');fixture(root/'inventory',stem='absent',present=False)
    remote=fixture(root/'inventory',stem='remote',present=False);remote.data['source'].update(kind='youtube',url='https://synthetic.invalid/video');write(remote)
    partial=fixture(root/'inventory',stem='partial');partial.data['pipeline']['max_frames']=3;write(partial)
    image=fixture(root/'inventory',stem='thumbnail',kind='image')
    bundle=item.channel/'legacy';bundle.mkdir();duplicate=dict(item.data)
    for field in ['layout','archive_stem','files']:duplicate.pop(field)
    (bundle/'manifest.json').write_text(json.dumps(duplicate))
    hidden=item.channel/'.hidden';hidden.mkdir();(hidden/'ignored.uf.json').write_text('invalid JSON')
    (item.output/'alias').symlink_to(item.channel,target_is_directory=True)
    events,rows=manage('inventory categories/dedup/hidden/symlink recursion Python parity',item)
    assert len(rows)==1 and rows[0]['originals']==2 and rows[0]['streamed']==1 and rows[0]['absent']==1 and rows[0]['ineligible']==1
    manage('inventory selected channel Python parity',item,channels=['channel'])
    events,rows=manage('inventory repeated roots preserves image repeats',item,channels=['channel','channel'])
    assert rows[0]['originals']==3
    (item.channel/'bad.uf.json').write_text('{')
    manage('inventory malformed manifest is reported and others retained',item)
    check('manager channel path escape rejected',['manage','--output_root',item.output,'--ui','plain','--keep-originals','--channel_ids','../outside'],expected=1,marker='escapes output_root')

    item=fixture(root/'missing-info',info=False)
    events,rows=manage('missing info blocks channel dry-run',item,dry=True)
    assert rows[0]['blocked'] and not any(event['event']=='cleanup_preview' for event in events)
    events,rows=manage('explicit missing-info policy permits dry preview',item,dry=True,allow_missing=True)
    assert not rows[0]['blocked'] and any(event['event']=='cleanup_preview' for event in events) and item.source.exists()
    finished=[event for event in events if event['event']=='channel_deletion_finished']
    assert len(finished)==1 and finished[0]['dry_run'] and finished[0]['failed']==0
    item=fixture(root/'blockers');(item.sources/'unprocessed.mp4').write_bytes(b'fixture');(item.sources/'orphan.info.json').write_text('{}')
    (item.sources/'.hidden.mp4').write_bytes(b'fixture');(item.sources/'ignored.uf.info.json').write_text('{}')
    events,rows=manage('unprocessed video and metadata-only blockers Python parity',item,dry=True)
    assert len(rows[0]['blockers'])==2 and not any(event['event']=='cleanup_preview' for event in events)
    item=fixture(root/'inventory-remap');item.data['source']['path']=str(root/'gone-source');write(item)
    manage('inventory base remapping Python parity',item,base=item.sources)
    item.data['source']['relative_path']='../outside.mp4';write(item)
    events,rows=manage('inventory base escape reports error without channel',item,base=item.sources)
    assert not rows
print(str(passed)+' lifecycle CLI checks passed; no deletion, models or checkpoints',flush=True)
