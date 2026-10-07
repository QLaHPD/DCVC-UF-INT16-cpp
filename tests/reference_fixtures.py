#!/usr/bin/env python3
"""Generate small independent reference fixtures without models/checkpoints."""
import argparse
import ast
import io
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import tempfile

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--reference-root',type=Path,default=root.parent/'DCVC-UF-INT16')
parser.add_argument('--output',type=Path,default=root/'build/model-free-fixtures.json')
args=parser.parse_args()
reference = args.reference_root
stream = runpy.run_path(str(reference/'src/utils/stream_helper.py'))
out = {'adaptive': [], 'random': [], 'entropy': []}
shared_config={'runtime':'int16','variant':'hts','qp':36,'skip_threshold':0.1,'tag':'пример 😀'}
out['shared_pipeline_config']=shared_config
out['shared_pipeline_id']=hashlib.sha256(json.dumps(shared_config,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for value in [0, 1, 126, 127, 128, 255, 16383, 16384, 65535, 16777216, (1<<30)-1]:
    buffer = io.BytesIO(); length = stream['write_uint_adaptive'](buffer, value)
    out['adaptive'].append(dict(value=value, bytes=list(buffer.getvalue()), length=length))
for seed in [0, 1, -42, 20260912, (1<<128)+987654321]:
    rng = random.Random(seed); draws=[rng.random() for _ in range(12)]
    rng=random.Random(seed); bounds=[1, 2, 3, 17, 2**31, 2**48+17]; choices=[rng.randrange(x) for x in bounds]
    rng=random.Random(seed); shuffled=list(range(19)); rng.shuffle(shuffled)
    rng=random.Random(seed); gaussian=[rng.gauss(1.25, 0.75) for _ in range(8)]
    out['random'].append(dict(seed=str(seed), draws=draws, bounds=bounds, choices=choices, shuffled=shuffled, gaussian=gaussian))
buffer=io.BytesIO();stream['write_sps'](buffer,dict(sps_id=3,height=128,width=16384));stream['write_ip'](buffer,True,3,63,1,0,bytes(range(17)));stream['write_ip'](buffer,False,3,0,1,1,bytes([255,0,128]))
out['packet_bytes']=list(buffer.getvalue())
# Only dependency primitives are imported. No UF model module,
# model object, checkpoint/prepared artifact or network is accessed.
import numpy as np
from MLCodec_extensions_cpp import RansEncoder, RansDecoder
cdf=np.array([[0,4096,16384,32768,49152,65536],[0,8192,24576,36864,61440,65536]],dtype=np.int32)
lengths=np.array([6,6],dtype=np.int32)
rng=random.Random(888)
for size in [0,1,17,256,1024]:
    z=np.array([rng.randint(-128,127) for _ in range(size)],dtype=np.int8)
    y=np.array([rng.randint(-128,127) for _ in range(size+7)],dtype=np.int16)
    indexes=np.array([i%2 for i in range(len(y))],dtype=np.uint8)
    packed=((y<<8)|indexes.astype(np.int16)).astype(np.int16)
    enc=RansEncoder();enc.set_entropy_coder_parallel(1)
    enc.set_cdf(cdf,lengths,0);enc.set_cdf(cdf,lengths,1);enc.reset()
    enc.encode_y(packed);enc.encode_z(z,0,2);enc.flush();bits=enc.get_encoded_stream().copy()
    dec=RansDecoder();dec.set_entropy_coder_parallel(1);dec.set_cdf(cdf,lengths,0);dec.set_cdf(cdf,lengths,1);dec.set_stream(bits)
    if size:
        dec.decode_z(size,0,2);actual_z=dec.get_decoded_tensor().copy();assert actual_z.tolist()==z.tolist()
    dec.decode_y(indexes);actual_y=dec.get_decoded_tensor().copy();assert actual_y.tolist()==y.tolist()
    out['entropy'].append(dict(cdf=cdf.reshape(-1).tolist(),lengths=lengths.tolist(),z=z.tolist(),y=y.tolist(),indexes=indexes.tolist(),packed=packed.tolist(),bytes=bits.tolist()))
# Three scalar parameters exercise the installed Python optimizer, not a model.
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
parameter=torch.tensor([1.0,-2.0,0.5],requires_grad=True)
optimizer=torch.optim.AdamW([parameter],lr=0.01,betas=(0.8,0.95),weight_decay=0.1,amsgrad=True,foreach=False)
out['adamw']=[]
for gradient in [[0.25,-0.5,1.0],[0.0,0.75,-0.25],[-1.0,0.0,0.5]]:
    parameter.grad=torch.tensor(gradient)
    optimizer.step()
    state=optimizer.state[parameter]
    out['adamw'].append(dict(gradient=gradient,parameter=parameter.detach().tolist(),exp_avg=state['exp_avg'].tolist(),exp_avg_sq=state['exp_avg_sq'].tolist(),step=state['step'].item()))
# The UF callback policy is non-reentrant with no RNG preservation. Exercise
# only scalar functions through the installed dependency, never UF modules.
from torch.utils.checkpoint import checkpoint
out['activation_recompute']=[]
for nested in [False,True]:
    x=torch.tensor([-0.75,0.25,1.5],requires_grad=True)
    weight=torch.tensor([0.5,-1.0,2.0],requires_grad=True)
    calls={'outer':0,'inner':0}
    def inner(a,b):
        calls['inner']+=1
        return a*b
    def outer(a,b):
        calls['outer']+=1
        product=checkpoint(inner,a,b,use_reentrant=False,preserve_rng_state=False) if nested else a*b
        return product.sin(),product.square()
    outputs=checkpoint(outer,x,weight,use_reentrant=False,preserve_rng_state=False)
    loss=(outputs[0]+0.125*outputs[1]).sum()
    loss.backward(retain_graph=True)
    first=dict(x_gradient=x.grad.tolist(),weight_gradient=weight.grad.tolist(),calls=dict(calls))
    x.grad=None;weight.grad=None
    loss.backward()
    out['activation_recompute'].append(dict(nested=nested,outputs=[y.detach().tolist() for y in outputs],first=first,
        second=dict(x_gradient=x.grad.tolist(),weight_gradient=weight.grad.tolist(),calls=dict(calls))))
torch.manual_seed(314159)
x=torch.tensor([1.0,2.0,3.0],requires_grad=True)
calls=0
def stochastic(a):
    global calls
    calls+=1
    return a*torch.rand_like(a)
value=checkpoint(stochastic,x,use_reentrant=False,preserve_rng_state=False)
value.sum().backward()
out['recompute_rng']=dict(output=value.detach().tolist(),gradient=x.grad.tolist(),next_draw=torch.rand(3).tolist(),calls=calls)
x=torch.tensor([-0.75,0.25,1.5],requires_grad=True)
value=checkpoint(lambda a:a.sin().square(),x,use_reentrant=False,preserve_rng_state=False)
first=torch.autograd.grad(value.sum(),x,create_graph=True)[0]
second=torch.autograd.grad(first.sum(),x)[0]
out['recompute_higher_order']=dict(output=value.detach().tolist(),first=first.detach().tolist(),second=second.tolist())
out['recompute_autocast']=[]
for device,dtype in [('cpu',torch.bfloat16),('cuda',torch.float16)]:
    x=((torch.arange(8,dtype=torch.float32).reshape(2,4)-3)/8).to(device).requires_grad_()
    weight=((torch.arange(12,dtype=torch.float32).reshape(4,3)-5)/16).to(device).requires_grad_()
    seen=[]
    def autocast_function(a,b):
        seen.append(dict(enabled=torch.is_autocast_enabled(device),dtype=str(torch.get_autocast_dtype(device)),cache=torch.is_autocast_cache_enabled()))
        return (a@b).square()
    with torch.autocast(device_type=device,dtype=dtype,cache_enabled=True):
        value=checkpoint(autocast_function,x,weight,use_reentrant=False,preserve_rng_state=False)
    previous_dtype=torch.get_autocast_dtype(device);previous_cache=torch.is_autocast_cache_enabled()
    torch.set_autocast_dtype(device,torch.float32);torch.set_autocast_cache_enabled(False)
    try:
        value.sum().backward()
        assert not torch.is_autocast_enabled(device) and torch.get_autocast_dtype(device)==torch.float32 and not torch.is_autocast_cache_enabled()
    finally:
        torch.set_autocast_dtype(device,previous_dtype);torch.set_autocast_cache_enabled(previous_cache)
    out['recompute_autocast'].append(dict(device=device,output_dtype=str(value.dtype),output=value.detach().float().cpu().reshape(-1).tolist(),
        x_gradient=x.grad.cpu().reshape(-1).tolist(),weight_gradient=weight.grad.cpu().reshape(-1).tolist(),seen=seen))
parameter=torch.tensor([1.0,-2.0,0.5],requires_grad=True)
optimizer=torch.optim.AdamW([parameter],lr=0.01,betas=(0.8,0.95),weight_decay=0.1,amsgrad=True,fused=True)
out['adamw_fused_amp']=[]
for gradient,overflow in [([0.25,-0.5,1.0],False),([1.0,1.0,1.0],True),([-1.0,0.0,0.5],False)]:
    parameter.grad=torch.tensor(gradient)*8
    optimizer.grad_scale=torch.tensor(8.0);optimizer.found_inf=torch.tensor(float(overflow))
    optimizer.step();state=optimizer.state[parameter]
    out['adamw_fused_amp'].append(dict(gradient=gradient,overflow=overflow,parameter=parameter.detach().tolist(),
        exp_avg=state['exp_avg'].tolist(),exp_avg_sq=state['exp_avg_sq'].tolist(),max_exp_avg_sq=state['max_exp_avg_sq'].tolist(),step=state['step'].item()))
# Execute stateless reference functions only; no UF model classes are imported.
def reference_functions(path,names,namespace):
    nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace
metrics=runpy.run_path(str(reference/'src/utils/metrics.py'))
out['metric_gaussian']=[dict(size=size,sigma=sigma,values=metrics['fspecial_gauss'](size,sigma).reshape(-1).tolist()) for size,sigma in [(4,1.2),(11,1.5)]]
def metric_pair(height,width,channels=None):
    shape=(height,width) if channels is None else (channels,height,width)
    index=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)
    original=(index*37)%256
    return original,np.clip(original+(index%7)-3,0,255)
original,reconstruction=metric_pair(12,5,3)
out['metric_psnr']=[metrics['calc_psnr'](a,b) for a,b in [(original,reconstruction),(original.transpose(1,0,2),reconstruction.transpose(1,0,2)),(original[:,:,::2],reconstruction[:,:,::2])]]
out['metric_ssim']=[]
for height,width in [(12,13),(6,7),(1,1)]:
    original,reconstruction=metric_pair(height,width)
    ssim,cs=metrics['calc_ssim'](original,reconstruction)
    out['metric_ssim'].append(dict(height=height,width=width,shape=list(ssim.shape),ssim=ssim.reshape(-1).tolist(),cs=cs.reshape(-1).tolist()))
out['metric_msssim']=[dict(height=h,width=w,value=float(metrics['calc_msssim'](*metric_pair(h,w)))) for h,w in [(88,91),(176,179)]]
out['metric_msssim_rgb']=float(metrics['calc_msssim_rgb'](*metric_pair(88,91,3)))
log_function=reference_functions(reference/'src/utils/common.py',['generate_log_json'],{'np':np})['generate_log_json']
out['quality_logs']=[]
for fields in [1,4]:
    for all_intra in [False,True]:
        types=[0,0,0,0] if all_intra else [0,1,0,1]
        bits=[2**63+5,2**63+9,103,211]
        psnr=[[30+frame+component/4 for component in range(fields)] for frame in range(4)]
        ssim=[[0.9+frame/100+component/1000 for component in range(fields)] for frame in range(4)]
        result=log_function(3,91,2.5,types,bits,psnr,ssim,True,0.02,0.01)
        out['quality_logs'].append(dict(fields=fields,types=types,bits=bits,psnr=psnr,ssim=ssim,expected=result))
transforms=runpy.run_path(str(reference/'src/utils/transforms.py'))
layers=reference_functions(reference/'src/layers/layers.py',['get_mse_yuv_rgb','mse_weighted_average','mse_8frames_sum','bit_estimator_z_prob','bit_estimator_z_fwd'],{'torch':torch,'F':torch.nn.functional,'ycbcr2rgb':transforms['ycbcr2rgb']})
x=(torch.arange(144,dtype=torch.float32).reshape(2,3,4,6)%29/32).requires_grad_()
reconstruction=(x.detach().flip([3])*0.95+0.02).requires_grad_()
yuv,rgb=layers['get_mse_yuv_rgb'](x,reconstruction)
weighted=layers['mse_weighted_average'](yuv,rgb,24);weighted.sum().backward()
out['weighted_mse']=dict(yuv=yuv.detach().reshape(-1).tolist(),rgb=rgb.detach().tolist(),weighted=weighted.detach().tolist(),input_gradient=x.grad.reshape(-1).tolist(),reconstruction_gradient=reconstruction.grad.reshape(-1).tolist())
color=(torch.arange(24,dtype=torch.float32)-5).reshape(1,3,2,4)/16
out['color_transforms']=[]
for bgr in [False,True]:
    out['color_transforms'].append(dict(bgr=bgr,yuv=transforms['rgb2ycbcr'](color,bgr).reshape(-1).tolist(),
        rgb_clamped=transforms['ycbcr2rgb'](color,bgr,True).reshape(-1).tolist(),rgb_unclamped=transforms['ycbcr2rgb'](color,bgr,False).reshape(-1).tolist()))
y,uv=transforms['yuv_444_to_420'](x.detach());out['color_pool']=dict(y=y.reshape(-1).tolist(),uv=uv.reshape(-1).tolist())
out['eight_frame_mse']=layers['mse_8frames_sum']([torch.tensor([i+1.0]) for i in range(9)],[1,0.5,0.25,99]).item()
x=((torch.arange(24,dtype=torch.float32)-12)/8).reshape(1,2,3,4).requires_grad_()
h=((torch.arange(8,dtype=torch.float32)-4)/16).reshape(1,2,4).requires_grad_()
b=((torch.arange(8,dtype=torch.float32)-3)/32).reshape(1,2,4).requires_grad_()
a=((torch.arange(6,dtype=torch.float32)-2)/16).reshape(1,2,3).requires_grad_()
probability=layers['bit_estimator_z_prob'](x,h,b,a);mass=layers['bit_estimator_z_fwd'](x,h,b,a)
(probability.sum()+mass.sum()).backward()
out['bit_probability']=dict(probability=probability.detach().reshape(-1).tolist(),mass=mass.detach().reshape(-1).tolist(),gradients=[t.grad.reshape(-1).tolist() for t in [x,h,b,a]])
# Dataset classes own only filenames/settings. Execute their declarations
# without importing a UF package/model, and create tiny temporary PNG inputs.
from PIL import Image
class TraceRandom(random.Random):
    def __init__(self,seed):
        super().__init__(seed);self.bounds=[]
    def _randbelow(self,upper):
        self.bounds.append(upper)
        return super()._randbelow(upper)
def dataset_class(filename,name,rng):
    path=reference/'src/datasets'/filename
    nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef) and node.name==name]
    assert len(nodes)==1
    namespace={'__name__':'_uf_dataset_fixture','json':json,'np':np,'os':os,'random':rng,'torch':torch,'Image':Image,
        'Dataset':torch.utils.data.Dataset,'rgb2ycbcr_np':transforms['rgb2ycbcr_np']}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]
def sample_record(sample):
    return [dict(shape=list(t.shape),dtype=str(t.dtype),values=t.reshape(-1).tolist()) for t in sample]
def sampler_indices(seed,size,batch,workers,rank=-1,world=1,epoch=0):
    torch.manual_seed(seed)
    base=torch.empty((),dtype=torch.int64).random_().item()
    if rank<0:
        sampler_seed=torch.empty((),dtype=torch.int64).random_().item()
        indices=torch.randperm(size,generator=torch.Generator().manual_seed(sampler_seed)).tolist()
        local_batch=batch
    else:
        sampler=torch.utils.data.DistributedSampler(range(size),num_replicas=world,rank=rank,seed=0,shuffle=True,drop_last=False)
        sampler.set_epoch(epoch);indices=list(sampler);local_batch=batch//world
    return base,indices[:len(indices)//local_batch*local_batch],local_batch
out['training_sampler']=[]
for rank,world,epoch,batch in [(-1,1,0,2),(1,3,4,6)]:
    base,indices,local_batch=sampler_indices(2026,7,batch,2,rank,world,epoch)
    streams=[random.Random(base+i) for i in range(2)];batches=[]
    for offset in range(0,len(indices),local_batch):
        worker=offset//local_batch%2;selected=indices[offset:offset+local_batch]
        batches.append(dict(indices=selected,draws=[streams[worker].randrange(1000) for _ in selected]))
    out['training_sampler'].append(dict(seed=2026,rank=rank,world=world,epoch=epoch,batch=batch,batches=batches))
out['image_dataset']=[];out['video_dataset']=[]
with tempfile.TemporaryDirectory(prefix='uf-reference-dataset-') as directory:
    image_root=Path(directory)/'images';video_root=Path(directory)/'videos'
    image_root.mkdir();(video_root/'seq').mkdir(parents=True)
    pixels=lambda phase:((np.arange(6*8*3,dtype=np.uint32).reshape(6,8,3)*13+phase*17)%256).astype(np.uint8)
    names=['пример.png','two.png']
    for phase,name in enumerate(names):Image.fromarray(pixels(phase)).save(image_root/name)
    (image_root/'description.json').write_text(json.dumps(names))
    frame_names=[f'{i}.png' for i in range(5)]
    for phase,name in enumerate(frame_names):Image.fromarray(pixels(phase)).save(video_root/'seq'/name)
    (video_root/'description.json').write_text(json.dumps(dict(frames=frame_names,seqs=[dict(path='seq',width=8,height=6,seq_length=5)])))
    for seed,patch_h,patch_w,index in [(0,4,6,0),(5,4,6,-1),(7,8,10,0)]:
        rng=TraceRandom(seed);cls=dataset_class('image_dataset.py','ImageFolder',rng)
        sample=cls(str(image_root),patch_h,patch_w,3,[0.1,0.2,0.4])[index]
        out['image_dataset'].append(dict(seed=seed,height=patch_h,width=patch_w,index=index,bounds=rng.bounds,sample=sample_record(sample)))
    for seed,patch_h,patch_w,count,group in [(0,4,6,3,1),(5,8,10,8,2),(7,4,6,5,2)]:
        rng=TraceRandom(seed);cls=dataset_class('video_dataset.py','VideoFolder',rng)
        sample=cls(str(video_root),patch_h,patch_w,3,[0.1,0.2,0.4],count,group)[0]
        out['video_dataset'].append(dict(seed=seed,height=patch_h,width=patch_w,count=count,group=group,bounds=rng.bounds,sample=sample_record(sample)))
    base,indices,_=sampler_indices(2026,2,1,1)
    rng=TraceRandom(base);cls=dataset_class('image_dataset.py','ImageFolder',rng)
    dataset=cls(str(image_root),4,6,3,[0.1,0.2,0.4])
    out['process_image_loader']=[sample_record([t.unsqueeze(0) for t in dataset[index]]) for index in indices]
# Viewer indexing/display functions do not construct FrameSource or codecs.
from dataclasses import dataclass
path=reference/'src/archive/frame_source.py'
viewer_namespace={'__name__':'__main__','dataclass':dataclass,'np':np,
    **{name:stream[name] for name in ['NalType','read_header','read_ip_remaining','read_sps_remaining']}}
nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ['Packet','packet_index','raw_rgb']]
assert len(nodes)==3
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),viewer_namespace)
out['viewer_index']=[]
for runtime in ['fp16','int16']:
    for variant in ['ld','hts','htl']:
        buffer=io.BytesIO()
        preamble=b'' if runtime=='fp16' else b'UF16\x01\0\0\0'+bytes([7])*32+bytes([9])*32
        buffer.write(preamble)
        stream['write_sps'](buffer,dict(sps_id=0,height=4,width=8))
        stream['write_ip'](buffer,True,0,10,1,0,b'a');stream['write_ip'](buffer,False,0,11,1,1,b'bb')
        stream['write_sps'](buffer,dict(sps_id=1,height=4,width=8))
        stream['write_ip'](buffer,True,1,12,1,0,b'ccc');stream['write_ip'](buffer,False,1,13,1,1,b'dddd')
        frames=4 if variant=='ld' else 13
        reader=io.BufferedReader(io.BytesIO(buffer.getvalue()));reader.seek(len(preamble))
        packets=viewer_namespace['packet_index'](reader,dict(width=8,height=4,frames=frames,packets=4),1 if variant=='ld' else 8)
        out['viewer_index'].append(dict(runtime=runtime,variant=variant,frames=frames,bytes=list(buffer.getvalue()),
            packets=[dict(offset=p.offset,first=p.first,count=p.count,nal_type=int(p.nal_type),sps=p.sps,qp=p.qp) for p in packets]))
out['viewer_display']=[]
for width,height in [(8,4),(4,6)]:
    raw=((np.arange(width*height*3//2,dtype=np.uint32)*29+17)%256).astype(np.uint8).tobytes()
    rgb=viewer_namespace['raw_rgb'](raw,width,height)
    for crop_width,crop_height in [(width,height),(3,3),(0,height)]:
        out['viewer_display'].append(dict(width=width,height=height,crop_width=crop_width,crop_height=crop_height,
            raw=list(raw),rgb=rgb[:crop_height,:crop_width,:].reshape(-1).tolist()))
out['viewer_crop']=[dict(stop=json.dumps(stop),expected=len(range(17)[:stop])) for stop in [None,False,True,0,5,17,999,-1,-5,-17,-999,2**200,-2**200]]
out['viewer_images']=[]
for runtime in ['fp16','int16']:
    buffer=io.BytesIO()
    if runtime=='int16':buffer.write(b'UF16\x01\0\0\0'+bytes([7])*32+bytes(32))
    stream['write_sps'](buffer,dict(sps_id=0,height=4,width=8));stream['write_ip'](buffer,True,0,20,1,0,b'opaque_image')
    payload=buffer.getvalue();pipeline={'runtime':runtime,'variant':'image','image_only':True,'tag':'galería'}
    if runtime=='int16':pipeline['integer']={'image':'07'*32}
    metadata={'format':'dcvc-uf-image','pipeline':pipeline,'video':{'width':8,'height':4,'frames':1,'packets':1},
        'image_width':3,'image_height':-1,'payload_size':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),
        'validation':{'mode':'artifact-hashes','decode_performed':False}}
    out['viewer_images'].append(dict(runtime=runtime,metadata_json=json.dumps(metadata),bytes=list(payload),pipeline_repr=repr(pipeline)))
# Small raster fixtures for the native image reader and its current fallbacks.
from PIL import ImageOps
image_root=args.output.parent/(args.output.stem+'-images');image_root.mkdir(parents=True,exist_ok=True)
rgb=Image.fromarray(((np.arange(144,dtype=np.uint32).reshape(6,8,3)*13+17)%256).astype(np.uint8))
gray=Image.fromarray(((np.arange(48,dtype=np.uint32).reshape(6,8)*17)%256).astype(np.uint8))
rgba=rgb.convert('RGBA');rgba.putalpha(gray)
palette=Image.fromarray((np.arange(48,dtype=np.uint8).reshape(6,8)%16),'P')
palette.putpalette([value for i in range(256) for value in [(i*17)%256,(i*31)%256,(i*43)%256]])
palette.info['transparency']=bytes([i*17 for i in range(16)])
wide=Image.fromarray((np.arange(48,dtype=np.uint16).reshape(6,8)*1300))
out['raster_images']=[]
def raster(name,image,**options):
    path=image_root/name;image.save(path,**options)
    expected=[]
    for oriented in [False,True]:
        with Image.open(path) as opened:
            value=(ImageOps.exif_transpose(opened) if oriented else opened).convert('RGB')
            expected.append(dict(oriented=oriented,width=value.width,height=value.height,rgb=np.asarray(value).reshape(-1).tolist()))
    out['raster_images'].append(dict(name=name,path=str(path.resolve()),expected=expected))
raster('rgb.png',rgb);raster('rgba.png',rgba);raster('gray.png',gray);raster('palette.png',palette);raster('gray16.png',wide)
opaque_palette=palette.copy();opaque_palette.info.pop('transparency',None)
raster('palette-opaque.png',opaque_palette)
raster('rgb-trns.png',rgb,transparency=rgb.getpixel((0,0)))
raster('bilevel-trns.png',gray.convert('1'),transparency=0)
raster('rgb.jpg',rgb,quality=91);raster('gray.jpg',gray,quality=91);raster('cmyk.jpg',rgb.convert('CMYK'),quality=91)
raster('rgb.bmp',rgb);raster('palette.bmp',palette)
raster('rgba.webp',rgba,lossless=True);raster('rgb.webp',rgb,quality=87)
raster('rgb.tiff',rgb,compression='raw');raster('gray.tiff',gray,compression='raw')
for name,image,orientation,options in [('oriented.jpg',rgb,6,{'quality':91}),('oriented.png',rgba,8,{}),
    ('oriented.webp',rgb,3,{'lossless':True}),('oriented.tiff',rgb,6,{'compression':'raw'})]:
    exif=Image.Exif();exif[274]=orientation;raster(name,image,exif=exif,**options)
raster('rgba-lzw.tiff',rgba,compression='tiff_lzw')
raster('palette-lzw.tiff',opaque_palette,compression='tiff_lzw')
raster('gray16-deflate.tiff',wide,compression='tiff_adobe_deflate')
raster('float32.tiff',Image.fromarray((np.arange(48,dtype=np.float32).reshape(6,8)-12)*17),compression='raw')
raster('signed32.tiff',Image.fromarray((np.arange(48,dtype=np.int32).reshape(6,8)-12)*17),compression='raw')
raster('bilevel-group4.tiff',gray.convert('1'),compression='group4')
raster('rgb444.avif',rgb,quality=95,subsampling='4:4:4',speed=8,max_threads=1)
raster('rgb420.avif',rgb,quality=95,subsampling='4:2:0',speed=8,max_threads=1)
raster('rgba444.avif',rgba,quality=95,subsampling='4:4:4',speed=8,max_threads=1)
raster('gray.avif',gray,quality=95,speed=8,max_threads=1)
exif=Image.Exif();exif[274]=6
raster('oriented.avif',rgb,quality=95,subsampling='4:4:4',speed=8,max_threads=1,exif=exif)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(out,indent=2)+'\n')
print('Generated Python stream/RNG and UF rANS reference fixtures; no models/checkpoints loaded',flush=True)
