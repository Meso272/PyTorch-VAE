import yaml
import argparse
import numpy as np
import torch
#from bitstring import BitArray
from torch.autograd import Variable
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn


def compress(array,error_bound):#error_bound is relative 
    rng=np.max(array)-np.min(array)
    abseb=rng*error_bound
    size=array.shape[0]
    qs=np.zeros((size,),dtype=np.int32)
    ds=np.zeros((size,),dtype=np.float32)
    for i in range(size):
        ele=array[i]
        q=abs(ele)//abseb
        if ele<0:
            q=-q
        qs[i]=q
        ds[i]=q*abseb
    qs=qs-np.min(qs)
    return qs,ds







parser = argparse.ArgumentParser(description='NN predictor')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--ckpt',  '-k',type=str
                   )
#parser.add_argument('--mode',  '-m',type=str,default="c")
#parser.add_argument('--lsize','-ls',type=int,default=0)
parser.add_argument('--input',  '-i',type=str
                   )
parser.add_argument('--dimension',  '-d',type=int,default=2
                   )
parser.add_argument('--error',  '-e',type=float,default=0
                   )
parser.add_argument('--latents',  '-l',type=str,default=None
                   )
#parser.add_argument('--quant',  '-q',type=str,default=None)
#parser.add_argument('--unpred',  '-u',type=str,default=None)
parser.add_argument('-recon',  '-r',type=str,default=None
                   )
#parser.add_argument('--decomp',  '-d',type=str,default=None)
#parser.add_argument('--bits',  '-b',type=int,default=32)
parser.add_argument('--xsize',  '-x',type=int,
                   default=449)
parser.add_argument('--ysize',  '-y',type=int,
                   default=449)
parser.add_argument('--zsize',  '-z',type=int,
                   default=235)
parser.add_argument('--blocksize','-s',type=int,
                   default=16)
parser.add_argument('--eval','-v',type=int,
                   default=0)
parser.add_argument('--padding','-p',type=int,
                   default=0)
#parser.add_argument('--transpose','-t',type=int,default=1)
parser.add_argument('--gpu','-gpu',type=int,
                   default=1)
#parser.add_argument('--lossmode','-lm',type=int,default=1)
parser.add_argument('--max','-mx',type=float,
                   default=1)
parser.add_argument('--min','-mi',type=float,
                   default=0)
parser.add_argument('--epsilon',  '-eps',type=float,default=-1)
#parser.add_argument('--singlerange','-sr',type=int,default=0)
args = parser.parse_args()





if args.gpu:
    device='cuda'
else:
    device='cpu'
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


with torch.no_grad():
    model = vae_models[config['model_params']['name']](**config['model_params'])
    test = VAEXperiment(model,config['exp_params'])
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    test.load_state_dict(checkpoint['state_dict'])
    test=test.model
    if args.gpu:
        test=test.cuda()
    if args.eval:
        test.eval()

xsize=args.xsize
ysize=args.ysize
zsize=args.zsize
blocksize=args.blocksize
dim=args.dimension
error_bound=args.error_bound
eps=args.eps
if dim==3:
    array_size=(xsize,ysize,zsize)
    block_size=(blocksize,blocksize,blocksize)
    
else:
    array_size=(xsize,ysize)
    block_size=(blocksize,blocksize)
array=np.fromfile(args.input,dtype=np.float32).reshape(array_size)
minimum=np.min(array)
maximum=np.max(array)
rng=maximum-minimum

global_max=args.max
global_min=args.min
picts=[]

if eps>0:
    idxlist=[]
    meanlist=[]

idx=0
if dim==3:
    for x in range(0,xsize,blocksize):
        for y in range(0,ysize,blocksize):
           for z in range(0,zsize,blocksize):
                endx=min(x+blocksize,xsize)
                endy=min(y+blocksize,ysize)
                endz=min(z+blocksize,zsize)
                pict=array[x:endx,y:endy,z:endz]
                padx=blocksize-pict.shape[0]
                pady=blocksize-pict.shape[1]
                padz=blocksize-pict.shape[2]

                if eps>0:
                    r=np.max(pict)-np.min(pict)
                    if r>eps*(global_max-global_min):
                    
                    
                    #picts.append(pict)
                        idxlist.append(idx)

                    else:

                        meanlist.append( (idx,np.mean(pict) ) )
           
            

                pict=(pict-global_min)/(global_max-global_min)
                pict=np.pad(pict,((0,padx),(0,pady),(0,padz)))
                    #print(array[x:x+size,y:y+size])
                pict=pict*2-1
                pict=np.expand_dims(pict,0)
                picts.append(pict)

                idx+=1
else:
    for x in range(0,xsize,blocksize):
        for y in range(0,yszie,size):
            endx=min(x+blocksize,xsize)
            endy=min(y+blocksize,ysize)
            pict=array[x:endx,y:endy]
            padx=blocksize-pict.shape[0]
            pady=blocksize-pict.shape[1]
            if eps>0:
                r=np.max(pict)-np.min(pict)
                if r>eps*(global_max-global_min):
                    
                    
                    #picts.append(pict)
                    idxlist.append(idx)

                else:

                    meanlist.append( (idx,np.mean(pict) ) )
          
            pict=(pict-global_min)/(global_max-global_min)
            pict=np.pad(pict,((0,padx),(0,pady)))
           
            pict=pict*2-1
        
            pict=np.expand_dims(pict,0)
                    #print(array[x:x+size,y:y+size])
            picts.append(pict)
            idx+=1

picts=np.array(picts)

with torch.no_grad():
    #try:
    outputs=test(torch.from_numpy(picts).to(device) )
zs=outputs[2].cpu().detach().numpy()
predict=outputs[0].cpu().detach().numpy()
latent_size=zs.shape[1]
zs=zs.flatten()



recon=np.zeros(arraysize,dtype=np.float32)
predict=(predict+1)/2
predict=predict*(global_max-global_min)+global_min


if eps>0:
    predict_temp=np.zeros((predict.shape[0]+len(meanlist),1)+block_size,dtype=np.float32)
    for i in range(predict.shape[0]):
        predict_temp[idxlist[i]][0]=predict[i][0]
    for idx,mean in meanlist:
        predict_temp[idx][0]=np.full(block_size,fill_value=mean,dtype=np.float32)
    predict=predict_temp
if error_bound>0:
    latents=np.array(zs)
    ql,dl=compress(latents,error_bound)
    dl=dl.reshape((-1,latent_size))
    with torch.no_grad():
    
        if args.gpu:
            predict2=test.decode(torch.from_numpy(dl).to(device)).cpu().detach().numpy()
        else:
            predict2=test.decode(torch.from_numpy(dl)).detach().numpy()
    predict2=(predict2+1)/2
    predict2=predict2*(global_max-global_min)+global_min
    recon2=np.zeros(array_size,dtype=np.float32)    
    if eps>0:
        predict_temp=np.zeros((predict2.shape[0]+len(meanlist),1)+block_size,dtype=np.float32)
        for i in range(predict2.shape[0]):
            predict_temp[idxlist[i]][0]=predict2[i][0]
        for idx,mean in meanlist:
            predict_temp[idx][0]=np.full(block_size,fill_value=mean,dtype=np.float32)
        predict2=predict_temp

idx=0
if dim==3:
    for x in range(0,xsize,size):
        for y in range(0,ysize,size):
            for z in range(0,zsize,size):
                endx=min(x+size,xsize)
                endy=min(y+size,ysize)
                endz=min(z+size,zsize)
                recon[x:endx,y:endy,z:endz]=predict[idx][0][:endx-x,:endy-y,:endz-z]
                if error_bound>0:
                    recon2[x:endx,y:endy,z:endz]=predict2[idx][0][:endx-x,:endy-y,:endz-z]
                idx+=1
else:
    for x in range(0,height,size):
        for y in range(0,width,size):
            endx=min(x+size,height)
            endy=min(y+size,width)

            recon[x:endx,y:endy]=predict[idx][0][:endx-x,:endy-y]
            if error_bound>0:
                recon2[x:endx,y:endy]=predict2[idx][0][:endx-x,:endy-y]
            idx+=1




recon.tofile(args.recon)
zs.tofile(args.latents)
if error_bound>0:
    recon2.tofile(args.recon+".compress")
    dl.tofile(args.latents+".compress")
if args.padding:
    padded_array=np.pad(array,((1,0),))
    padded_array.tofile(args.input+".padded")
