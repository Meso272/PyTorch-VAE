import yaml
import argparse
import numpy as np
import torch
#from bitstring import BitArray
from torch.autograd import Variable
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
import time


def compress(array,error_bound):#error_bound is relative 
    rng=np.max(array)-np.min(array)
    abseb=rng*error_bound
    size=array.shape[0]
    qs=np.zeros((size,),dtype=np.int32)
    ds=np.zeros((size,),dtype=np.float32)
    for i in range(size):
        ele=array[i]
        #q=abs(ele)//abseb
        q=round(abs(ele)/(2*abseb))
        if ele<0:
            q=-q
        qs[i]=q
        #ds[i]=q*abseb
        ds[i]=q*2*abseb
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
parser.add_argument('--mode',  '-m',type=str,default="c")
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
parser.add_argument('--split',  '-sp',type=int,
                   default=0)
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
parser.add_argument('--transpose','-t',type=int,default=1)
parser.add_argument('--gpu','-gpu',type=int,
                   default=1)
#parser.add_argument('--lossmode','-lm',type=int,default=1)
parser.add_argument('--max','-mx',type=float,
                   default=1)
parser.add_argument('--min','-mi',type=float,
                   default=0)
parser.add_argument('--epsilon',  '-eps',type=float,default=-1)
parser.add_argument('--parallel',  '-para',type=int,default=0)
parser.add_argument('--time',  '-tm',type=int,default=0)
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
    if args.parallel:
        test=torch.nn.DataParallel(test)
        test=test.module

xsize=args.xsize
ysize=args.ysize
zsize=args.zsize
blocksize=args.blocksize
dim=args.dimension
error_bound=args.error
eps=args.epsilon
global_max=args.max
global_min=args.min
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


picts=[]

if eps>0:
    idxlist=[]
    meanlist=[]

idx=0

totaltime=0



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
        for y in range(0,ysize,blocksize):
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
start=time.clock()
with torch.no_grad():
    #try:
    if eps<=0:
        input_data=picts
        
    else:
        input_data=picts[idxlist]
    if error_bound<=0:
        outputs=test(torch.from_numpy(input_data).to(device) )
        totaltime+=time.clock()-start
        zs=outputs[2].cpu().detach().numpy()
        predict=outputs[0].cpu().detach().numpy()
    else:
        outputs=test.encode(torch.from_numpy(input_data).to(device) )
        totaltime+=time.clock()-start
        zs=outputs.cpu().detach().numpy()


latent_size=zs.shape[1]
zs=zs.flatten()



print(zs.shape[0])

recon=np.zeros(array_size,dtype=np.float32)

latents=np.array(zs)
if args.transpose:
    latents=latents.reshape((-1,latent_size)).transpose().flatten()

if args.gpu:
    torch.cuda.empty_cache()

if error_bound>0:
    #start=time.clock()
    
    ql,dl=compress(latents,error_bound)
    if args.transpose:
        dl=dl.reshape((latent_size,-1)).transpose()
    else:
        dl=dl.reshape((-1,latent_size))
    #totaltime+=time.clock()-start
    with torch.no_grad():
    
        if args.gpu:
            if args.split==0:
                predict=test.decode(torch.from_numpy(dl).to(device)).cpu().detach().numpy()
            else:
                split=args.split
                len_dl=dl.shape[0]
                start=0
                predict=None
                while(start<len_dl):
                    end=min(len_dl,start+split)
                    dl_split=dl[start:end]
                    predict_s=test.decode(torch.from_numpy(dl_split).to(device)).cpu().detach().numpy()
                    if start==0:
                        predict=predict_s
                    else:
                        predict=np.concatenate((predict,predict_s))
                    torch.cuda.empty_cache()
                    start+=split

        else:
            
            predict=test.decode(torch.from_numpy(dl)).detach().numpy()



predict=(predict+1)/2
predict=predict*(global_max-global_min)+global_min
if eps>0:

    predict_temp=np.zeros((predict.shape[0]+len(meanlist),1)+block_size,dtype=np.float32)
    for i in range(predict.shape[0]):
        predict_temp[idxlist[i]][0]=predict[i][0]
    for idx,mean in meanlist:
        predict_temp[idx][0]=np.full(block_size,fill_value=mean,dtype=np.float32)
    predict=predict_temp

idx=0

#start=time.clock()
if dim==3:
    for x in range(0,xsize,blocksize):
        for y in range(0,ysize,blocksize):
            for z in range(0,zsize,blocksize):
                endx=min(x+blocksize,xsize)
                endy=min(y+blocksize,ysize)
                endz=min(z+blocksize,zsize)
                recon[x:endx,y:endy,z:endz]=predict[idx][0][:endx-x,:endy-y,:endz-z]
                
                idx+=1
else:
    for x in range(0,xsize,blocksize):
        for y in range(0,ysize,blocksize):
            endx=min(x+blocksize,xsize)
            endy=min(y+blocksize,ysize)

            recon[x:endx,y:endy]=predict[idx][0][:endx-x,:endy-y]
            
            idx+=1

#totaltime+=time.clock()-start
if args.time:
    print(totaltime)

recon.tofile(args.recon)
if args.latents!=None:
    latents.tofile(args.latents)
if error_bound>0:


    dl.tofile(args.latents+".decompress")
    ql.tofile(args.latents+".q")

if args.padding:
    padded_array=np.pad(array,((1,0),))
    padded_array.tofile(args.input+".padded")
