import yaml
import argparse
import numpy as np
import torch
#from bitstring import BitArray
from torch.autograd import Variable
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from pytorch_lightning.callbacks import ModelCheckpoint
#trainer = Trainer()

def quantize(data,pred,error_bound):
    radius=32768
    diff = data - pred
    quant_index = (int) (abs(diff)/ error_bound) + 1
    #print(quant_index)
    if (quant_index < radius * 2) :
        quant_index =quant_index>> 1
        half_index = quant_index
        quant_index =quant_index<< 1
        #print(quant_index)
        quant_index_shifted=0
        if (diff < 0) :
            quant_index = -quant_index
            quant_index_shifted = radius - half_index
        else :
            quant_index_shifted = radius + half_index
        
        decompressed_data = pred + quant_index * error_bound
        #print(decompressed_data)
        if abs(decompressed_data - data) > error_bound :
            #print("b")
            return 0,data
        else:
            #print("c")
            data = decompressed_data
            return quant_index_shifted,data
        
    else:
        #print("a")
        return 0,data
    


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--ckpt',  '-k',type=str
                   )
parser.add_argument('--mode',  '-m',type=str,default="c"
                   )
parser.add_argument('--lsize','-ls',type=int,default=0)
parser.add_argument('--input',  '-i',type=str
                   )
parser.add_argument('--error',  '-e',type=float,default=1e-2
                   )
parser.add_argument('--latents',  '-l',type=str,default=None
                   )
parser.add_argument('--quant',  '-q',type=str,default=None
                   )
parser.add_argument('--unpred',  '-u',type=str,default=None
                   )
parser.add_argument('-recon',  '-r',type=str,default=None
                   )
parser.add_argument('--decomp',  '-d',type=str,default=None
                   )
parser.add_argument('--bits',  '-b',type=int,
                   default=32)
parser.add_argument('--xsize',  '-x',type=int,
                   default=449)
parser.add_argument('--ysize',  '-y',type=int,
                   default=449)
parser.add_argument('--zsize',  '-z',type=int,
                   default=235)
parser.add_argument('--size','-s',type=int,
                   default=16)
parser.add_argument('--eval','-v',type=int,
                   default=0)
parser.add_argument('--transpose','-t',type=int,
                   default=1)
parser.add_argument('--gpu','-gpu',type=int,
                   default=1)
parser.add_argument('--lossmode','-lm',type=int,
                   default=1)
parser.add_argument('--max','-mx',type=float,
                   default=0.0386)
parser.add_argument('--min','-mi',type=float,
                   default=-0.0512)
args = parser.parse_args()

global_max=args.max
global_min=args.min

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
if args.gpu:
    device='cuda'
else:
    device='cpu'

def lorenzo(array,x_start,y_start,z_start,error_bound,block_size,cross_block=True):
    x_end=min(xsize,x_start+block_size)
    y_end=min(ysize,y_start+block_size)
    z_end=min(zsize,z_start+block_size)
    #backup=np.array(array[x_start:x_end,y:start:y_end])
    #preds=np.zeros((block_size,block_size),dtype=np.float32)
    qs=[]
    us=[]
    loss=0
    decomps=np.zeros((block_size,block_size,block_size),dtype=np.float32)
    for x in range(x_start,x_end):
        for y in range(y_start,y_end):
            for z in range(z_start,z_end):
                f_011=array[x-1][y][z] if x else 0
                f_101=array[x][y-1][z] if y else 0
                f_110=array[x][y][z-1] if z else 0
                f_001=array[x-1][y-1][z] if x and y else 0
                f_100=array[x][y-1][z-1] if y and z else 0
                f_010=array[x][y-1][z] if x and z else 0
                f_000=array[x-1][y-1][z-1] if x and y and z else 0

                orig=array[x][y][z]
                pred=f_000+f_011+f_101+f_110-f_001-f_010-f_100
                if args.lossmode==1:
                    loss+=abs(orig-pred)
                else:
                    loss+=abs(orig-pred)//error_bound
                q,decomp=quantize(orig,pred,error_bound)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                decomps[x-x_start][y-y_start][z-z_start]=decomp
                array[x][y][z]=decomp
    #array[x_start:x_end,y:start:y_end]=backup

    return loss,decomps,qs,us
model = vae_models[config['model_params']['name']](**config['model_params'])
test = VAEXperiment(model,config['exp_params'])
checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
test.load_state_dict(checkpoint['state_dict'])
test=test.model
test=test.cuda()
if args.eval:
    test.eval()
#test.eval()
xsize=args.xsize
ysize=args.ysize
zsize=args.zsize
size=args.size

array=np.fromfile(args.input,dtype=np.float32).reshape((xsize,ysize,zsize))
minimum=np.min(array)
maximum=np.max(array)
rng=maximum-minimum

picts=[]
for x in range(0,xsize,size):
    for y in range(0,ysize,size):
        for z in range(0,zsize,size):
            endx=min(x+size,xsize)
            endy=min(y+size,ysize)
            endz=min(z+size,zsize)
            pict=array[x:endx,y:endy,z:endz]
            padx=size-pict.shape[0]
            pady=size-pict.shape[1]
            padz=size-pict.shape[2]

            
            

            pict=(pict-global_min)/(global_max-global_min)
            pict=np.pad(pict,((0,padx),(0,pady),(0,padz)))
                    #print(array[x:x+size,y:y+size])
            pict=pict*2-1
            pict=np.expand_dims(pict,0)
            picts.append(pict)
picts=np.array(picts)



with torch.no_grad():
    outputs=test(torch.from_numpy(picts).to(device))
    '''
    length=picts.shape[0]

    outputs1=test(torch.from_numpy(picts[:length//2]).to(device))
    outputs2=test(torch.from_numpy(picts[length//2:]).to(device))

    outputs=torch.cat((outputs1,outputs2))
    '''
if args.mode=="c":
    zs=outputs[2].cpu().detach().numpy()

    predict=outputs[0].cpu().detach().numpy()
    

else:
    zs=np.fromfile(args.latents,dtype=np.float32)
    if args.transpose:
        zs=zs.reshape((args.lsize,-1)).transpose()
    else:
        zs=zs.reshape((-1,args.lsize))
    with torch.no_grad():
    
        if args.gpu:
            predict=test.decode(torch.from_numpy(zs).to(device)).cpu().detach().numpy()
        else:
            predict=test.decode(torch.from_numpy(zs)).detach().numpy()
    
#predict=outputs[0].numpy()
print(zs.shape)

qs=[]
us=[]
recon=np.zeros((xsize,ysize,zsize),dtype=np.float32)
eb=args.error*rng
picts=(picts+1)/2
picts=picts*(global_max-global_min)+global_min
nn_count=0
lorenzo_count=0
if args.bits==32:
    
   
    '''
    latents=[]
    for element in list(temp_latents):
        bar=BitArray(float=element,length=32)   
        bs="".join([str(int(x)) for x in bar])

        latents.append(eval(r'0b'+ bs))
    '''
    latents=zs
    predict=(predict+1)/2
    predict=predict*(global_max-global_min)+global_min

    idx=0
    for x in range(0,xsize,size):
        for y in range(0,ysize,size):
            for z in range(0,zsize,size):
                endx=min(x+size,xsize)
                endy=min(y+size,ysize)
                endz=min(z+size,zsize)
                origs=picts[idx][0][:endx-x,:endy-y,:endz-z]
                preds=predict[idx][0][:endx-x,:endy-y,:endz-z]
                recon[x:endx,y:endy,z:endz]=preds
                if args.lossmode==1:
                    loss_1=np.sum(np.abs(origs-preds))
                else:
                    loss_1=np.sum(np.abs(origs-preds)//eb)
                loss_2,decomp_block,q_block,u_block=lorenzo(array,x,y,z,eb,size)
                #print(idx)
                #print(loss_1)
                #print(loss_2)
                #print("--")
                if loss_2<=loss_1:
                    lorenzo_count+=1
                    qs=qs+q_block
                    us=us+u_block
                else:
                    nn_count+=1
                    for a in range(x,endx):
                        for b in range(y,endy):
                            for c in range(z,endz):
                                orig=origs[a-x][b-y][c-z]
                            
                                pred=preds[a-x][b-y][c-z]
                            
                            #recon[a][b][c]=pred
                                quant,decomp=quantize(orig,pred,eb)
                           
                            #print(orig-decomp)
                                qs.append(quant)
                                if quant==0:
                                    us.append(decomp)
                                array[a][b][c]=decomp
                idx=idx+1

else:
    radius=2**args.bits
    rs=outputs[0].cpu().detach().numpy()
    zmin=np.min(zs)
    zmax=np.max(zs)
    latents=[]
    for i in range(zs.shape[0]):
        for j in range(zs.shape[1]):
            tmp=int((zs[i][j]-zmin)*radius/(zmax-zmin))
            latents.append(tmp)
            zs[i][j]=(tmp/radius)*(zmax-zmin)+zmin
    with torch.no_grad():
        if args.gpu:
            predict=test.decode(torch.from_numpy(zs).to('cuda')).cpu().detach().numpy()
        else:
            predict=test.decode(torch.from_numpy(zs)).detach().numpy()
    predict=(predict+1)/2
    predict=predict*(global_max-global_min)+global_min
    idx=0
    for x in range(0,xsize,size):
        for y in range(0,ysize,size):
            for z in range(0,zsize,size):
                endx=min(x+size,xsize)
                endy=min(y+size,ysize)
                endz=min(z+size,zsize)

                for a in range(x,endx):
                    for b in range(y,endy):
                        for c in range(z,endz):
                            orig=picts[idx][0][a-x][b-y][c-z]
                            
                            pred=predict[idx][0][a-x][b-y][c-z]
                            
                            recon[a][b][c]=pred
                            quant,decomp=quantize(orig,pred,eb)
                            qs.append(quant)
                            if quant==0:
                                us.append(decomp)
                            array[a][b][c]=decomp
                idx=idx+1


latents=np.array(latents,dtype=np.float32)
quants=np.array(qs,dtype=np.int32)
unpreds=np.array(us,dtype=np.float32)
print("%d blocks used NN." % nn_count)
print("%d blocks used Lorenzo." % lorenzo_count)
if args.latents!=None and args.mode=="c":
    if args.transpose:
        latents=latents.transpose()
    latents.tofile(args.latents)
if args.quant!=None:
    quants.tofile(args.quant)
if args.unpred!=None:
    unpreds.tofile(args.unpred)
if args.recon!=None:
    recon.tofile(args.recon)
if args.decomp!=None:
    array.tofile(args.decomp)