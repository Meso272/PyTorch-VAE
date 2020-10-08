import yaml
import argparse
import numpy as np
import torch
from bitstring import BitArray
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
parser.add_argument('--ckpt',  '-k',type='str'
                   )
parser.add_argument('--input',  '-i',type='str'
                   )
parser.add_argument('--error',  '-e',type='float'
                   )
parser.add_argument('--latents',  '-l',type='str'
                   )
parser.add_argument('--quant',  '-q',type='str'
                   )
parser.add_argument('--unpred',  '-u',type='str'
                   )
parser.add_argument('--bits',  '-b',type='int'
                   default=32)
parser.add_argument('--height',  '-h',type='int'
                   default=1800)
parser.add_argument('--width',  '-w',type='int'
                   default=3600)
parser.add_argument('--size','-s',type='int'
                   default=64)
args = parser.parse_args()


with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])
test = VAEXperiment(model,config['exp_params']).load_from_checkpoint(args.k)

array=np.fromfile(args.i,dtype=np.float32)
height=args.h
width=args.w
size=args.s
for x in range(0,height,size):
    for y in range(0,width,size):
        endx=min(x+size,height)
        endy=min(y+size,width)
        pict=array[x:endx,y:endy]
        padx=size-pict.shape[0]
        pady=size-pict.shape[1]
        pict=np.pad(pict,((0,padx),(0,pady)))
        pict=np.expand_dims(pict,0)
                    #print(array[x:x+size,y:y+size])
        picts.append(pict)

minimum=np.min(picts)
maximum=np.max(picts)
rng=maximum-minimum
outputs=test(torch.from_numpy(picts).to('cuda'))
zs=outputs[2]
#predict=outputs[0].numpy()
qs=[]
us=[]
#decomp=np.zeros((height,width),dtype=np.float32)
eb=args.e*rng

if args.b==32:
    predict=outputs[0].numpy()
    temp_latents=zs.numpy().flatten()
    latents=[]
    for element in list(temp_latents):
        latents.append(int('0b'+BitArray(float=element,length=32)))

    idx=0
    for x in range(0,height,size):
        for y in range(0,width,size):
            endx=min(x+size,height)
            endy=min(y+size,width)

            for a in range(x,endx):
                for b in range(y,endy):
                    orig=picts[idx][0][a-x][b-y]
                    pred=predict[idx][0][a-x][b-y]
                    quant,decomp=quantize(orig,pred,eb)
                    qs.append(quant)
                    if quant==0:
                        us.append(decomp)
            idx=idx+1

else:
    radius=2**args.b
    zmin=np.min(zs.numpy())
    zmax=np.max(zs.numpy())
    latents=[]
    for i in range(zs.shape[0]):
        for j in range(zs.shape[1]):
            tmp=int((zs[i][j]-zmin)*radius/(zmax-zmin))
            latents.append(tmp)
            zs[i][j]=(tmp/radius)*(zmax-zmin)+zmin
    predict=test.model.decode(zs).numpy()
    idx=0
    for x in range(0,height,size):
        for y in range(0,width,size):
            endx=min(x+size,height)
            endy=min(y+size,width)

            for a in range(x,endx):
                for b in range(y,endy):
                    orig=picts[idx][0][a-x][b-y]
                    pred=predict[idx][0][a-x][b-y]
                    quant,decomp=quantize(orig,pred,eb)
                    qs.append(quant)
                    if quant==0:
                        us.append(decomp)
            idx=idx+1
latents=np.array(latents,dtype=np.int32)
quants=np.array(qs,dtype=np.int32)
unpreds=np.array(us,dtype=np.float32)
latents.tofile(args.l)
quants.tofile(args.q)
unpreds.tofile(args.u)