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
parser.add_argument('--ckpt',  '-k',type=str
                   )
parser.add_argument('--input',  '-i',type=str
                   )
parser.add_argument('--error',  '-e',type=float
                   )
parser.add_argument('--latents',  '-l',type=str
                   )
parser.add_argument('--quant',  '-q',type=str
                   )
parser.add_argument('--unpred',  '-u',type=str
                   )
parser.add_argument('-recon',  '-r',type=str
                   )
parser.add_argument('--decomp',  '-d',type=str
                   )
parser.add_argument('--bits',  '-b',type=int,
                   default=32)

args = parser.parse_args()


with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])
test = VAEXperiment(model,config['exp_params'])
checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
test.load_state_dict(checkpoint['state_dict'])



array=np.fromfile(args.input,dtype=np.float32).reshape((5423,3137))
mx=np.max(array)
mi=np.min(array)
array=array[4400:]
input_array=(array-mi)/(mx-mi)
num_instance=5423-4400
ins_dim=3137


minimum=np.min(array)
maximum=np.max(array)
rng=maximum-minimum
outputs=test(torch.from_numpy(input_array))
zs=outputs[2].detach().numpy()
#predict=outputs[0].numpy()
qs=[]
us=[]
recon=np.zeros((num_instance,ins_dim),dtype=np.float32)
eb=args.error*rng

if args.bits==32:
    predict=outputs[0].detach().numpy()*(mx-mi)+mi
    temp_latents=zs.flatten()
    latents=[]
    for element in list(temp_latents):
        bar=BitArray(float=element,length=32)   
        bs="".join([str(int(x)) for x in bar])

        latents.append(eval(r'0b'+ bs))


   
    for x in range(num_instance):
        for y in range(ins_dim):
            
            orig=array[x][y]
            pred=predict[x][y]
            recon[x][y]=pred
            quant,decomp=quantize(orig,pred,eb)
            qs.append(quant)
            if quant==0:
                us.append(decomp)
            array[x][y]=decomp
            

else:
    radius=2**args.bits
    #rs=outputs[0].detach().numpy()
    zmin=np.min(zs)
    zmax=np.max(zs)
    latents=[]
    for i in range(zs.shape[0]):
        for j in range(zs.shape[1]):
            tmp=int((zs[i][j]-zmin)*radius/(zmax-zmin))
            latents.append(tmp)
            zs[i][j]=(tmp/radius)*(zmax-zmin)+zmin
    predict=test.model.decode(torch.from_numpy(zs)).detach().numpy()*(mx-mi)+mi
    
    for x in range(num_instance):
        for y in range(ins_dim):
            
            orig=array[x][y]
            pred=predict[x][y]
            recon[x][y]=pred
            quant,decomp=quantize(orig,pred,eb)
            qs.append(quant)
            if quant==0:
                us.append(decomp)
            array[x][y]=decomp
           
latents=np.array(latents,dtype=np.int32)
quants=np.array(qs,dtype=np.int32)
unpreds=np.array(us,dtype=np.float32)
latents.tofile(args.latents)
quants.tofile(args.quant)
unpreds.tofile(args.unpred)
recon.tofile(args.recon)
array.tofile(args.decomp)