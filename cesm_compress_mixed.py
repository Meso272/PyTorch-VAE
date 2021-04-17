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
parser.add_argument('--height',  '-hgt',type=int,
                   default=1800)
parser.add_argument('--width',  '-w',type=int,
                   default=3600)
parser.add_argument('--size','-s',type=int,
                   default=16)
parser.add_argument('--normalize','-n',type=int,default=1)
parser.add_argument('--transpose','-t',type=int,
                   default=0)
parser.add_argument('--eval','-v',type=int,
                   default=0)
parser.add_argument('--qlatent','-ql',type=int,
                   default=0)
parser.add_argument('--max','-mx',type=float,
                   default=1.0)
parser.add_argument('--min','-mi',type=float,
                   default=0.0)
parser.add_argument('--gpu','-gpu',type=int,
                   default=1)
parser.add_argument('--multigpu','-mg',type=int,
                   default=0)
parser.add_argument('--gpus', '-gs',nargs='*', type=int,default=0)
parser.add_argument('--lossmode','-lm',type=int,
                   default=1)
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
    #if

height=args.height
width=args.width
size=args.size

array=np.fromfile(args.input,dtype=np.float32).reshape((height,width))
minimum=np.min(array)
maximum=np.max(array)
rng=maximum-minimum
picts=[]
for x in range(0,height,size):
    for y in range(0,width,size):
        endx=min(x+size,height)
        endy=min(y+size,width)
        pict=array[x:endx,y:endy]
        padx=size-pict.shape[0]
        pady=size-pict.shape[1]
        pict=np.pad(pict,((0,padx),(0,pady)))
        pict=(pict-global_min)/(global_max-global_min)
        pict=np.pad(pict,((0,padx),(0,pady),(0,padz)))
                    #print(array[x:x+size,y:y+size])
        if args.normalize:
            pict=(pict-args.min)/(args.max-args.min)
        pict=np.pad(pict,((0,padx),(0,pady)))
        if args.normalize:
            pict=pict*2-1
picts=np.array(picts)

with torch.no_grad():
    #try:
    outputs=test(torch.from_numpy(picts).to(device))
    '''
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise exception

    length=picts.shape[0]

    outputs1=test(torch.from_numpy(picts[:length//2]).to(device))
    outputs2=test(torch.from_numpy(picts[length//2:]).to(device))
    outputs=torch.cat((outputs1,outputs2))
    ''' 
def lorenzo(array,x_start,y_start,error_bound,block_size,cross_block=True):
    x_end=min(height,x_start+block_size)
    y_end=min(width,y_start+block_size)
    #backup=np.array(array[x_start:x_end,y:start:y_end])
    #preds=np.zeros((block_size,block_size),dtype=np.float32)
    qs=[]
    us=[]
    loss=0
    decomps=np.zeros((block_size,block_size),dtype=np.float32)
    for x in range(x_start,x_end):
        for y in range(y_start,y_end):
            a=array[x-1][y] if x>0 else 0
            b=array[x][y-1] if y>0 else 0
            c=array[x-1][y-1] if x>0 and y>0 else 0
            orig=array[x][y]
            pred=a+b-c
            if args.lossmode==1:
                loss+=abs(orig-pred)
            else:
                loss+=abs(orig-pred)//error_bound
            q,decomp=quantize(orig,pred,error_bound)
            qs.append(q)
            if q==0:
                us.append(decomp)
            decomps[x-x_start][y-y_start]=decomp
            array[x][y]=decomp
    #array[x_start:x_end,y:start:y_end]=backup

    return loss,decomps,qs,us
           






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
#print(predict.size)
qs=[]
us=[]
recon=np.zeros((height,width),dtype=np.float32)
eb=args.error*rng

nn_count=0
lorenzo_count=0
if args.normalize:
    picts=(picts+1)/2
    picts=picts*(args.max-args.min)+args.min
    predict=(predict+1)/2
    predict=predict*(args.max-args.min)+args.min
if args.bits==32:
    
    #temp_latents=zs.flatten()
    '''
    latents=[]
    for element in list(temp_latents):
        bar=BitArray(float=element,length=32)   
        bs="".join([str(int(x)) for x in bar])

        latents.append(eval(r'0b'+ bs))
    '''
    latents=zs


    idx=0
    for x in range(0,height,size):
        for y in range(0,width,size):
            endx=min(x+size,height)
            endy=min(y+size,width)
            origs=picts[idx][0][:endx-x,:endy-y]
            preds=predict[idx][0][:endx-x,:endy-y]
            recon[x:endx,y:endy]=preds
            if args.lossmode==1:
                loss_1=np.sum(np.abs(origs-preds))
            else:
                loss_1=np.sum(np.abs(origs-preds)//eb)
            loss_2,decomp_block,q_block,u_block=lorenzo(array,x,y,eb,size)
            if loss_2<=loss_1:
                lorenzo_count+=1
                qs=qs+q_block
                us=us+u_block
            else:
                nn_count+=1

                for a in range(x,endx):
                    for b in range(y,endy):

                        orig=origs[a-x][b-y]
                        pred=preds[a-x][b-y]
                        #recon[a][b]=pred
                        quant,decomp=quantize(orig,pred,eb)
                        qs.append(quant)
                        if quant==0:
                            us.append(decomp)
                        array[a][b]=decomp
            idx=idx+1

else:
    radius=2**args.bits
    rs=outputs[0].detach().numpy()
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
    idx=0
    for x in range(0,height,size):
        for y in range(0,width,size):
            endx=min(x+size,height)
            endy=min(y+size,width)

            for a in range(x,endx):
                for b in range(y,endy):
                    orig=picts[idx][0][a-x][b-y]
                    pred=predict[idx][0][a-x][b-y]
                    if args.normalize:
                        orig=(orig+1)/2
                        pred=(pred+1)/2
                    recon[a][b]=pred
                    quant,decomp=quantize(orig,pred,eb)
                    qs.append(quant)
                    if quant==0:
                        us.append(decomp)
                    array[a][b]=decomp
            idx=idx+1
if args.qlatent:
   latents=np.array(latents,dtype=np.int32)
   latents=latents-np.min(latents)
else:
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