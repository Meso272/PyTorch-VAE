import argparse
import numpy as np

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
    


parser = argparse.ArgumentParser()

parser.add_argument('--input',  '-i',type=str
                   )
parser.add_argument('--error',  '-e',type=float,default=1e-2
                   )
parser.add_argument('-recon',  '-r',type=str,default=None
                   )
parser.add_argument('--decomp',  '-d',type=str,default=None
                   )
parser.add_argument('--quant',  '-q',type=str,default=None
                   )
parser.add_argument('--unpred',  '-u',type=str,default=None
                   )
parser.add_argument('--global_max','-gma',type=float,default=None)
parser.add_argument('--global_min','-gmi',type=float,default=None)
args = parser.parse_args()

orig=np.fromfile(args.input,dtype=np.float32)
recon=np.fromfile(args.recon,dtype=np.float32)
eb=args.error*(np.max(orig)-np.min(orig))

qs=[]
us=[]

for i in range(orig.size):
    o=orig[i]
    r=orig[i]
    if args.global_max!=None:
        r=min(r,args.global_max)
    if args.global_min!=None:
        r=max(r,args.global_min)


    quant,decomp=quantize(o,r,eb)
    qs.append(quant)
    if quant==0:
        us.append(decomp)

quants=np.array(qs,dtype=np.int32)
unpreds=np.array(us,dtype=np.float32)

if args.quant!=None:
    quants.tofile(args.quant)
if args.unpred!=None:
    unpreds.tofile(args.unpred)
