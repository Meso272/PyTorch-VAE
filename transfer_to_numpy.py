import sys
import numpy as np

ifile=sys.argv[1]
ofile=sys.argv[2]
block_size=int(sys.argv[3])
dims=int(sys.argv[4])
dim=[]
for i in range(dims):
    dim.append(int(sys.argv[5+i]))

raw=[]
with open(ifile,"r") as f:
    raw=f.read().splitlines()
    raw=[float(x) for x in raw]

final=np.zeros(tuple(dim),dtype=np.float32)

i=0

curblock=[0 for _ in range(dims)]
curidx=list(curblock)
while(1):
    try:
        final[tuple(curidx)]=raw[i]
    except:
        print(curidx)
        print(i)
    i+=1
    curidx[-1]+=1
    curdim=dims-1
    while(curdim>0):
        if curidx[curdim]<dim[curdim] and curidx[curdim]-curblock[curdim]<block_size:
            break
        curidx[curdim]=curblock[curdim]
        curdim-=1
        curidx[curdim]+=1
    if curidx[0]>=dim[0] or curidx[0]-curblock[0]>=block_size:
        curblock[-1]+=block_size
        curdim=dims-1
        while(curdim>0):
            if curblock[curdim]<dim[curdim]:
                break
            curblock[curdim]=0
            curdim-=1
            curblock[curdim]+=block_size
    if curblock[0]>=dim[0]:
        break
    curidx=list(curblock)

print(i)
final.tofile(ofile)


