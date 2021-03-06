import sys
import numpy as np

orifile=sys.argv[1]
predfile1=sys.argv[2]
predfile2=sys.argv[3]
block_size=int(sys.argv[4])
dims=int(sys.argv[5])
dim=[]
for i in range(dims):
    dim.append(int(sys.argv[6+i]))

ori=np.fromfile(orifile,dtype=np.float32).reshape(tuple(dim))
pred1=np.fromfile(predfile1,dtype=np.float32).reshape(tuple(dim))

pred2=np.fromfile(predfile2,dtype=np.float32).reshape(tuple(dim))
print(ori.shape)
curblock=[0 for _ in range(dims)]
win1=0
win2=0
while(1):
    print(curblock)
    orislice=np.array(ori)
    pred1slice=np.array(pred1)
    pred2slice=np.array(pred2)

    for i,start in enumerate(curblock):
        end=min(dim[i],start+block_size)
        #print(start)
        #print(end)
        orislice=orislice.take(axis=i,indices=range(start,end))
        pred1slice=pred1slice.take(axis=i,indices=range(start,end))
        pred2slice=pred2slice.take(axis=i,indices=range(start,end))
    #orislice=np.array(orislice)
    #pred1slice=np.array(pred1slice)
    #pred2slice=np.array(pred2slice)
    mse1=np.square(np.subtract(orislice.flatten(),pred1slice.flatten())).mean()
    mse2=np.square(np.subtract(orislice.flatten(),pred2slice.flatten())).mean()
    if mse1<mse2:
        win1+=1
    else:
        win2+=1
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
    

print(win1)
print(win2)

