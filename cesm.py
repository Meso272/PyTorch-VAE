from torch.utils.data import Dataset
import numpy as np
import os
class CLDHGH(Dataset):
    def __init__(self,path,start,end,size,normalize=True):
        height=1800
        width=3600
        picts=[]
        for i in range(start,end):
            s=str(i)
            if i<10:
                s="0"+s
            filename="CLDHGH_%s.dat" % s
            filepath=os.path.join(path,filename)
            array=np.fromfile(filepath,dtype=np.float32).reshape((height,width))
        #print(array)
            for x in range(0,height,size):
                for y in range(0,width,size):
                    endx=min(x+size,height)
                    endy=min(y+size,width)
                    pict=array[x:endx,y:endy]
                    padx=size-pict.shape[0]
                    pady=size-pict.shape[1]
                    pict=np.pad(pict,((0,padx),(0,pady)))
                    if normalize:
                        pict=pict*2-1
                    pict=np.expand_dims(pict,0)
                    #print(array[x:x+size,y:y+size])
                    picts.append(pict)
        self.picts=np.array(picts)
    def __len__(self):
        return self.picts.shape[0]
    def __getitem__(self,idx):
        return self.picts[idx],0

class CESM(Dataset):
    def __init__(self,path,start,end,size,field='FREQSH',global_max=None,global_min=None):
        height=1800
        width=3600
        picts=[]
        path=os.path.join(path,field)
        for i in range(start,end):
            s=str(i)
            if i<10:
                s="0"+s
            filename="%s_%s.dat" % (field,s)
            filepath=os.path.join(path,filename)
            array=np.fromfile(filepath,dtype=np.float32).reshape((height,width))
        #print(array)
            for x in range(0,height,size):
                for y in range(0,width,size):
                    endx=min(x+size,height)
                    endy=min(y+size,width)
                    pict=array[x:endx,y:endy]
                    padx=size-pict.shape[0]
                    pady=size-pict.shape[1]
                    pict=np.pad(pict,((0,padx),(0,pady)))
                    if global_max!=None:
                        pict=(pict-global_min)/(global_max-global_min)
                        pict=pict*2-1
                    pict=np.expand_dims(pict,0)
                    #print(array[x:x+size,y:y+size])
                    picts.append(pict)
        self.picts=np.array(picts)
    def __len__(self):
        return self.picts.shape[0]
    def __getitem__(self,idx):
        return self.picts[idx],0
