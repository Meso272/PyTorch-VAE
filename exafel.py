from torch.utils.data import Dataset
import numpy as np
import os
class EXAFEL(Dataset):
    def __init__(self,path,start,end,size,global_max,global_min,epsilon=-1):
        height=185*32
        width=388
        picts=[]
        for i in range(start,end):
            
            filename="group_renamed_%d.dat" % i
            filepath=os.path.join(path,filename)
            array=np.fromfile(filepath,dtype=np.float32).reshape((32,height,width))

        #print(array)
            
            for x in range(0,height,size):
                for y in range(0,width,size):
                    endx=min(x+size,height)
                    endy=min(y+size,width)
                    pict=array[x:endx,y:endy]
                    padx=size-pict.shape[0]
                    pady=size-pict.shape[1]

                    pict=(pict-global_min)/(global_max-global_min)

                    pict=np.pad(pict,((0,padx),(0,pady)))
                    pict=pict*2-1
                    if epsilon>0:
                        v=np.var(pict)
                        if v<=epsilon:
                            continue
                    pict=np.expand_dims(pict,0)
                    #print(array[x:x+size,y:y+size])
                    picts.append(pict)
        self.picts=np.array(picts)
    def __len__(self):
        return self.picts.shape[0]
    def __getitem__(self,idx):
        return self.picts[idx],0

