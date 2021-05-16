from torch.utils.data import Dataset
import numpy as np
import os
class NYX(Dataset):
    def __init__(self,path,field,start,end,size,log=0,global_max=None,global_min=None,norm_min=-1,cache_size=None,epsilon=-1):
        size_x=512
        size_y=512
        size_z=512
        picts=[]
       # count=[0,0,0,0]
        for i in range(start,end):
            s=str(i)
            
            filename="%s_%s.dat" % (field,s) 
            if log:
                filename+=".log10"
            filepath=os.path.join(path,filename)
            array=np.fromfile(filepath,dtype=np.float32).reshape((size_x,size_y,size_z))
        #print(array)
            for x in range(0,size_x,size):
                for y in range(0,size_y,size):
                    for z in range(0,size_z,size):
                        endx=min(x+size,size_x)
                        endy=min(y+size,size_y)
                        endz=min(z+size,size_z)
                        pict=array[x:endx,y:endy,z:endz]
                        padx=size-pict.shape[0]
                        pady=size-pict.shape[1]
                        padz=size-pict.shape[2]
                        
                        if global_max!=None:
                            rng=global_max-global_min
                            if epsilon>0:
                                r=np.max(pict)-np.min(pict)
                            
                                if r<rng*epsilon:
                                    continue
                            if norm_min==0:
                                pict=(pict-global_min)/(global_max-global_min)
                            else:
                                pict=(pict-global_min)*2/(global_max-global_min)-1
                        pict=np.pad(pict,((0,padx),(0,pady),(0,padz)),constant_values=norm_min)
                        
                        pict=np.expand_dims(pict,0)
                    #print(array[x:x+size,y:y+size])
                        picts.append(pict)
        #print(count)
        self.picts=np.array(picts,dtype=np.float32)
        print(self.picts.shape[0])

        
    def __len__(self):
        return self.picts.shape[0]
    def __getitem__(self,idx):
        return self.picts[idx],0
