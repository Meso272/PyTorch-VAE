from torch.utils.data import Dataset
import numpy as np
import os
class ARAMCO(Dataset):
    def __init__(self,path,start,end,size,global_max=None,global_min=None,norm_min=-1,cache_size=None,epsilon=-1):
        size_x=449
        size_y=449
        size_z=235
        picts=[]
        count=[0,0,0,0]
        for i in range(start,end):
            s=str(i)
            if i<10:
                s="0"+s
            filename="aramco-snapshot-%s.f32" % s
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
                            if norm_min==0:
                                pict=(pict-global_min)/(global_max-global_min)
                            else:
                                pict=(pict-global_min)*2/(global_max-global_min)-1
                        pict=np.pad(pict,((0,padx),(0,pady),(0,padz)),constant_values=norm_min)
                        if epsilon>0:
                            var=np.var(pict)
                            if var<1e-5:
                                count[0]+=1
                            if var<1e-4:
                                count[1]+=1
                            if var<1e-3:
                                count[2]+=1
                            if var<1e-2:
                                 count[3]+=1   
                            if var<=epsilon:
                                continue
                        pict=np.expand_dims(pict,0)
                    #print(array[x:x+size,y:y+size])
                        picts.append(pict)
        print(count)
        self.picts=np.array(picts)
        print(self.picts.shape[0])

        
    def __len__(self):
        return self.picts.shape[0]
    def __getitem__(self,idx):
        return self.picts[idx],0
