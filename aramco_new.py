from torch.utils.data import Dataset
import numpy as np
import os
class ARAMCO(Dataset):
    def __init__(self,path,start,end,size,global_max=None,global_min=None,norm_min=0):
        size_x=449
        size_y=449
        size_z=235
        self.per_file_block_num=((size_x-1)//size+1)*((size_y-1)//size+1)*((size_z-1)//size+1)
        self.x_length=
        self.len=(end-start)*self.per_file_block_num
        picts=[]
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
                        pict=np.pad(pict,((0,padx),(0,pady),(0,padz)))
                        pict=np.expand_dims(pict,0)
                        if global_max!=None:
                            if norm_min==0:
                                 pict=(pict-global_min)/(global_max-global_min)
                            else:
                                pict=(pict-global_min)*2/(global_max-global_min)-1
                    #print(array[x:x+size,y:y+size])
                        picts.append(pict)
        self.picts=np.array(picts)
        
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        block=idx//self.per_file_block_num
        local_idx=idx % self.per_file_block_num

        return self.picts[idx],0
