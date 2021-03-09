from torch.utils.data import Dataset
import numpy as np
import os
import random
class ARAMCO(Dataset):
    def __init__(self,path,start,end,size,global_max=None,global_min=None,norm_min=0,cache_size=None):
        self.size_x=449
        self.size_y=449
        self.size_z=235
        self.size=size
        self.x_length=((self.size_x-1)//size+1)
        self.y_length=((self.size_y-1)//size+1)
        self.z_length=((self.size_z-1)//size+1)
        self.per_file_block_num=self.x_length*self.y_length*self.z_length
        self.len=(end-start)*self.per_file_block_num
        if cache_size!=None:
            self.cache_size=self.len
        else:
            self.cache_size=cache_size
        self.global_max=global_max
        self.global_min=global_min
        self.norm_min=norm_min
        self.cache={}
        self.cache_count=0
        self.path=path
        
        #print(array)
      
        
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        if idx in self.cache:
            return self.cache[idx],0

        block_idx=idx//self.per_file_block_num
        local_idx=idx % self.per_file_block_num
        x_idx=local_idx//(self.y_length*self.z_length)
        temp_idx=local_idx % (self.y_length*self.z_length)
        y_idx=temp_idx //self.z_length
        z_idx=temp_idx %self.z_length

        x=x_idx*self.size
        y=y_idx*self.size
        z=z_idx*self.size
        endx=min(x+self.size,self.size_x)
        endy=min(y+self.size,self.size_y)
        endz=min(z+self.size,self.size_z)
        s=str(block_idx)
        filename="aramco-snapshot-%s.f32" % s
        filepath=os.path.join(self.path,filename)
        array=np.fromfile(filepath,dtype=np.float32).reshape((self.size_x,self.size_y,self.size_z))
        pict=array[x:endx,y:endy,z:endz]
        padx=self.size-pict.shape[0]
        pady=self.size-pict.shape[1]
        padz=self.size-pict.shape[2]
        pict=np.pad(pict,((0,padx),(0,pady),(0,padz)))
        pict=np.expand_dims(pict,0)
            if self.global_max!=None:
                if self.norm_min==0:
                    pict=(pict-self.global_min)/(self.global_max-self.global_min)
                else:
                    pict=(pict-self.global_min)*2/(self.global_max-self.global_min)-1
        if self.cache_count<self.cache_size:
            

            self.cache_count+=1

        else:
            self.cache.popitem()
        self.cache[idx]=pict


        return pict,0
