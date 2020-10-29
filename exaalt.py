from torch.utils.data import Dataset
import numpy as np
import os
class EXAALT(Dataset):
    def __init__(self,path,start,end):
        elements=17011951
        seq_length=3137
        seq_num=5423
        picts=[]
       
        filename="xx.dat" 
        filepath=os.path.join(path,filename)
        array=np.fromfile(filepath,dtype=np.float32).reshape((seq_num,seq_length))
        mx=np.max(array)
        mi=np.min(array)
        self.array=(array[start:end,:]-mi)/(mx-mi);
        
    def __len__(self):
        return self.array.shape[0]
    def __getitem__(self,idx):
        return self.array[idx],0
