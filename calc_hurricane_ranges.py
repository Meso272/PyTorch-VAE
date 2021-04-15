import os
import numpy as np 

datafolder="/home/jliu447/lossycompression/Hurricane/clean-data-Jinyang"

#ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
fields=["CLOUD","PRECIP","QGRAUP","QRAIN","QVAPOR","U","W","P","QCLOUD","QICE","QSNOW","TC","V"]

maxs=[]
mins=[]
for field in enumerate(fields):
    
    for i in range(1,49):
        s=str(i)
        if i<10:
            s='0'+s
        
        
        filename="%sf%s.bin" % (field,s)
        filepath=os.path.join(datafolder,filename)
        a=np.fromfile(filepath,dtype=np.float32)

        if i==0:
            mi=np.min(a)
            mx=np.max(a)
        else:
            mi=min(mi,np.min(a))
            mx=max(np.max(a))
    maxs.append(mx)
    mins.append(mi)

print(maxs)
print(mins)

