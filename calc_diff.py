import sys
import numpy as np 

dtype=sys.argv[1]

if dtype=="i":
    dtype=np.int32
else:
    dtype=np.float32

ori=sys.argv[2]
tar=sys.argv[3]

o=np.fromfile(ori,dtype=dtype)
n=o.shape[0]
t=np.zeros((n,),dtype=dtype)
last=0
for i in range(n):
    t[i]=o[i]-last
    last=o[i]
t=t-np.min(t)
t.tofile(tar)