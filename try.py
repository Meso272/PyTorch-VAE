import numpy as np 
import sys
from math import ceil,round
a=np.fromfile(sys.argv[1],dtype=np.float32)
blocksize=int(sys.argv[2])
eb=float(sys,argv[3])
size=a.shape[0]
abseb=eb*(np.max(a)-np.min(a))
e=ceil(0.5/abseb)
c=np.zeros(size,dtype=np.int32)
d=np.zeros(size,dtype=np.float32)
for i in range(0,size,blocksize):
    end=min(i+blocksize,size)
    b=a[i:end]
    m=mean(b)
    for j in range(i,end):
        diff=a[j]-m
        c[j]=round(diff*e)
        d[j]=m+c[j]/e

c=c-np.min(c)
c.tofile("roundtest.dat")
d.tofile("latenttest.dat")

