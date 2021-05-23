import os
import numpy as np 
import sys

datafolder="/home/jliu447/lossycompression/Hurricane/clean-data-Jinyang"

#ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
ebs=[1e-3]
field=sys.argv[1]
#fields=["CLOUD","PRECIP","QGRAUP","QRAIN","QVAPOR","U","W","P","QCLOUD","QICE","QSNOW","TC","V"]

cr=np.zeros((len(ebs)+1,9),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,9),dtype=np.float32)
ctime=np.zeros((len(ebs)+1,9),dtype=np.float32)
dtime=np.zeros((len(ebs)+1,9),dtype=np.float32)
for i,eb in enumerate(ebs):
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    ctime[i+1][0]=eb
    dtime[i+1][0]=eb
    for j in range(41,49):
        cr[0][j-40]=j
        psnr[0][j-40]=j
        ctime[0][j-40]=j
        dtime[0][j-40]=j
        filename="%sf%d.bin" % (field,j)
        filepath=os.path.join(datafolder,filename)
        comm="sz_interp %s -3 100 500 500 %f>interp_%s.txt" % (filepath,eb,field)
        os.system(comm)
        with open("interp_%s.txt" % field,"r") as f:
            lines=f.read().splitlines()
            print(lines[-2])
            print(lines[-10])
            p=eval(lines[-2].split(',')[4].split(':')[-1])
            
            r=eval(lines[-2].split(',')[1].split(' ')[-1])
            ct=eval(lines[-2].split(',')[2].split(':')[-1])
            dt=eval(lines[-10].split(':')[-1].split('s')[0])
            
            
            cr[i+1][j-40]=r
            psnr[i+1][j-40]=p
            ctime[i+1][j-40]=ct
            dtime[i+1][j-40]=dt

        
        os.system("rm -f interp_%s.txt" % field)


np.savetxt("interpres/sz_%s_cr.txt" % field,cr,delimiter='\t')
np.savetxt("interpres/sz_%s_psnr.txt" % field,psnr,delimiter='\t')
np.savetxt("interpres/sz_%s_ctime.txt" % field,ctime,delimiter='\t')
np.savetxt("interpres/sz_%s_dtime.txt" % field,dtime,delimiter='\t')