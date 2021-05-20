import os
import numpy as np 

datafolder="/home/jliu447/lossycompression/aramco"

ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
#ebs=[1e-3,1e-2]
cr=np.zeros((29,11),dtype=np.float32)
psnr=np.zeros((29,11),dtype=np.float32)
ctime=np.zeros((29,11),dtype=np.float32)
dtime=np.zeros((29,11),dtype=np.float32)
for i,eb in enumerate(ebs):
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    ctime[i+1][0]=eb
    dtime[i+1][0]=eb
    for j in range(1510,1601,10):
        y_index=(j-1510)//10+1
        if j==1600:
            j=1599
        cr[0][y_index]=j
        psnr[0][y_index]=j
        ctime[0][y_index]=j
        dtime[0][y_index]=j
        filename="aramco-snapshot-%s.f32" % str(j)
        filepath=os.path.join(datafolder,filename)
        comm="sz_autotuning %s 449 449 235 %f>auto_aramco.txt" % (filepath,eb)
        os.system(comm)
       
        with open("auto_aramco.txt","r") as f:
            lines=f.read().splitlines()
            print(lines[-2])
            print(lines[-10])
            p=eval(lines[-2].split(',')[4].split(':')[-1])
            
            r=eval(lines[-2].split(',')[1].split(' ')[-1])
            ct=eval(lines[-2].split(',')[2].split(':')[-1])
            dt=eval(lines[-10].split(':')[-1].split('s')[0])
            
            cr[i+1][y_index]=r
            psnr[i+1][y_index]=p
            ctime[i+1][y_index]=ct
            dtime[i+1][y_index]=dt

        
        os.system("rm -f auto_aramco.txt")


np.savetxt("autores/sz_aramco_cr.txt",cr,delimiter='\t')
np.savetxt("autores/sz_aramco_psnr.txt",psnr,delimiter='\t')
np.savetxt("autores/sz_aramco_ctime.txt",ctime,delimiter='\t')
np.savetxt("autores/sz_aramco_dtime.txt",dtime,delimiter='\t')