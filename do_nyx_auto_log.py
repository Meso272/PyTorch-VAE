import os
import numpy as np 

datafolder="/home/jliu447/lossycompression/NYX/512x512x512"

ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
fields=["baryon_density","temperature","dark_matter_density"]
cr=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)
ctime=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)
dtime=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)
for i,eb in enumerate(ebs):
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    ctime[i+1][0]=eb
    dtime[i+1][0]=eb
    for j,field in enumerate(fields):
        
        
        filename="%s.dat.log10" % str(field)
        filepath=os.path.join(datafolder,filename)
        comm="sz_autotuning %s 512 512 512 %f>auto_%s.txt" % (filepath,eb,field)
        os.system(comm)
        
        with open("auto_%s.txt" % field,"r") as f:
            lines=f.read().splitlines()
            print(lines[-2])
            print(lines[-10])
            p=eval(lines[-2].split(',')[4].split(':')[-1])
            
            r=eval(lines[-2].split(',')[1].split(' ')[-1])
            ct=eval(lines[-2].split(',')[2].split(':')[-1])
            dt=eval(lines[-10].split(':')[-1].split('s')[0])
            
            
            cr[i+1][j+1]=r
            psnr[i+1][j+1]=p
            ctime[i+1][j+1]=ct
            dtime[i+1][j+1]=dt


        
        os.system("rm -f auto_%s.txt" %)


np.savetxt("autores/sz_nyxlog10_cr.txt",cr,delimiter='\t')
np.savetxt("autores/sz_nyxlog10_psnr.txt",psnr,delimiter='\t')
np.savetxt("autores/sz_nyxlog10_ctime.txt",ctime,delimiter='\t')
np.savetxt("autores/sz_nyxlog10_dtime.txt",ctime,delimiter='\t')