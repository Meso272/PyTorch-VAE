import os
import numpy as np 

datafolder="/home/jliu447/lossycompression/multisnapshot-data-cleaned/CLDHGH"

ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]

cr=np.zeros((12,29),dtype=np.float32)
psnr=np.zeros((12,29),dtype=np.float32)
for i,eb in enumerate(ebs):
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    for j in range(52,63):
        cr[0][j-51]=j
        psnr[0][j-51]=j
        filename="CLDHGH_%s.dat" % str(j)
        filepath=os.path.join(datafolder,filename)
        comm="sz -z -f -i %s -M REL -R %f -2 3600 1800" % (filepath,eb)
        os.system(comm)
        szpath=filepath+".sz"
        comm="sz -x -f -i %s -s %s -2 3600 1800 -a>temp.txt" % (filepath,szpath)
        os.system(comm)
        with open("temp.txt","r") as f:
            lines=f.read().splitlines()
            p=eval(lines[4].split(',')[0].split('=')[1])
            
            r=eval(lines[7].split('=')[1])
            
            cr[i+1][j-51]=r
            psnr[i+1][j-51]=p

        comm="rm -f %s" % szpath
        os.system(comm)
        comm="rm -f %s" % szpath+".out"
        os.system(comm)
        os.system("rm -f temp.txt")


np.savetxt("sz_cldhgh_cr.txt",cr,delimiter='\t')
np.savetxt("sz_cldhgh_psnr.txt",psnr,delimiter='\t')