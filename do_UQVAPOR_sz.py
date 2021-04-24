import os
import numpy as np 
import sys

datafolder="/home/jliu447/lossycompression/Hurricane/clean-data-Jinyang"

ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
field=sys.argv[1]
#fields=["CLOUD","PRECIP","QGRAUP","QRAIN","QVAPOR","U","W","P","QCLOUD","QICE","QSNOW","TC","V"]

cr=np.zeros((len(ebs)+1,9),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,9),dtype=np.float32)
for i,eb in enumerate(ebs):
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    for j in range(41,49):
        r[0][j-40]=j
        psnr[0][j-40]=j
        filename="%sf%d.bin" % (field,j)
        filepath=os.path.join(datafolder,filename)
        comm="sz -z -f -i %s -M REL -R %f -3 500 500 100" % (filepath,eb)
        os.system(comm)
        szpath=filepath+".sz"
        comm="sz -x -f -i %s -s %s -3 500 500 100 -a>sztemp.txt" % (filepath,szpath)
        os.system(comm)
        with open("sztemp.txt","r") as f:
            lines=f.read().splitlines()
            #print(lines)
            
            p=eval(lines[4].split(',')[0].split('=')[1])
           
         
            r=eval(lines[7].split('=')[1])
            
            
            cr[i+1][j-40]=r
            psnr[i+1][j-40]=p

        comm="rm -f %s" % szpath
        os.system(comm)
        comm="rm -f %s" % szpath+".out"
        os.system(comm)
        os.system("rm -f sztemp.txt")


np.savetxt("sz_%s_cr.txt" % field,cr,delimiter='\t')
np.savetxt("sz_%s_psnr.txt" % field,psnr,delimiter='\t')