import os
import numpy as np 

datafolder="/home/jliu447/lossycompression/Hurricane/clean-data-Jinyang"

ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
fields=["CLOUD","PRECIP","QGRAUP","QRAIN","QVAPOR","U","W","P","QCLOUD","QICE","QSNOW","TC","V"]
cr=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)
for i,eb in enumerate(ebs):
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    for j,field in enumerate(fields):
        
        
        filename="%sf45.bin" % str(field)
        filepath=os.path.join(datafolder,filename)
        comm="sz -z -f -i %s -M REL -R %f -3 500 500 100" % (filepath,eb)
        os.system(comm)
        szpath=filepath+".sz"
        comm="sz -x -f -i %s -s %s -3 500 500 100 -a>temp.txt" % (filepath,szpath)
        os.system(comm)
        with open("temp.txt","r") as f:
            lines=f.read().splitlines()
            #print(lines)
            
            p=eval(lines[4].split(',')[0].split('=')[1])
           
         
            r=eval(lines[7].split('=')[1])
            
            
            cr[i+1][j+1]=r
            psnr[i+1][j+1]=p

        comm="rm -f %s" % szpath
        os.system(comm)
        comm="rm -f %s" % szpath+".out"
        os.system(comm)
        os.system("rm -f temp.txt")


np.savetxt("sz_hurricane_cr.txt",cr,delimiter='\t')
np.savetxt("sz_hurricane_psnr.txt",psnr,delimiter='\t')