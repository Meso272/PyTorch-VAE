import os
import numpy as np 
import sys
field=sys.argv[1]
datafolder="/home/jliu447/lossycompression/Hurricane/clean-data-Jinyang"
#fields=["CLOUD","PRECIP","QGRAUP","QRAIN","QVAPOR","U","W","P","QCLOUD","QICE","QSNOW","TC","V"]
ebs=[-x for x in range(-5,18)]
cr=np.zeros((len(ebs)+1,9),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,9),dtype=np.float32)
maxpwerr=np.zeros((len(ebs)+1,9),dtype=np.float32)

for i,eb in enumerate(ebs):
    eb=2**eb
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    maxpwerr[i+1][0]=eb
    for j in range(41,49):
        cr[0][j-40]=j
        psnr[0][j-40]=j
        maxpwerr[0][j-40]=j
        filename="%sf%d.bin" % (field,j)
        filepath=os.path.join(datafolder,filename)
        arr=np.fromfile(filepath,dtype=np.float32)
        rng=np.max(arr)-np.min(arr)
        comm="zfp -s -i %s -z out.dat -f -3 500 500 100 -a %f &>tempzfp.txt" % (filepath,eb)
        
        
        os.system(comm)
        
        
        with open("tempzfp.txt","r") as f:
            line=f.read()
            
            #print(line[0])
            p=eval(line.split(' ')[12].split('=')[1])
            
            r=eval(line.split(' ')[7].split('=')[1])

            e=eval(line.split(' ')[11].split('=')[1])/rng
            #print(p)
            #print(r)
            #print(e)
            cr[i+1][j-40]=r
            psnr[i+1][j-40]=p
            maxpwerr[i+1][j-40]=e

        os.system("rm -f out.dat")
        os.system("rm -f tempzfp.txt")


np.savetxt("zfp_%s_cr.txt" % field, cr ,delimiter='\t')
np.savetxt("zfp_%s_psnr.txt" % field, psnr ,delimiter='\t')
np.savetxt("zfp_%s_maxpwerr.txt" % field, maxpwerr ,delimiter='\t')