import os
import numpy as np 

datafolder="/home/jliu447/lossycompression/aramco"

ebs=[0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]

cr=np.zeros((len(ebs)+1,11),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,11),dtype=np.float32)
maxpwerr=np.zeros((len(ebs)+1,11),dtype=np.float32)

for i,eb in enumerate(ebs):
    eb=2**eb
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    maxpwerr[i+1][0]=eb
    for j in range(1510,1601,10):
        y_index=(j-1510)//10+1
        if j==1600:
            j=1599
        cr[0][y_index]=j
        psnr[0][y_index]=j
        maxpwerr[0][y_index]=j
        filename="aramco-snapshot-%s.f32" % str(j)
        filepath=os.path.join(datafolder,filename)
        arr=np.fromfile(filepath,dtype=np.float32)
        rng=np.max(arr)-np.min(arr)
        comm="zfp -s -i %s -z out.dat -f -3 235 449 449 -a %f &>temp.txt" % (filepath,eb)
        os.system(comm)
        
        
        with open("temp.txt","r") as f:
            line=f.read()
            
            #print(line[0])
            p=eval(line.split(' ')[12].split('=')[1])
            
            r=eval(line.split(' ')[7].split('=')[1])

            e=eval(line.split(' ')[11].split('=')[1])/rng
            #print(p)
            #print(r)
            #print(e)
            cr[i+1][y_index]=r
            psnr[i+1][y_index]=p
            maxpwerr[i+1][y_index]=e

        os.system("rm -f out.dat")
        os.system("rm -f temp.txt")


np.savetxt("zfp_aramco_cr.txt",cr,delimiter='\t')
np.savetxt("zfp_aramco_psnr.txt",psnr,delimiter='\t')
np.savetxt("zfp_aramco_maxpwerr.txt",maxpwerr,delimiter='\t')