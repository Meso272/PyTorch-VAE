import os
import numpy as np 
import sys
#fieldname=sys.argv[1]
datafolder="/home/jliu447/lossycompression/NYX"
fields=["velocity_x","velocity_y","velocity_z"]
ebs=[-x for x in range(14,28)]

cr=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)
maxpwerr=np.zeros((len(ebs)+1,len(fields)+1),dtype=np.float32)

for i,eb in enumerate(ebs):
    eb=2**eb
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    maxpwerr[i+1][0]=eb
    for j,field in enumerate(fields):
        cr[0][j+1]=j
        psnr[0][j+1]=j
        maxpwerr[0][j+1]=j
        filename="%s_3.dat" % field
        filepath=os.path.join(datafolder,filename)
        arr=np.fromfile(filepath,dtype=np.float32)
        rng=np.max(arr)-np.min(arr)
        comm="zfp -s -i %s -z out.dat -f -3 512 512 512 -a %f &>tempzfp.txt" % (filepath,eb)
        
        
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
            cr[i+1][j+1]=r
            psnr[i+1][j+1]=p
            maxpwerr[i+1][j+1]=e

        os.system("rm -f out.dat")
        os.system("rm -f tempzfp.txt")


np.savetxt("zfp_nyx3_cr.txt" , cr ,delimiter='\t')
np.savetxt("zfp_nyx3_psnr.txt" , psnr ,delimiter='\t')
np.savetxt("zfp_nyx3_maxpwerr.txt" , maxpwerr ,delimiter='\t')