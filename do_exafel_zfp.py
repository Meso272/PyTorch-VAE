import os
import numpy as np 
import sys
fieldname=sys.argv[1]
datafolder="/home/jliu447/lossycompression/cesm-multisnapshot-5fields/%s" % fieldname

ebs=[-x for x in range(0,16)]

cr=np.zeros((len(ebs)+1,53),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,53),dtype=np.float32)
maxpwerr=np.zeros((len(ebs)+1,53),dtype=np.float32)

for i,eb in enumerate(ebs):
    eb=2**eb
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    maxpwerr[i+1][0]=eb
    for j in range(301,352):
        cr[0][j-300]=j
        psnr[0][j-300]=j
        maxpwerr[0][j-300]=j
        ffilename="group_renamed_%d.dat" % j
        filepath=os.path.join(datafolder,filename)
        arr=np.fromfile(filepath,dtype=np.float32)
        rng=np.max(arr)-np.min(arr)
        comm="zfp -s -i %s -z out.dat -f -2 185 5920 -a %f &>temp.txt" % (filepath,eb)
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
            cr[i+1][j-300]=r
            psnr[i+1][j-300]=p
            maxpwerr[i+1][j-300]=e

        os.system("rm -f out.dat")
        os.system("rm -f temp.txt")


np.savetxt("zfp_exafel_cr.txt" , cr ,delimiter='\t')
np.savetxt("zfp_exafel_psnr.txt" , psnr ,delimiter='\t')
np.savetxt("zfp_exafel_maxpwerr.txt" , maxpwerr ,delimiter='\t')