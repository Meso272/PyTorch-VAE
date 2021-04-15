import os
import numpy as np 
import sys
fieldname=sys.argv[1]
datafolder="/home/jliu447/lossycompression/cesm-multisnapshot-5fields/%s" % fieldname

ebs=[-x for x in range(-4,16)]

cr=np.zeros((len(ebs)+1,12),dtype=np.float32)
psnr=np.zeros((len(ebs)+1,12),dtype=np.float32)
maxpwerr=np.zeros((len(ebs)+1,12),dtype=np.float32)

for i,eb in enumerate(ebs):
    eb=2**eb
    cr[i+1][0]=eb
    psnr[i+1][0]=eb
    maxpwerr[i+1][0]=eb
    for j in range(52,63):
        cr[0][j-51]=j
        psnr[0][j-51]=j
        maxpwerr[0][j-51]=j
        filename="%s_%d.dat" % (fieldname,j)
        filepath=os.path.join(datafolder,filename)
        arr=np.fromfile(filepath,dtype=np.float32)
        rng=np.max(arr)-np.min(arr)
        comm="zfp -s -i %s -z out.dat -f -2 3600 1800 -a %f &>temp.txt" % (filepath,eb)
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
            cr[i+1][j-51]=r
            psnr[i+1][j-51]=p
            maxpwerr[i+1][j-51]=e

        os.system("rm -f out.dat")
        os.system("rm -f temp.txt")


np.savetxt("zfp_%s_cr.txt" % fieldname, cr ,delimiter='\t')
np.savetxt("zfp_%s_psnr.txt" % fieldname, psnr ,delimiter='\t')
np.savetxt("zfp_%s_maxpwerr.txt" % fieldname, maxpwerr ,delimiter='\t')