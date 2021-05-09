import os
import sys
import numpy as np
configpath=sys.argv[1]
ckptpath=sys.argv[2]
field=sys.argv[3]
blocksize=int(sys.argv[4])
coeff=float(sys.argv[5])
output=sys.argv[6]
maximum={"baryon_density":5.06394195556640625,"temperature":6.6796627044677734375,"dark_matter_density":4.1392154693603515625}
minimum={"baryon_density":-1.306397557258605957,"temperature":2.7645518779754638672,"dark_matter_density":-10}
compress_mode=0# 0 is all, 1 is NN, 2 is lorenzo,3 is only latent cr, 5 is 
if len(sys.argv)>=8:
    compress_mode=int(sys.argv[7])
eps=1e-4
preset_latent_rate=-1
if len(sys.argv)>=9:
    eps=float(sys.argv[8])
if len(sys.argv)>=10:
    if compress_mode==5:
        sz3_bs=int(sys.argv[9])
    else:
        preset_latent_rate=int(sys.argv[9])
print(eps)
#ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
ebs=[4e-2]
idxrange=[0]
datafolder="/home/jliu447/lossycompression/NYX/512x512x512" 
pid=str(os.getpid()).strip()
data=np.zeros((len(ebs)+1,len(idxrange)+1,5),dtype=np.float32)
data[:,:,0]=np.ones((len(ebs)+1,len(idxrange)+1))
for i in range(5):
    data[1:,0,i]=ebs
    data[0,1:,i]=idxrange


'''
latent_crs=np.zeros((len(ebs)+1,12),dtype=np.float32)
nn_ratios=np.zeros((len(ebs)+1,12),dtype=np.float32)
qucrs=np.zeros((len(ebs)+1,12),dtype=np.float32)
d_psnrs=np.zeros((len(ebs)+1,12),dtype=np.float32)
dl_nn_ratios=np.zeros((len(ebs)+1,12),dtype=np.float32)
dl_qucrs=np.zeros((len(ebs)+1,12),dtype=np.float32)
dl_d_psnrs=np.zeros((len(ebs)+1,12),dtype=np.float32)
'''

split=((512/blocksize)**3)//8
for j,idx in enumerate(idxrange):
    latent_rate=preset_latent_rate
    for i,eb in enumerate(ebs):
        print("niewanlong")    
        filename="%s.dat.log10" % field
        filepath=os.path.join(datafolder,filename)
        latent_eb=eb*coeff
        if(compress_mode!=2 or i==0):
            if compress_mode!=5:
                comm="python3 predict.py -sp %d -c %s -k %s -i %s -d 3 -e %f -l %sl.dat -r %sr.dat -s %d -p 1 -x 512 -y 512 -z 512 -mx %f -mi %f -eps %f -para 1 -v 1 >%s_t1.txt" % (split,configpath,ckptpath,filepath,latent_eb,pid,pid,blocksize,maximum[field],minimum[field],eps,pid)
            else:
                comm="python3 predict.py -sp %d -c %s -k %s -i %s -d 3 -l %sl.dat -r %sr.dat -s %d -p 1 -x 512 -y 512 -z 512 -mx %f -mi %f -eps %f -para 1 -v 1 -t 0 >%s_t1.txt" % (split,configpath,ckptpath,filepath,pid,pid,blocksize,maximum[field],minimum[field],eps,pid)
            os.system(comm)
            with open("%s_t1.txt" % pid,"r") as f:
                latent_nbele=eval(f.read())
                if latent_rate==-1:
                    latent_rate=512*512*512/latent_nbele
                    print(latent_rate)
            os.system("rm -f %s_t1.txt" % pid)
    
            #comm="huffmanZstd %sl.dat.q %d 1048576&>%s_t2.txt" % (pid,latent_nbele,pid)
            #os.system(comm)
            #with open("%s_t2.txt" % pid,"r") as f:
                #latent_cr=eval(f.read().splitlines()[-1])
                #if latent_cr==0:
            if compress_mode!=5 and coeff>0:
                comm="huffmanZstd %sl.dat.q %d 1048576&>%s_t2.txt" % (pid,latent_nbele,pid)
                os.system(comm)
                with open("%s_t2.txt" % pid,"r") as f:
                    latent_cr=eval(f.read().splitlines()[-1])
                    if latent_cr==0:
                        comm="sz_demo %sl.dat -1 %d %f %d 0 1 &>%s_t2.5.txt"% (pid,latent_nbele,latent_eb,latent_nbele,pid)
                        os.system(comm)
                        with open("%s_t2.5.txt" % pid,"r") as f:
                            try:
                                lines=f.read().splitlines()
                                latent_cr=eval(lines[8].split("=")[-1])
                            except:
                                latent_cr=0
                        os.system("rm -f %s_t2.5.txt" % pid)
                        os.system("rm -f %s*sz3*")
                        if latent_cr==0:
                            latent_cr=1
                    data[i+1][j+1][0]=latent_cr
                os.system("rm -f %s_t2.txt" % pid)
            elif compress_mode==5:
                comm="sz_demo %sl.dat -1 %d %f %d &>%s_t2.txt"% (pid,latent_nbele,latent_eb,sz3_bs,pid)
                os.system(comm)
                with open("%s_t2.txt" % pid,"r") as f:
                    try:
                        lines=f.read().splitlines()
                        latent_cr=eval(lines[7].split("=")[-1])
                    except:
                        latent_cr=0
                        
                os.system("rm -f %s*sz3*")
                if latent_cr==0:
                    latent_cr=1
                data[i+1][j+1][0]=latent_cr
                os.system("rm -f %s_t2.txt" % pid)

        if(compress_mode<=2):
            comm="compress %s.padded %sr.dat %f %d 3 512 512 512 %d&>%s_t3.txt" % (filepath,pid,eb,blocksize,compress_mode,pid)
            os.system(comm)
            with open("%s_t3.txt" % pid,"r") as f:
                lines=f.read().splitlines()
                nn_block=eval(lines[3].split(" ")[0])
                lorenzo_block=eval(lines[4].split(" ")[0])
                nn_ratio=nn_block/(nn_block+lorenzo_block)
                data[i+1][j+1][1]=nn_ratio
            os.system("rm -f %s_t3.txt" % pid)


            comm="sz_backend %s.padded.q %s.padded.q.u&>%s_t4.txt" % (filepath,filepath,pid)
            os.system(comm)
            with open("%s_t4.txt" % pid,"r") as f:
                lines=f.read().splitlines()
                qucr=eval(lines[4].split("=")[-1])
                data[i+1][j+1][2]=qucr
            os.system("rm -f %s_t4.txt" % pid)

            comm="compareData -f %s %s.padded.q.u.d&>%s_t5.txt" % (filepath,filepath,pid)
            os.system(comm)
            with open("%s_t5.txt" % pid,"r") as f:
                lines=f.read().splitlines()
                psnr=eval(lines[6].split(',')[0].split('=')[1])
                data[i+1][j+1][3]=psnr
            os.system("rm -f %s_t5.txt" % pid)
            final_cr=1/(1/qucr+nn_ratio/(latent_rate*latent_cr))
            data[i+1][j+1][4]=final_cr
        if(compress_mode==4):
            comm="compareData -f %s %sr.dat &>%s_t5.txt" % (filepath,pid,pid)
            os.system(comm)
            data[i+1][j+1][1]=1
            with open("%s_t5.txt" % pid,"r") as f:
                lines=f.read().splitlines()
                psnr=eval(lines[6].split(',')[0].split('=')[1])
                
                data[i+1][j+1][3]=psnr

            latent_cr=data[i+1][j+1][0]
            final_cr=latent_rate*latent_cr

            data[i+1][j+1][4]=final_cr
            os.system("rm -f %s_t5.txt" % pid)
        if compress_mode!=2:
            #pass
            os.system("rm -f %sl.* %sr.* %s.padded*" % (pid,pid,filepath))
    if compress_mode==2:
        os.system("rm -f %sl.* %sr.* %s.padded*" % (pid,pid,filepath))


np.savetxt("%s_psnr.txt" % output,data[:,:,3],delimiter='\t')
if compress_mode!=2:
    np.savetxt("%s_latentcr.txt" % output,data[:,:,0],delimiter='\t')
if compress_mode==0:
    np.savetxt("%s_nnratio.txt" % output,data[:,:,1],delimiter='\t')
if compress_mode<=2:
    np.savetxt("%s_qucr.txt" % output,data[:,:,2],delimiter='\t')

if compress_mode<=2 or compress_mode==4:
    np.savetxt("%s_final_cr.txt" % output,data[:,:,4],delimiter='\t')

















