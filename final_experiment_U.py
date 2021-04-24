import os
import sys
import numpy as np
configpath=sys.argv[1]
ckptpath=sys.argv[2]
blocksize=int(sys.argv[3])
coeff=float(sys.argv[4])
output=sys.argv[5]
eps=1e-4
if len(sys.argv)>=7:
    eps=float(sys.argv[6])
print(eps)
ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
#ebs=[1e-2,1e-3]
idxrange=[x for x in range(41,49)]
datafolder="/home/jliu447/lossycompression/Hurricane/clean-data-Jinyang" 
pid=str(os.getpid()).strip()
data=np.zeros((len(ebs)+1,len(idxrange)+1,7),dtype=np.float32)
for i in range(7):
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

for i,eb in enumerate(ebs):
    for j,idx in enumerate(idxrange):
        
        filename="Uf%d.bin" % idx
        filepath=os.path.join(datafolder,filename)
        latent_eb=eb/coeff

        comm="python3 predict.py -c %s -k %s -i %s -d 3 -e %f -l %sl.dat -r %sr.dat -s %d -p 1 -x 100 -y 500 -z 500 -mx 85.17703 -mi -79.47297 -eps %f &>%s_t1.txt" % (configpath,ckptpath,filepath,latent_eb,pid,pid,blocksize,eps,pid)
        os.system(comm)
        with open("%s_t1.txt" % pid,"r") as f:
            latent_nbele=eval(f.read())
        os.system("rm -f %s_t1.txt" % pid)
    
        comm="huffmanZstd %sl.dat.q %d 1048576&>%s_t2.txt" % (pid,latent_nbele,pid)
        os.system(comm)
        with open("%s_t2.txt" % pid,"r") as f:
            latent_cr=eval(f.read().splitlines()[-1])
            if latent_cr==0:
                latent_cr=1
            data[i+1][j+1][0]=latent_cr
        os.system("rm -f %s_t2.txt" % pid)


        comm="compress %s.padded %sr.dat %f %d 3 100 500 500&>%s_t3.txt" % (filepath,pid,eb,blocksize,pid)
        os.system(comm)
        with open("%s_t3.txt" % pid,"r") as f:
            lines=f.read().splitlines()
            nn_block=eval(lines[3].split(" ")[0])
            lorenzo_block=eval(lines[4].split(" ")[0])
            data[i+1][j+1][1]=nn_block/(nn_block+lorenzo_block)
        os.system("rm -f %s_t3.txt" % pid)


        comm="../sz_refactory/test/sz_backend %s.padded.q %s.padded.q.u&>%s_t4.txt" % (filepath,filepath,pid)
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
            d_psnr=eval(lines[6].split(',')[0].split('=')[1])
            data[i+1][j+1][3]=d_psnr
        os.system("rm -f %s_t5.txt" % pid)


        comm="compress %s.padded %sr.dat.decompress %f %d 3 100 500 500&>%s_t3.txt" % (filepath,pid,eb,blocksize,pid)
        os.system(comm)
        with open("%s_t3.txt" % pid,"r") as f:
            lines=f.read().splitlines()
            dl_nn_block=eval(lines[3].split(" ")[0])
            dl_lorenzo_block=eval(lines[4].split(" ")[0])
            data[i+1][j+1][4]=dl_nn_block/(dl_nn_block+dl_lorenzo_block)
        os.system("rm -f %s_t3.txt" % pid)


        comm="../sz_refactory/test/sz_backend %s.padded.q %s.padded.q.u&>%s_t4.txt" % (filepath,filepath,pid)
        os.system(comm)
        with open("%s_t4.txt" % pid,"r") as f:
            lines=f.read().splitlines()
            dl_qucr=eval(lines[4].split("=")[-1])
            data[i+1][j+1][5]=dl_qucr
        os.system("rm -f %s_t5.txt" % pid)

        comm="compareData -f %s %s.padded.q.u.d&>%s_t5.txt" % (filepath,filepath,pid)
        os.system(comm)
        with open("%s_t5.txt" % pid,"r") as f:
            lines=f.read().splitlines()
            dl_d_psnr=eval(lines[6].split(',')[0].split('=')[1])
            data[i+1][j+1][6]=dl_d_psnr
        os.system("rm -f %s_t5.txt" % pid)
        os.system("rm -f %sl.* %sr.* %s.padded*" % (pid,pid,filepath))


np.savetxt("%s_latentcr.txt" % output,data[:,:,0],delimiter='\t')
np.savetxt("%s_nnratio.txt" % output,data[:,:,1],delimiter='\t')
np.savetxt("%s_qucr.txt" % output,data[:,:,2],delimiter='\t')
np.savetxt("%s_dpsnr.txt" % output,data[:,:,3],delimiter='\t')
np.savetxt("%s_dlnnratio.txt" % output,data[:,:,4],delimiter='\t')
np.savetxt("%s_dlqucr.txt" % output,data[:,:,5],delimiter='\t')
np.savetxt("%s_dldpsnr.txt" % output,data[:,:,6],delimiter='\t')














