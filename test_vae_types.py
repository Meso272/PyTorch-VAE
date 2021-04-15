import os
import numpy as np 
configs_list=['ae','bbvae','bhvae','dip_vae','infovae','logcosh_vae','swae_fast','vae','wae_mmd_imq','wae_mmd_rbf_fast']
namelist=['AE','BetaVAE','BetaVAE_H','DIPVAE','InfoVAE','LogCoshVAE','SWAE','VanillaVAE','WAEIMQ','WAERBF']

psnrs=np.zeros((len(configs_list),11))
datafolder="../multisnapshot-data-cleaned/CLDHGH"
for i,name in enumerate(namelist):
    config=configs_list[i]
    for j in range(52,63):
        datafile="CLDHGH_%d.dat" % j
        datapath=os.path.join(datafolder,datafile)
        comm="python cesm_comress.py -c configs/%s.yaml -k ckpts/%s/last.ckpt -i %s -r recon.dat" % (config,name,datapath)
        os.system(comm)
        comm="compareData -f %s recon.dat&>temp.txt" % datapath
        with open("temp.txt","r") as f:
            lines=f.read().splitlines()
            p=eval(lines[6].split(',')[0].split('=')[1])
            print(p)
            psnrs[i][j]=p
        os.system("rm -f recon.dat;rm -f temp.txt")
np.savetxt("vaetypes.txt",psnrs,delimiter="\t")