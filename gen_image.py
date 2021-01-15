import os
ifolder="/home/jliu447/lossycompression/PyTorch-VAE/compress64_plus_expand"
ofolder="/home/jliu447/lossycompression/PyTorch-VAE/images"
for file in os.listdir(ifolder):
    if "." not in file :
        continue

    if file.split(".")[0][-1]!='r' and file.split(".")[0][-1]!='d':
        continue
    i=os.path.join(ifolder,file)
    o=os.path.join(ofolder,file+".png")
    comm="PlotSliceImage -f -i %s -2 3600 1800 -m INDV -n ORI -o %s" % (i,o)
    os.system(comm)
