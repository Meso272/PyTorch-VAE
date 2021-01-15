import os
folder="compress64_plus_expand"
for file in os.listdir(folder):
    if file.split(".")[0][-1]!='r' and file.split(".")[0][-1]!='d':
        continue
    i=os.path.join(folder,file)
    o=os.path.join("images",file+".png")
    comm="PlotSliceImage -f -i %s -2 3600 1800 -m INDV -n ORI -o %s" % (i,o)
    os.system(comm)
