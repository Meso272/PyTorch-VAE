import os
crs=[]
for i in range(986):
    filename="group_%d.dat" % i
    szname=filename+".sz"
    os.system("sz -z -f -i %s -M REL -3 388 185 32 -R 1e-2&>f.txt" % filename)
    os.system("sz -x -f -s %s -3 388 185 32 -a -i %s&>temp.txt" %(szname,filename))
    with open("temp.txt","r") as f:
        lines=f.read().splitlines()
        
            
        r=eval(lines[7].split('=')[1])
        crs.append(r)
    os.system("rm -f *sz*")
    os.system("rm -f temp.txt;rm -f f.txt")

with open("crs.txt",'w') as f:
    for i in range(986):
        f.write(str(i)+"\t"+str(crs[i])+"\n")


        
       