import sys
import numpy as np 

dtype=sys.argv[1]
if dtype=="i":
    dtype=np.int32
else:
    dtype=np.float32

a=np.fromfile(sys.argv[2],dtype=dtype)

eb=float(sys.argv[3])

return eb*(np.max(a)-np.min(a))