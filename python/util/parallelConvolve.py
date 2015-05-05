import numpy as np
from __main__ import sc

def parallelConvolve(x, k, n):
    
    Lx = x.shape[0]
    Lk = k.shape[0]

    if np.mod(Lk, 2) == 0:
        lpad = Lk/2
        rpad = lpad-1
    else:
        lpad = (Lk-1)/2
        rpad = lpad


    inds = np.linspace(0, Lx, n+1).astype('int') + lpad
    x = np.r_[np.zeros(lpad), x, np.zeros(rpad)]

    chunks = [(i, x[inds[i]-lpad:inds[i+1]+rpad]) for i in xrange(inds.shape[0]-1)]
    rdd = sc.parallelize(chunks)

    kBC = sc.broadcast(k)
    newrdd =  rdd.mapValues(lambda v: np.convolve(v, kBC.value, 'valid'))

    return np.concatenate(newrdd.sortByKey().values().collect())
