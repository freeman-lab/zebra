"""Basic preprocessing of ephys data"""

import numpy as np

def load(inFile):
    """Load 10chFlt data from disk, return as a [channels,samples] sized numpy array
    """
    fd = open(inFile, 'rb')
    data = np.fromfile(file=fd, dtype=np.float32)
    data = data.reshape(data.size/10,10).T
    return data
