import h5py
import os
import numpy as np
with h5py.File(os.path.join('..', 'data', 'ecg_tracings.hdf5'), "r") as f:
    x = np.array(f['tracings'])
    print('done')