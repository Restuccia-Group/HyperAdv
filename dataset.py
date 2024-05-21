import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class RadioML(Dataset):
    """
    custom pytorch dataset to load radioml2018.01 data
    arguments:
        hdf5_file: radioml dataset, hdf5 object
        snr (optional): signal to noise ratio, integer for a single snr or iterator for mixed snr
    """
    def __init__(self, hdf5_file, snr=None):
        if snr is not None:
            if isinstance(snr,int):
                mask = (hdf5_file['Z'][:] == snr).reshape(-1)
            else:
                mask = (hdf5_file['Z'][:] == snr[0]).reshape(-1)
                if len(snr) > 1:
                    for i in snr[1:]:
                        mask = np.logical_or(mask, (hdf5_file['Z'][:] == i).reshape(-1))
            self.label = hdf5_file['Y'][mask]
            self.data = hdf5_file['X'][mask]
        else:
            self.label = hdf5_file['Y']
            self.data = hdf5_file['X']

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data = np.swapaxes(self.data[index],0,1)
        label = self.label[index]
        return data, label