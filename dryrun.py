import torch
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from dataset import RadioML
from utils import HyperCNNTrainer,HyperCNNEval
from model import HyperCNN

import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

import h5py
# radioml2018.01 dataset, X, Y, Z denotes IQ samples (1024x2), onehot labels (24 classes) and SNR (-20, 30)
f = h5py.File('../dataset/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')

radioML = RadioML(f,snr=np.arange(10,31))
train_size = int(0.8*len(radioML))
test_size = len(radioML)-train_size
radioML_train,radioML_test = random_split(radioML,[train_size,test_size],torch.Generator().manual_seed(42))

train_Loader = DataLoader(radioML_train,batch_size=1024,shuffle=True)
test_Loader = DataLoader(radioML_test,batch_size=1024,shuffle=True)

model = HyperCNN(2,24,256,8)
device = "cuda"
model.load_state_dict(torch.load("bestmodel_hn_affine.pth"),strict=False)
model.to(device)

handler2 = HyperCNNEval(model,device,radioML_test,1024) 

accuracy = handler2.test()
print("\n")
print("clean accuracy: %6f"%(accuracy/radioML_test.__len__()))

correct = handler2.attack(0.05,5,False,False,5000)
print("\n")
print("robust accuracy: %6f"%(correct/5000))