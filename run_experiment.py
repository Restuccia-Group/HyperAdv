import torch
import numpy as np
from torch.utils.data import random_split
from dataset import RadioML
from utils import TrainValHandler,EvalHandler
from model import HyperCNN,ConvNet
import h5py
import argparse
import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def arguments_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dnn', '--DNN', type=str, metavar='', default='cnn',
                        help = 'specify the classifier (cnn, hyper)')
    parser.add_argument('-md', '--Mode', type=str, metavar='', default='nt',
                        help = 'specify the training mode (nt, at, trades)')
    parser.add_argument('-dp', '--Data_path', type=str, metavar='', default='../dataset/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
                        help = 'specify the dataset directory')
    parser.add_argument('-cp', '--Ckpt_path', type=str, metavar='', default='resource/',
                        help='specify the checkpoint directory')
    parser.add_argument('-d', '--device', type=int, metavar='', default=0,
                        help='specify the gpu device, -1 means cpu')
    parser.add_argument('-bs','--Batch_size', type=int, metavar='', default=1024,
                        help='specify the batchsize')
    parser.add_argument('-t', '--Test_only', default=False, action='store_true',
                        help='Specify training or testing')
    return parser.parse_args()

def run_exp():
    args = arguments_parser()
    assert args.DNN in ['cnn', 'hyper']
    assert args.Mode in ['nt', 'at', 'trades']
    # set up device
    cuda_id = torch.cuda.device_count()
    if args.device == -1 or cuda_id == 0:
        device = "cpu"
    else:
        device = "cuda:%d"%args.device if args.device < cuda_id else "cuda:%d"%(cuda_id-1)
    # set up dataset    
    try:
        f = h5py.File(args.Data_path,'r')
    except:
        raise ValueError('dataset is not found.')
    radioML = RadioML(f,snr=np.arange(10,31))
    train_size = int(0.8*len(radioML))
    test_size = len(radioML)-train_size
    radioML_train,radioML_test = random_split(radioML,[train_size,test_size],torch.Generator().manual_seed(42))

    lr = 0.0001
    epochs = 1
    patience = 20
    path = '%s/%s_%s.pth'%(args.Ckpt_path,args.DNN,args.Mode)
    if args.DNN == 'cnn':
        model = ConvNet(2,24)
        hyper = False
    else:
        model = HyperCNN(2,24,256,8)
        hyper = True
    
    if args.Test_only:
        model.load_state_dict(torch.load(path),strict=False)
    else:
        handler_train = TrainValHandler(model,device,radioML_train,radioML_test,args.Batch_size,lr,epochs,patience,path,args.Mode,hyper)
        handler_train.train()

    handler_test = EvalHandler(model,device,radioML_test,args.Batch_size,hyper)

    accuracy = handler_test.test()
    print("\n")
    print("clean accuracy: %6f"%(accuracy/radioML_test.__len__()))

    correct = handler_test.attack(0.05,5,5000)
    print("\n")
    print("robust accuracy: %6f"%(correct/5000))

if __name__ == '__main__':
    run_exp()  