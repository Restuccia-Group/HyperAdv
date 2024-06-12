import torch
import numpy as np
from torch.utils.data import random_split
from dataset import RadioML
from utils import TrainValHandler,TeachHandler,EvalHandler
from model import HyperCNN,HyperCNNChunk
import h5py
import argparse
import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def arguments_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    # set up teacher model
    teacher_name = "%s/hyper_%s.pth"%(args.Ckpt_path,args.Mode)
    teacher = HyperCNN(2,24,256,8)
    teacher.load_state_dict(torch.load(teacher_name),strict=False)
    # set up student model
    student_name = "%s/hyper_s_%s.pth"%(args.Ckpt_path,args.Mode)
    student = HyperCNNChunk(2,24,56,8,teacher.weight,teacher.bias)

    if args.Test_only:
        student.load_state_dict(torch.load(student_name),strict=False)
    else:
        handler = TeachHandler(teacher,student,device,0.001,50000,10)
        handler.teach()
        handler = TrainValHandler(student,device,radioML_train,radioML_test,args.Batch_size,0.0001,1,1,student_name,args.Mode,True)
        handler.train()

    handler = EvalHandler(student,device,radioML_test,args.Batch_size,True)

    accuracy = handler.test()
    print("\n")
    print("clean accuracy: %6f"%(accuracy/radioML_test.__len__()))

    correct = handler.attack(0.05,5,5000)
    print("\n")
    print("robust accuracy: %6f"%(correct/5000))

if __name__ == '__main__':
    run_exp()  