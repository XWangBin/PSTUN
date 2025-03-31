import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import imgvision as iv
import scipy.io as sio

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
device = torch.device('cuda:0')
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--ratio", type=int, default=4, help="ratio rate")
parser.add_argument('--data_root', type=str, default='/home/sysadmin/wangbin/PSTUN-Sharpening/datasets/')
parser.add_argument('--method', type=str, default='ours')
parser.add_argument('--pretrained_model_path', type=str, default='/home/sysadmin/HIFpanshareing/ours/net_100epoch.pth')
parser.add_argument('--outf', type=str, default='./exp_wv3/wvresult/')
parser.add_argument("--gpu_id", type=str, default='1')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
val_data = TestDataset(data_root=opt.data_root)
print("Test set samples: ", len(val_data))
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


def validate(val_loader, model):
    model.eval()
    for i, (inputs, target,lrhsis) in enumerate(val_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        lrhsis = lrhsis.to(device)
        print(inputs.shape,target.shape,lrhsis.shape)
        with torch.no_grad():
            # compute output
            output = model(inputs, lrhsis)
        # save results
        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = f'{opt.method}wvreal{i+1}.mat'
        mat_dir = os.path.join(opt.outf, mat_name)
        sio.savemat(mat_dir, {'HSI': result})
    return 0

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method,opt.ratio,device, pretrained_model_path).to(device)
    _ = validate(val_loader, model)
    print(f'method:{method}, test successfull !!!')