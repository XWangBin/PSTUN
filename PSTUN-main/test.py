import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import chikuseiValidDataset, xionganValidDataset
from torch.utils.data import DataLoader
import imgvision as iv
import scipy.io as sio

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
device = torch.device('cuda:1')
parser = argparse.ArgumentParser(description="HSI and MSI Fusion Toolbox")
parser.add_argument('--data_root', type=str, default='/home/sysadmin/wangbin/PSTUN-main/datasets/')
parser.add_argument("--dataset", type=str, default='chikusei',help='chikusei, xiongan,')
parser.add_argument('--method', type=str, default='ours')
parser.add_argument("--ratio", type=int, default=32, help="ratio rate")
parser.add_argument('--pretrained_model_path', type=str, default='/home/sysadmin/exp/ours/net_115epoch.pth')
parser.add_argument('--outf', type=str, default='./exp/result/')
parser.add_argument("--gpu_id", type=str, default='1')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def validate(val_loader, model):
    model.eval()
    losses_ssim = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ergas = AverageMeter()
    losses_sam = AverageMeter()
    for i, (input, target,lrhsis) in enumerate(val_loader):
        inputs = input.to(device)
        target = target.to(device)
        lrhsis = lrhsis.to(device)
        print(inputs.shape,target.shape,lrhsis.shape)
        with torch.no_grad():
            # compute output
            output = model(lrhsis,inputs)
            Metric = iv.spectra_metric(target[0].permute(1,2,0).detach().cpu().numpy(),output[0].permute(1,2,0).detach().cpu().numpy(),scale=opt.ratio)
            PSNR = Metric.PSNR()
            SSIM = Metric.SSIM()
            ERGS = Metric.ERGAS()
            SAM = Metric.SAM()
            RMSE = np.sqrt(Metric.MSE())

        # record loss
        losses_ssim.update(SSIM)
        losses_rmse.update(RMSE)
        losses_psnr.update(PSNR)
        losses_ergas.update(ERGS)
        losses_sam.update(SAM)

        # # save results
        # result = output.cpu().numpy() * 1.0
        # result = np.transpose(np.squeeze(result), [1, 2, 0])
        # result = np.minimum(result, 1.0)
        # result = np.maximum(result, 0)
        # mat_name = f'{opt.method}chik{opt.ratio}.mat'
        # mat_dir = os.path.join(opt.outf, mat_name)
        # sio.savemat(mat_dir, {'HSI': result})

    return torch.tensor(losses_psnr.avg), torch.tensor(losses_ssim.avg),torch.tensor(losses_ergas.avg),torch.tensor(losses_sam.avg),torch.tensor(losses_rmse.avg)

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method

    # load dataset
    if opt.dataset == 'chikusei':
        val_data = chikuseiValidDataset(data_root=opt.data_root)
        print('Dataset:',opt.dataset, "Test set samples: ", len(val_data),'test ratio:',opt.ratio)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    elif opt.dataset == 'xiongan':
        val_data = xionganValidDataset(data_root=opt.data_root)
        print('Dataset:',opt.dataset, "Test set samples: ", len(val_data),'test ratio:',opt.ratio)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = model_generator(method,opt.ratio,device, pretrained_model_path).to(device)
    psnr,ssim,ergas,sam,rmse = validate(val_loader, model)
    print(f'method:{method}, ratio:{opt.ratio}, RMSE:{rmse:.4f}, ERGAS:{ergas:.3f}, SSIM:{ssim:.4f}, SAM:{sam:.3f}, PSNR:{psnr:.2f}')



# if __name__ == '__main__':
#     cudnn.benchmark = True
#     pretrained_model_path = opt.pretrained_model_path
#     method = opt.method
#     ratios = [4,8,16,32,64,128]
#     for i in range(len(ratios)):
#         opt.ratio = ratios[i]

#         # load dataset
#         val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True,ratio=opt.ratio)
#         print("Validation set samples: ", len(val_data),'val ratio:',opt.ratio)
#         val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

#         model = model_generator(method,opt.ratio,device, pretrained_model_path).to(device)
#         psnr,ssim,ergas,sam,rmse = validate(val_loader, model)
#         print(f'method:{method}, ratio:{ratios[i]}, RMSE:{rmse:.4f}, ERGAS:{ergas:.3f}, SSIM:{ssim:.4f}, SAM:{sam:.3f}, PSNR:{psnr:.2f}')
#         print('============================================================================================')
