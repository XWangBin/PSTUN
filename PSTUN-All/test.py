import torch
import numpy as np
import argparse
import os
import cv2
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import chikuseiValidDataset,xionganValidDataset
from torch.utils.data import DataLoader
import imgvision as iv

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
device = torch.device('cuda:1')
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--data_root", type=str, default='/home/sysadmin/wangbin/PSTUN-main/datasets/')
parser.add_argument("--dataset", type=str, default='chikusei',help='chikusei, xiongan,')
parser.add_argument('--method', type=str, default='ours')
parser.add_argument("--ratio", type=str, default='all', help="ratio rate")
parser.add_argument('--pretrained_model_path', type=str, default='/home/sysadmin/wangbin/HIF1/exp/ours5y/chikusei/net_234epoch.pth')
parser.add_argument('--outf', type=str, default='./exp/ours5y/')
parser.add_argument("--gpu_id", type=str, default='1')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
if opt.dataset == 'chikusei':
    val_data = chikuseiValidDataset(data_root=opt.data_root)
    print('Dataset:',opt.dataset, "Test set samples: ", len(val_data),'test ratio:',opt.ratio)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
elif opt.dataset == 'xiongan':
    val_data = xionganValidDataset(data_root=opt.data_root)
    print('Dataset:',opt.dataset, "Test set samples: ", len(val_data),'test ratio:',opt.ratio)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


def validate(val_loader, model):
    model.eval()
    losses_ssim = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ergas = AverageMeter()
    losses_sam = AverageMeter()
    for i, (inputs, target) in enumerate(val_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        ratios = [4,8,16,32,64,128]
        for ratio in ratios:
            lrhsis = torch.zeros((target.shape[0], target.shape[1], target.shape[2] // ratio, target.shape[3] // ratio))
            for c in range(target.shape[0]):
                lrhsi = torch.from_numpy(cv2.GaussianBlur(target[c].permute(1,2,0).detach().cpu().numpy(),ksize=[ratio*2+1]*2,sigmaX=ratio*0.666,sigmaY=ratio*0.666)[ratio//2::ratio,ratio//2::ratio]).permute(2,0,1)
                lrhsis[c] = lrhsi

            lrhsis = lrhsis.to(device)
            # print('test',inputs.shape,target.shape,lrhsis.shape)
            with torch.no_grad():
                # compute output
                output = model(lrhsis,inputs)
                Metric = iv.spectra_metric(target[0].permute(1,2,0).detach().cpu().numpy(),output[0].permute(1,2,0).detach().cpu().numpy(),scale=ratio)

                PSNR = Metric.PSNR()
                SSIM = Metric.SSIM()
                ERGAS = Metric.ERGAS()
                SAM = Metric.SAM()
                RMSE = np.sqrt(Metric.MSE())
                print(f' ratio:{ratio}, RMSE:{RMSE:.4f}, ERGAS:{ERGAS:.3f}, SSIM:{SSIM:.4f}, SAM:{SAM:.3f}, PSNR:{PSNR:.2f}')

            # record loss
            losses_ssim.update(SSIM)
            losses_rmse.update(RMSE)
            losses_psnr.update(PSNR)
            losses_ergas.update(ERGAS)
            losses_sam.update(SAM)

            # # save results
            # result = output.cpu().numpy() * 1.0
            # result = np.transpose(np.squeeze(result), [1, 2, 0])
            # result = np.minimum(result, 1.0)
            # result = np.maximum(result, 0)
            # mat_name = f'{opt.method}chik{ratio}.mat'
            # mat_dir = os.path.join(opt.outf, mat_name)
            # sio.savemat(mat_dir, {'HSI': result})
    return torch.tensor(losses_psnr.avg), torch.tensor(losses_ssim.avg),torch.tensor(losses_ergas.avg),torch.tensor(losses_sam.avg),torch.tensor(losses_rmse.avg)

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, device,opt.dataset, pretrained_model_path).to(device)
    psnr,ssim,ergas,sam,rmse = validate(val_loader, model)
    print(f'method:{method}, ratio:{opt.ratio}, meanRMSE:{rmse:.4f}, meanERGAS:{ergas:.3f}, meanSSIM:{ssim:.4f}, meanSAM:{sam:.3f}, meanPSNR:{psnr:.2f}')