import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import numpy as np
from hsi_dataset import chikuseiTrainDataset, chikuseiValidDataset, xionganTrainDataset, xionganValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, time2file_name
import datetime
import matplotlib.pyplot as plt
import imgvision as iv

device = torch.device('cuda:2')
parser = argparse.ArgumentParser(description="HSI and MSI Fusion Toolbox")
parser.add_argument('--method', type=str, default='ours')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--end_epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=2e-4, help="initial learning rate")
parser.add_argument("--ratio", type=int, default=32, help="ratio rate")
parser.add_argument("--outf", type=str, default='./exp/ours/', help='path log files')
parser.add_argument("--data_root", type=str, default='/home/sysadmin/PSTUN-main/datasets/')
parser.add_argument("--dataset", type=str, default='chikusei',help='chikusei, xiongan,')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='1', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# load dataset
print(f"\nloading dataset: {opt.dataset} ...")
if opt.dataset == 'chikusei':
    train_data = chikuseiTrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, arg=True, stride=opt.stride)
    print(f"Iteration per epoch: {len(train_data)}",'train ratio:',opt.ratio)
    val_data = chikuseiValidDataset(data_root=opt.data_root)
    print("Validation set samples: ", len(val_data),'val ratio:',opt.ratio)

elif opt.dataset == 'xiongan':
    train_data = xionganTrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, arg=True, stride=opt.stride)
    print(f"Iteration per epoch: {len(train_data)}",'train ratio:',opt.ratio)
    val_data = xionganValidDataset(data_root=opt.data_root)
    print("Validation set samples: ", len(val_data),'val ratio:',opt.ratio)

# iterations
per_epoch_iteration = 100
total_iteration = per_epoch_iteration*opt.end_epoch

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(method,opt.ratio, device,pretrained_model_path).to(device)
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.to(device)


# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = True
    iteration = 0
    record_psnr_loss = 0.0
    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        for i, (images, labels,lrhsis) in enumerate(train_loader):
            labels = labels.to(device)
            images = images.to(device)
            lrhsis = lrhsis.to(device)
            images = Variable(images)
            labels = Variable(labels)
            lrhsis = Variable(lrhsis)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()

            output = model(lrhsis,images)
            LOSS = nn.L1Loss()
            loss = LOSS(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration+1
            if iteration % (per_epoch_iteration/5) == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'% (iteration, total_iteration, lr, losses.avg))
            if iteration % per_epoch_iteration == 0:
                psnr_loss,ssim_loss,ergas_loss,sam_loss,rmse_loss = validate(val_loader, model)
                if torch.abs(psnr_loss - record_psnr_loss) < 0.0001 or psnr_loss > record_psnr_loss:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // per_epoch_iteration), iteration, model, optimizer)
                    if psnr_loss > record_psnr_loss:
                        record_psnr_loss = psnr_loss
                if iteration % (per_epoch_iteration*100) == 0:
                    save_checkpoint(opt.outf, (iteration // per_epoch_iteration), iteration, model, optimizer)

                # print loss
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train loss: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//per_epoch_iteration, lr, losses.avg, rmse_loss, psnr_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, "
                      "Test RMSE: %.4f, Test PSNR: %.4f , Test SSIM: %.4f , Test SAM: %.4f , Test ERGAS: %.4f " % (iteration, iteration//per_epoch_iteration, lr, losses.avg, rmse_loss, psnr_loss,ssim_loss,sam_loss,ergas_loss))
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_ssim = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ergas = AverageMeter()
    losses_sam = AverageMeter()
    for i, (inputs, target,lrhsis) in enumerate(val_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        lrhsis = lrhsis.to(device)
        with torch.no_grad():
            # compute output
            output = model(lrhsis,inputs)

            Metric = iv.spectra_metric(target[0].permute(1,2,0).detach().cpu().numpy(),output[0].permute(1,2,0).detach().cpu().numpy(),scale=opt.ratio)
            # Metric.Evaluation()
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
    return torch.tensor(losses_psnr.avg), torch.tensor(losses_ssim.avg),torch.tensor(losses_ergas.avg),torch.tensor(losses_sam.avg),torch.tensor(losses_rmse.avg)

if __name__ == '__main__':
    main()
    print(torch.__version__)