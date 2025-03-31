from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import scipy.io as sio
img_scale = 2047.0
class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, stride=8):
        self.crop_size = crop_size
        self.gts = []
        self.mss = []
        self.pans = []
        self.arg = arg
        h,w = 64,64  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        data = h5py.File(data_root+'train_wv3-001.h5')
        gt = data["gt"][...]  # convert to np tpye for CV2.filter
        gt = np.array(gt, dtype=np.float32) / img_scale
        print(gt.shape)
        ms = data["ms"][...]  # convert to np tpye for CV2.filter
        ms = np.array(ms, dtype=np.float32) / img_scale
        pan = data['pan'][...]  # Nx1xHxW
        pan = np.array(pan, dtype=np.float32) / img_scale # Nx1xHxW

        self.gts = gt
        self.mss = ms
        self.pans = pan

        print(f'Real Dataset scene is loaded.')
        self.img_num = gt.shape[0]
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        ms = self.mss[img_idx]
        gt = self.gts[img_idx]
        pan = self.pans[img_idx]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            ms = self.arguement(ms, rotTimes, vFlip, hFlip)
            gt = self.arguement(gt, rotTimes, vFlip, hFlip)
            pan = self.arguement(pan,rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(ms), np.ascontiguousarray(gt),np.ascontiguousarray(pan)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root):
        self.gts = []
        self.mss = []
        self.pans = []

        data = h5py.File(data_root+'valid_wv3.h5')
        gt = data["gt"][...]  # convert to np tpye for CV2.filter
        gt = np.array(gt, dtype=np.float32) / img_scale
        ms = data["ms"][...]  # convert to np tpye for CV2.filter
        ms = np.array(ms, dtype=np.float32) / img_scale
        pan = data['pan'][...]  # Nx1xHxW
        pan = np.array(pan, dtype=np.float32) / img_scale # Nx1xHxW

        self.gts = gt
        self.mss = ms
        self.pans = pan
        print(f'real val scene is loaded.')

    def __getitem__(self, idx):
        ms = self.mss[idx]
        gt = self.gts[idx]
        pan = self.pans[idx]
        return np.ascontiguousarray(ms), np.ascontiguousarray(gt),np.ascontiguousarray(pan)

    def __len__(self):
        return self.gts.shape[0]

class TestDataset(Dataset):
    def __init__(self, data_root):
        self.gts = []
        self.mss = []
        self.pans = []

        for i in range(20):
            
            data = sio.loadmat(data_root+'test/Test(HxWxC)_wv3_data_fr{i+1}.mat')
            gt=0
            ms = data["ms"][...]  # convert to np tpye for CV2.filter
            ms = np.array(ms, dtype=np.float32) / img_scale
            pan = data['pan'][...]  # Nx1xHxW
            pan = np.array(pan, dtype=np.float32) / img_scale # Nx1xHxW

            self.gts.append(gt)
            self.mss.append(ms.transpose(2,0,1))
            self.pans.append(pan.reshape(1,512,512))
            print(f'wv test scene {i+1} is loaded.')

    def __getitem__(self, idx):
        ms = self.mss[idx]
        gt = self.gts[idx]
        pan = self.pans[idx]
        return np.ascontiguousarray(ms), np.ascontiguousarray(gt),np.ascontiguousarray(pan)

    def __len__(self):
        return len(self.mss)