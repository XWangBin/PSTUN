from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import scipy.io as sio
class chikuseiTrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8, ratio=32):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.lrhsis = []
        self.arg = arg
        h,w = 2335,2517  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.ratio = ratio

        for i in range(1):
            with h5py.File(data_root+'Chikusei.mat', 'r') as mat:
                hyper = np.float32(np.array(mat['chikusei'])).transpose(1, 2, 0)
                hyper[300:812, 300:812] = hyper[812:1324, 812:1324]

            hyper = hyper / hyper.max()
            srf = sio.loadmat(data_root+'chikusei_128_4.mat')['R']
            # print(hyper.shape)
            msi = hyper@srf
            hyper = np.transpose(hyper, [2,0,1])
            msi = np.float32(msi)
            msi = np.transpose(msi, [2, 0, 1])  # [3,482,512]
            self.hypers.append(hyper)
            self.bgrs.append(msi)
            mat.close()
            print(f'Chikusei scene {i} is loaded.')
        self.img_num = len(self.hypers)
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
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]

        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        lrhsi = cv2.GaussianBlur(hyper.transpose(1,2,0),ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio].transpose(2,0,1)
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
            lrhsi = self.arguement(lrhsi,rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper),np.ascontiguousarray(lrhsi)

    def __len__(self):
        return self.patch_per_img*self.img_num

class chikuseiValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True,ratio=32):
        self.hypers = []
        self.bgrs = []
        self.lrhsis = []
        self.ratio = ratio

        for i in range(1):
            hyper =np.float32(np.array(np.load(data_root+'GT.npy')))
            hyper = hyper / hyper.max()
            srf = sio.loadmat(data_root+'chikusei_128_4.mat')['R']
            bgr = hyper@srf
            lrhsi = cv2.GaussianBlur(hyper,ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]
            hyper = np.transpose(hyper, [2,0,1])

            bgr = np.float32(bgr)
            bgr = np.transpose(bgr, [2, 0, 1])
            lrhsi = np.transpose(np.float32(lrhsi), [2, 0, 1])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            self.lrhsis.append(lrhsi)
            # mat.close()
            print(f'Chikusei scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        lrhsi = self.lrhsis[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper),np.ascontiguousarray(lrhsi)

    def __len__(self):
        return len(self.hypers)


class xionganTrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, stride=8, ratio=32):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.lrhsis = []
        self.arg = arg
        h,w = 3750,1580  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.ratio = ratio

        for i in range(1):
            hyper = np.float32(np.array(np.load(data_root+'xantrain.npy')))
            print(hyper.max())

            hyper[1750:1750 + 512, 650:650 + 512] = hyper[2000:2000 + 512, 0:512]
            hyper = hyper / hyper.max()
            
            srf = sio.loadmat(data_root+'chikuseisrf.mat')['R']
            msi = hyper@srf
            hyper = np.transpose(hyper, [2,0,1])

            msi = np.float32(msi)
            msi = np.transpose(msi, [2, 0, 1])  # [3,482,512]

            self.hypers.append(hyper)
            self.bgrs.append(msi)

            print(f'Xiongan scene {i} is loaded.')
        self.img_num = len(self.hypers)
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
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]

        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        lrhsi = cv2.GaussianBlur(hyper.transpose(1,2,0),ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio].transpose(2,0,1)
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
            lrhsi = self.arguement(lrhsi,rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper),np.ascontiguousarray(lrhsi)

    def __len__(self):
        return self.patch_per_img*self.img_num

class xionganValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True,ratio=32):
        self.hypers = []
        self.bgrs = []
        self.lrhsis = []
        self.ratio = ratio

        for i in range(1):

            hyper = np.float32(np.array(np.load(data_root+'xantrain.npy')))
            hyper = hyper / hyper.max()
            hyper = hyper[1750:1750 + 512, 650:650 + 512]

            srf = sio.loadmat(data_root+'chikuseisrf.mat')['R']
            bgr = hyper@srf
            lrhsi = cv2.GaussianBlur(hyper,ksize=[self.ratio*2+1]*2,sigmaX=self.ratio*0.666,sigmaY=self.ratio*0.666)[self.ratio//2::self.ratio,self.ratio//2::self.ratio]

            hyper = np.transpose(hyper, [2,0,1])
            bgr = np.float32(bgr)

            bgr = np.transpose(bgr, [2, 0, 1])
            lrhsi = np.transpose(np.float32(lrhsi), [2, 0, 1])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            self.lrhsis.append(lrhsi)
            # mat.close()
            print(f'Xiongan scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        lrhsi = self.lrhsis[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper),np.ascontiguousarray(lrhsi)

    def __len__(self):
        return len(self.hypers)
