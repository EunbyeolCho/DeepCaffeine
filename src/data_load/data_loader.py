import cv2
import os, glob
import pydicom
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomHorizontalFlip, TenCrop
from torch.utils.data import DataLoader
#need to add center crop!!

#자연 : flip 좋은 생각 아닌거 같아, 우리가 detect하는것이 오른쪽/왼쪽이 정해져 있어서 편파적으로 학습하게 두는게 더 좋을것같은데 어때?


def dicom2png(dcm_pth):

#reference https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
#reference https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
#reference https://opencv-python.readthedocs.io/en/latest/doc/20.imageHistogramEqualization/imageHistogramEqualization.html
    
    # dc = pydicom.dcmread(dcm_pth)
    dc = pydicom.read_file(dcm_pth)
    dc_arr = np.array(dc.pixel_array)


    """자연
    HU(Hounsfield Units)에 맞게 pixel 값 조정
    """
    dc_arr = dc_arr.astype(np.int16)
    #자연 : 범위를 넘어간 pixel value는 -2000으로 고정되는데, 이를 water값인 0으로 바꿔줘야함.
    dc_arr[dc_arr == -2000] = 0

    intercept = dc.RescaleIntercept
    slope = dc.RescaleSlope
    # print('\nslope : %05f\nintercept : %05f\n'%(slope, intercept))

    if slope != 1:
        dc_arr = slope*dc_arr.astype(np.float64)
        dc_arr = dc_arr.astype(np.int16)

    dc_arr +=np.int16(intercept)

    """자연
    Normalization-1(0-255)
    -1024~2000 -> -1000~400만 보고싶어 -> 0-255
    """

    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0


    dc_arr = 255.0 * (dc_arr - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    dc_arr[dc_arr>255.0] = 255.0
    dc_arr[dc_arr<0.0] = 0.0

    """자연
    Histogram Equalization & Normalization-2(0-1)
    """
    #자연 : create a CLAHE(Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.create(clipLimit = 2.0, tileGridSize = (8,8))
    nor_dc_arr = clahe.apply(dc_arr)

    #자연 : type error날것같은데 안돌려봄
    nor_dc_arr = np.array(nor_dc_arr/255.0, dtype = np.float8)
    
    #자연 : dcm있는 자리에 png로저장
    # png_pth = dcm_pth.replace('.dcm', '.png')
    # cv2.imwrite(png_pth, nor_dc_arr)

    return nor_dc_arr



class DatasetFromFolder(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromFolder, self).__init__()

        #ex) img_list = [1000000, 1000001,1000002,1000003,1000006,1000010,...]
        self.img_list = os.listdir(opt.train_dir)
        self.train_dir = opt.train_dir
        self.test_dir = opt.test_dir


    def __getitem__(self, idx):

        #ex) img_list_path = '/data/train/1000000'
        img_list_path = os.path.join(train_dir, self.img_list[idx])
        #ex) img_files = ['100000.dcm','100000_AorticKnob.png','100000_Carina.png', ...] -> 1개의 dcm + 8개 mask
        img_files = os.listdir(img_list_path)

        masks = np.array([])
        
        for img in img_files : 
            if '.dcm' in img : 
                #input_img : 0-1 histogram equalization 한 후 
                input_img = dicom2png(os.path.join(img_list_path, img))
                W , H = input_img.shape
            else : #.png
                mask = cv2.imread(os.path.join(img_list_path, img))
                masks = np.append(masks, mask)

        masks = masks.reshape(8, W, H )


        return input_img, masks, img_list_path

    def __len__(self):
        return len(self.img_list)


def get_data_loader(opt):
    #train dataset만 오픈되어있지만 그래도 valid를 확인하고 싶으니깐 
    #train dataset을 train : valid = 8:2 로 나누는 코드

    dataset = DatasetFromFolder(opt)

    train_len = int(opt.train_ratio * len(dataset))
    valid_len = int(len(dataset) - train_len)

    train_dataset, valid_dataset = data.random_split(dataset, lengths = [train_len, valid_len])

    train_data_loader = DataLoader(dataset = train_dataset, 
                                    batch_size = opt.batch_size,
                                    shuffle = True)
    valid_data_loader = DataLoader(dataset = valid_dataset,
                                    batch_size = opt.batch_size,
                                    shuffle = False)

    return train_data_loader, valid_data_loader


#valid 생성 안할 때 train_data_loader
"""
def get_data_loader(opt):
    
    dataset = DatasetFromFolder(opt)

    train_data_loader = DataLoader(dataset = dataset, 
                                    batch_size = opt.batch_size,
                                    shuffle = True)


    return train_data_loader

"""