import cv2
from PIL import Image
import os, glob
import pydicom
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomHorizontalFlip, TenCrop
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../options')))
# from options import args

#need to add center crop!!


def dicom2png(dcm_pth):

#reference https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
#reference https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
#reference https://opencv-python.readthedocs.io/en/latest/doc/20.imageHistogramEqualization/imageHistogramEqualization.html
    
    dc = pydicom.dcmread(dcm_pth)
    dc_arr = np.array(dc.pixel_array)


    """자연
    HU(Hounsfield Units)에 맞게 pixel 값 조정
    """

    #-값이 있어서 uint 말고 int 로 했어
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
    H, W = dc_arr.shape
    #dc_arr = dc_arr.reshape(H, W, 1 )
    dc_arr = np.uint8(dc_arr)

    """자연
    Histogram Equalization & Normalization-2(0-1)
    """
    #자연 : create a CLAHE(Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize = (8,8))
    nor_dc_arr = clahe.apply(dc_arr)

    #예진 : _getitem_에서 reshape 해준다
    #nor_dc_arr = nor_dc_arr.reshape(1, H, W)
    
    #그냥 histogram equalization
    # eq_dc_arr = cv2.equalizeHist(dc_arr)

    return nor_dc_arr


def normalize(img):

    max = img.max()
    min = img.min()
    img = (img-min)/(max-min)
    # print(img.dtype)

    return img


def mask_transform(opt):
    if opt.augmentation:
        compose = Compose([
            transforms.Scale(opt.img_size),
            # transforms.CenterCrop(224),
            #RandomHorizontalFlip(),
            # transforms.ToTensor(),
        ])
    else:
        compose = Compose([
            transforms.Scale(opt.img_size),
            # transforms.ToTensor()
        ])

    return compose

def img_transform(opt):
    """
    # Normalize or Histogram Equalization? choose 1
    # 예진: transfroms에서 normalize하려면 tensor로 변경해야하므로 따로 함
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    """
    if opt.augmentation:
        compose = Compose([
            transforms.Scale(opt.img_size),
            # transforms.CenterCrop(224),
            #RandomHorizontalFlip(),
            # transforms.ToTensor(),
        ])
    else:
        compose = Compose([
            transforms.Scale(opt.img_size),
            # transforms.ToTensor()
        ])
        
    return compose


def resize_mask(masks,  H, W):
    print(type(masks))
    new_mask = np.array([])
    new_mask = masks[0:H, 0:W]
    return new_mask

def resize_img(img, H, W):
    print(type(img))
    new_img = np.array([])
    new_img = img[:, 0:H, 0:W]
    return new_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromFolder, self).__init__()

        #ex) img_list = [1000000, 1000001,1000002,1000003,1000006,1000010,...]
        #예진: mode 사용하여 img_list를 train과 test하는 경우로 분리
        if opt.mode == 'train':
            self.img_list = os.listdir(opt.train_dir)
        else: #test
            #ex) img_list = [1000008.dcm, 10000010.dcm,...]
            self.img_list = os.listdir(opt.test_dir)

        self.train_dir = opt.train_dir
        self.test_dir = opt.test_dir
        self.opt = opt
        self.img_transform = img_transform(opt)
        self.mask_transform = mask_transform(opt)

    def __getitem__(self, idx):

        #예진: train/test 분리
        if self.opt.mode == 'train':
            #ex) img_list_path = '/data/train/1000000'
            img_list_path = os.path.join(self.train_dir, self.img_list[idx])
            #ex) img_files = ['100000.dcm','100000_AorticKnob.png','100000_Carina.png', ...] -> 1개의 dcm + 8개 mask
            img_files = os.listdir(img_list_path)

            masks = np.array([])
            
            for img in img_files : 
                # print('\n\n',img)
                if '.dcm' in img : 
                    #input_img : 0-1 histogram equalization 한 후 
                    input_img = dicom2png(os.path.join(img_list_path, img))
                    #print("img.shape before transform: ", input_img.shape) #(3001, 2983)
                    #예진 : transforms.Compose 이용하기 위해 PIL Image로 변형
                    input_img = Image.fromarray(np.uint8(input_img))
                    #transform: rescale
                    input_img = self.img_transform(input_img)
                    #자연 : bce때문에 normaliz 추가함 
                    #예진 : PIL Image를 다시 normalize하기 위해 numpy array로 변경
                    input_img = np.array(input_img)
                    input_img = normalize(input_img)
                    #print("transform 후 최종: img.shape: ", input_img.shape) #(1006, 1000)
                    H, W = input_img.shape
                    input_img = input_img.reshape(1, H, W)

                elif '.png' in img : #.png
                    mask = cv2.imread(os.path.join(img_list_path, img), cv2.IMREAD_GRAYSCALE)
                    #print('mask size : ', mask.shape) #(3001, 2983)
                    #mask = fake_dcm2png(os.path.join(img_list_path,img))
                    #예진 : transforms.Compose 이용하기 위해 PIL Image로 변형
                    mask = Image.fromarray(np.uint8(mask))
                    #transform: rescale
                    mask = self.mask_transform(mask)
                    #예진 : PIL Image를 다시 normalize하기 위해 numpy array로 변경
                    mask = np.array(mask)
                    #print("transform 후 mask.shape: ", mask.shape) #(1006, 1000)
                    #mask = normalize(mask)
                    masks = np.append(masks, mask)
        
                else : #.png
                    pass

            background = np.zeros((H,W))
            #background  = 0th class
            masks = np.append(background, masks)
            masks = masks.reshape(self.opt.num_class + 1, H, W )
            
            #자연 : cross entropy loss 일때, target value : 0 <= target[] <= class-1
            masks = masks.argmax(axis = 0)
            
            #train dataset 에만 transform 반영
            #input_img = self.img_transform(input_img)
            #masks = self.mask_transform(masks)
        
        else: #test
            img_list_path = os.path.join(self.test_dir, self.img_list[idx])
            if '.dcm' in img_list_path : 
                #input_img : 0-1 histogram equalization 한 후 
                input_img = dicom2png(os.path.join(img_list_path, img))
                # input_img = fake_dcm2png(img_list_path)
                input_img = self.img_transform(input_img)
                input_img = normalize(input_img)

            masks = np.array([])

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


#자연: valid 생성 안할 때 train_data_loader
#예진: test_data_loader로 사용

def get_test_data_loader(opt):
    
    dataset = DatasetFromFolder(opt)

    test_data_loader = DataLoader(dataset = dataset, 
                                    batch_size = opt.batch_size,
                                    shuffle = False)

    return test_data_loader
