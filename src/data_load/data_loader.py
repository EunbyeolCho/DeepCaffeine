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
#import sys
#    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../options')))
from options import args

# from skimage.external.tifffile import imsave, imread, imshow

#need to add center crop!!


def dicom2png(opt, dcm_pth):

#reference https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
#reference https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
#reference https://opencv-python.readthedocs.io/en/latest/doc/20.imageHistogramEqualization/imageHistogramEqualization.html
    
    dc = pydicom.dcmread(dcm_pth)
    dc_arr = np.array(dc.pixel_array) #uint16
    #dc_arr = dc_arr.astype(np.int16)
    #print("dc_arr.dtype: ", dc_arr.dtype)

    if opt.histo_equal:
        """자연
        Normalization-1(0-255)
        -1024~2000 -> -1000~400만 보고싶어 -> 0-255
        """

        MIN_BOUND = dc_arr.min() #uint16
        MAX_BOUND = dc_arr.max()

        #float64  <-- uint16 :overflow 발생
        dc_arr = 255.0 * (dc_arr - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        dc_arr[dc_arr>255.0] = 255.0
        dc_arr[dc_arr<0.0] = 0.0
        #cv2 clahe 사용하기 위해 uint8로 바꿈
        dc_arr = np.uint8(dc_arr)

        """자연
        Histogram Equalization
        """
        #자연 : create a CLAHE(Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (16,16))
        dc_arr = clahe.apply(dc_arr)
    
        #그냥 histogram equalization
        # dc_arr = cv2.equalizeHist(dc_arr)

    return dc_arr

#0-1 normalize and change dtype float64 -> float16
def normalize(img):

    max = img.max()
    min = img.min()
    if (max==min):
        img = img-min
    else:
        img = (img-min)/(max-min)
    #print(img.dtype)
    img = img.astype(np.float16)
    #print(img.dtype)
    return img


# def mask_transform(opt, mask):
#     # h,w 중 작은 사이즈에 맞춰 crop 후 resize
#     H, W = mask.shape

#     if(H > W):
#         diff = H-W
#         crop_mask = mask[diff:H, 0:W]
#     else:
#         diff = W-H
#         crop_mask = mask[0:H, diff:W]

#     resize_mask = cv2.resize(crop_mask, (opt.img_size, opt.img_size), interpolation=cv2.INTER_CUBIC)

#     return resize_mask

def mask_transform(opt, mask) :
    H, W = mask.shape

    if (H<2048) or (W<2048):
        resize_mask = cv2.resize(mask, (opt.img_size, opt.img_size), interpolation = cv2.INTER_CUBIC)
    else : 
        diff_h = round((H-2048)*0.8)
        diff_w = round((W-2048)*0.5)

        crop_mask = mask[diff_h : diff_h + 2048, diff_w : diff_w + 2048]
        resize_mask = cv2.resize(crop_mask, (opt.img_size, opt.img_size), interpolation = cv2.INTER_CUBIC)

    return resize_mask


def img_transform(opt, img):
    H, W = img.shape

    if (H<2048) or (W<2048):
        resize_img = cv2.resize(img, (opt.img_size, opt.img_size), interpolation = cv2.INTER_CUBIC)
    else : 
        diff_h = round((H-2048) * 0.8)
        diff_w = round((W-2048) * 0.5)

        crop_img = img[diff_h : diff_h + 2048, diff_w : diff_w + 2048]
        resize_img = cv2.resize(crop_img, (opt.img_size, opt.img_size), interpolation = cv2.INTER_CUBIC)

    return resize_img


# def img_transform(opt, img):
#     # h,w 중 작은 사이즈에 맞춰 crop 후 resize
#     H, W = img.shape

#     if(H > W):
#         diff = H-W
#         crop_img = img[diff:H, 0:W]
#     else:
#         diff = W-H
#         crop_img = img[0:H, diff:W]


#     resize_img = cv2.resize(crop_img, (opt.img_size, opt.img_size), interpolation = cv2.INTER_CUBIC)

#     return resize_img


'''
if __name__ == "__main__":

    opt= args
    
    path = './data/test'
    imgs = os.listdir(path)
    img = os.path.join(path, imgs[0])

    dcm = dicom2png(opt, img)

    # dcm = cv2.normalize(dcm, dcm, 0, 1, cv2.NORM_MINMAX)
    print(dcm.shape)
    # h, w = dcm.shape
    # dcm = dcm.reshape(h, w, 1)
    # compose = transforms.Compose([transforms.ToTensor()])
    # dcm = compose(dcm)
    # dcm = np.array(dcm)

    dcm = normalize(dcm)
    dcm = dcm.astype(np.float32)
    dcm_name = os.path.join(path, 'before_aug.tiff')

    # cv2.imwrite(dcm_name, dcm)
    imsave(dcm_name, dcm)


    # dcm = dcm.reshape(h,w)
    trans = img_transform(opt, dcm)
    # trans = normalize(trans)
    trans_name = os.path.join(path, 'after_aug.tiff')
    imsave(trans_name, trans)
'''

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

    def __getitem__(self, idx):

        masks = np.array([])
        img_size = np.array([])

        #예진: train/test 분리
        if self.opt.mode == 'train':
            #ex) img_list_path = '/data/train/1000000'
            img_list_path = os.path.join(self.train_dir, self.img_list[idx])
            #ex) img_files = ['100000.dcm','100000_AorticKnob.png','100000_Carina.png', ...] -> 1개의 dcm + 8개 mask
            img_files = os.listdir(img_list_path)

            for img in img_files : 
                if '.dcm' in img : 

                    input_img = dicom2png(self.opt, os.path.join(img_list_path, img))
                    #print("img.shape before transform: ", input_img.shape) #(3001, 2983)

                    H, W = input_img.shape
                    # crop & resize
                    input_img = img_transform(self.opt, input_img)
                    
                    #0-1 normalize and change dtype float64 -> float16
                    input_img = normalize(input_img)
                    #print("transform 후 최종: img.shape: ", input_img.shape) #(512, 512)
                    input_img = input_img.reshape(1, self.opt.img_size, self.opt.img_size)

                    img_size = np.append(img_size, (H,W))

                elif '.png' in img : #.png
                    mask = cv2.imread(os.path.join(img_list_path, img), cv2.IMREAD_GRAYSCALE)
                    #print('mask size : ', mask.shape) #(3001, 2983)

                    #crop & resize
                    mask = mask_transform(self.opt, mask)
                    #print("transform 후 mask.shape: ", mask.shape) #(1006, 1000)

                    masks = np.append(masks, mask)
        
                else : #.png
                    raise TypeError("[*]EXTENSION ERROR : extension is not (dcm, png)")
                    

            background = np.zeros((self.opt.img_size,self.opt.img_size))
            #background  = 0th class
            masks = np.append(background, masks)
            masks = masks.reshape(self.opt.num_class + 1, self.opt.img_size, self.opt.img_size )
            
            #자연 : cross entropy loss 일때, target value : 0 <= target[] <= class-1
            if not self.opt.loss == "dice":
                masks = masks.argmax(axis = 0)
            
        
        else: #test
            img_list_path = os.path.join(self.test_dir, self.img_list[idx])

            if '.dcm' in img_list_path : 
                input_img = dicom2png(self.opt, os.path.join(img_list_path))
                H, W = input_img.shape
                # crop & resize    
                input_img = img_transform(self.opt, input_img)
                #0-1 normalize and change dtype float64 -> float16
                input_img = normalize(input_img)

                input_img = input_img.reshape(1, self.opt.img_size, self.opt.img_size)

                img_size = np.append(img_size, (H,W))

        img_size = img_size.reshape((-1,2))


        return input_img, masks, img_list_path, img_size

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
