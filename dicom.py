import numpy as np
import pydicom
import cv2
import os
import torch
import torch.utils.data as data
from torchvision.transforms import Compose #ToTensor, Resize, Normalize, RandomHorizontalFlip, TenCrop
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch.nn as nn
from dcm_options import args
import unet
from matplotlib import pyplot as plt



def mask_transform(opt):
    if opt.augmentation:
        compose = Compose([
            #transforms.Scale(opt.img_size),
            #transforms.CenterCrop([224, 224]),
            # RandomHorizontalFlip(),
            #transforms.ToTensor(),
        ])
    else:
        compose = Compose([
            # transforms.Scale(opt.img_size),
            # transforms.ToTensor()
        ])

    return compose


def img_transform(opt):
    if opt.augmentation:
        compose = Compose([
            # transforms.Scale(opt.img_size),
            transforms.CenterCrop([224,224]),
            # RandomHorizontalFlip(),
            transforms.ToTensor(),
            """
            # Normalize or Histogram Equalization? choose 1

            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            """
        ])
    else:
        compose = Compose([
            # transforms.Scale(opt.img_size),
            transforms.ToTensor()
        ])

    return compose

def normalize(img):

    max = img.max()
    min = img.min()
    img = (img-min)/(max-min)
    # print(img.dtype)

    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromFolder, self).__init__()

        # ex) img_list = [1000000, 1000001,1000002,1000003,1000006,1000010,...]
        # 예진: mode 사용하여 img_list를 train과 test하는 경우로 분리
        if opt.mode == 'train':
            self.img_list = os.listdir(opt.train_dir)
        else:  # test
            # ex) img_list = [1000008.dcm, 10000010.dcm,...]
            self.img_list = os.listdir(opt.test_dir)

        self.train_dir = opt.train_dir
        self.test_dir = opt.test_dir
        self.opt = opt
        self.img_transform = img_transform(opt)
        self.mask_transform = mask_transform(opt)

    def __getitem__(self, idx):

        # 예진: train/test 분리
        if self.opt.mode == 'train':
            # ex) img_list_path = '/data/train/1000000'
            img_list_path = os.path.join(self.train_dir)
            # ex) img_files = ['100000.dcm','100000_AorticKnob.png','100000_Carina.png', ...] -> 1개의 dcm + 8개 mask
            img_files = os.listdir(img_list_path)
            masks = np.array([])

            for img_file in img_files:
                # print('\n\n',img)
                if '.dcm' in img_file:
                    # input_img : 0-1 histogram equalization 한 후
                    # input_img = dicom2png(os.path.join(img_list_path, img))
                    input_img = dicom2png(os.path.join(img_list_path, img_file))

                    c, H, W = input_img.shape #dicom2png에서 이미 1.h.w를 반환함
                    input_img = normalize(input_img)
                    # input_img = self.img_transform(input_img)
                    #input_img = input_img.transpose((2, 0, 1))  # HWC to CHW
                    print(H,W,c)

                elif '.png' in img_file:  # .png
                    mask = cv2.imread(os.path.join(img_list_path, img_file), cv2.IMREAD_GRAYSCALE)
                    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    #mask = np.asarray(mask)
                    #print(mask.shape)
                    # mask = mask.reshape(self.opt.num_class, H, W)
                    # mask = mask.reshape(self.opt.num_class, Height, Width)
                    mask = normalize(mask)
                    masks = np.append(masks, mask)

            background = np.zeros((H,W))
            masks = np.append(background, masks)
            masks.reshape(self.opt.num_class + 1, H, W)

            masks = masks.argmax(axis = 0)
            input_img = self.img_transform(input_img)
            print(masks.shape)
        # masks = self.mask_transform(masks)
        # train dataset 에만 transform 반영
        # input_img = self.img_transform(input_img)
        # masks = self.mask_transform(masks)

        else:  # test
            img_list_path = os.path.join(self.test_dir, self.img_list[idx])
            if '.dcm' in img_list_path:
                # input_img : 0-1 histogram equalization 한 후
                # input_img = dicom2png(os.path.join(img_list_path, img))
                input_img = dicom2png(img_list_path)
                input_img = self.img_transform(input_img)
                input_img = normalize(input_img)
            masks = np.array([])

        return input_img, masks, img_list_path

    def __len__(self):
        return len(self.img_list)


def dicom2png(dcm_pth):

#reference https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
#reference https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
#reference https://opencv-python.readthedocs.io/en/latest/doc/20.imageHistogramEqualization/imageHistogramEqualization.html

    dc = pydicom.dcmread(dcm_pth)
    dc_arr = dc.pixel_array


    """자연
    HU(Hounsfield Units)에 맞게 pixel 값 조정
    """
    dc_arr = dc_arr.astype(np.int16)
    dc_arr[dc_arr == -2000] = 0
    #dc_arr = cv2.equalizeHist(dc_arr)


    intercept = dc.RescaleIntercept
    slope = dc.RescaleSlope

    if slope != 1:
        dc_arr = slope * dc_arr.astype(np.float64)
        dc_arr = dc_arr.astype(np.int16)

    dc_arr += np.int16(intercept)

    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    dc_arr = 255.0 * (dc_arr - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    dc_arr[dc_arr > 255.0] = 255.0
    dc_arr[dc_arr < 0.0] = 0.0
    H, W = dc_arr.shape
    dc_arr = dc_arr.reshape(H, W, 1)
    dc_arr = np.uint8(dc_arr)

    # 자연 : create a CLAHE(Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize = (8,8))
    nor_dc_arr = clahe.apply(dc_arr)
    nor_dc_arr = nor_dc_arr.reshape(1, H, W)

    # nor_dc_arr = np.array(nor_dc_arr/255.0, dtype = np.float)
    return nor_dc_arr

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


def trainer(opt, model, optimizer, data_loader, loss_criterion):
    print('====Training====')

    total_loss = 0.0
    model.train()
    for epoch in range(opt.n_epochs):
        print('Epoch {}/{}'.format(epoch, opt.n_epochs - 1))
        print('-' * 10)
        running_loss = 0.0

        for batch, (img, masks) in enumerate(data_loader):

            #img, masks = batch[0], batch[1]
            #inputs = inputs.to(opt.device)
            #labels = labels.to(opt.device)

            if opt.use_cuda:
                img = img.to(opt.device, dtype=torch.float)
                masks = masks.to(opt.device, dtype=torch.long)
                optimizer.zero_grad()


            output = model(img)

            loss = loss_criterion(output, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss = total_loss / opt.epoch_num

        print("***\nTraining => Epoch[%d/%d] :: Loss : %.10f\n" % (
        opt.epoch_num, opt.n_epochs, total_loss))

    return total_loss


if __name__ == "__main__":

    opt = args
    print(opt)

    train_data_loader, valid_data_loader = get_data_loader(opt)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    log_file = os.path.join(opt.log_dir, '%s_log.csv' % (opt.model))

    if opt.model == 'unet':
        # net = unet(opt)
        net = unet.UNet(opt.num_class)  # 채송: 6은 num_class!

    L2_criterion = nn.MSELoss()
    print(net)

    print('===> Setting GPU')
    print("CUDA Available", torch.cuda.is_available())

    if opt.use_cuda and torch.cuda.is_available():
        opt.use_cuda = True
        opt.device = 'cuda'
    else:
        opt.use_cuda = False
        opt.device = 'cpu'

    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("Use" + str(torch.cuda.device_count()) + 'GPUs')
        net = nn.DataParallel(net)

    if opt.use_cuda:
        net = net.to(opt.device)
        L2_criterion = L2_criterion.to(opt.device)

    print('===> Setting Optimizer')
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    best_loss = 1.0

    #for epoch in range(opt.n_epochs):
        #opt.epoch_num = epoch
        #train_loss = trainer(opt, net, optimizer, train_data_loader, loss_criterion=L2_criterion)
    model = trainer(opt, net, optimizer, train_data_loader, loss_criterion=L2_criterion)



