import os
import torch
from options import args
import glob
import torch.nn as nn
from data_load.data_loader import get_test_data_loader
from data_load.data_loader import normalize
from models.unet import UNet
from utils.load_model import load_model
from utils.one_hot import one_hot
import time
import cv2
import numpy as np
from utils.resize_output import resize_output


def class_name(num):
    cn = ['_background', 'Aortic Knob', 'Carina', 'DAO', 'LAA', 'Lt Lower CB', 'Pulmonary Conus', 'Rt Lower CB',
          'Rt Upper CB']
    return cn[num]


def inference(opt):
    print()
    print('====Testing====')

    start_time = time.time()

    # load test data
    test_data_loader = get_test_data_loader(opt)

    print("test_dir is : {}".format(opt.test_dir))

    # output_dir 확인
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # /data/volume/ID에서 저장된 model 중 best model load
    _, net = load_model(opt, opt.weight_dir)

    if opt.use_cuda and torch.cuda.is_available():
        opt.use_cuda = True
        opt.device = 'cuda'
    else:
        opt.use_cuda = False
        opt.device = 'cpu'

    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("Use " + str(torch.cuda.device_count()) + ' GPUs')
        net = nn.DataParallel(net)

    if opt.use_cuda:
        net = net.to(opt.device)

    # test하기
    with torch.no_grad():

        for i, batch in enumerate(test_data_loader):

            img, masks, filepath, img_size = batch[0], batch[1], batch[2], batch[3]

            if opt.use_cuda:
                img = img.to(opt.device, dtype=torch.float)
                masks = masks.to(opt.device, dtype=torch.long)

            out = net(img)
            out = out.cpu()

            print("*****************************************")
            print(filepath)

            maskDir = opt.output_dir

            for b in range(len(filepath)):
                # 결과를 /data/ouput에 저장
                case_id = os.path.basename(filepath[b])[:-4]
                batch_img = out[b, :, :, :]
                resize_img = resize_output(opt, batch_img, img_size[b])
                batch_mask = one_hot(resize_img)
                # print(case_id)

                for j in range(opt.num_class + 1):
                    maskDir_case = os.path.join(maskDir, case_id)

                    if j == 0:  # background
                        pass
                    else:
                        if not os.path.exists(maskDir_case):
                            os.makedirs(maskDir_case)

                        mask = batch_mask[j, :, :]

                        # one-hot 한거 눈에 보이게 하고싶으면 주석 푸세요 (0,1->0,255)
                        # mask[mask>0.5] = 255

                        mask = np.array(mask)

                        cv2.imwrite(os.path.join(maskDir_case, case_id + '_' + class_name(j) + '.png'), mask)

                        # one-hot 하기 전 이미지 보고 싶으면 밑에 주석 푸세요
                        # out = normalize(out)
                        # mask_before_one_hot = out[b, j, :, :]
                        # mask_before_one_hot = np.array(mask_before_one_hot)
                        # mask_before_one_hot = mask_before_one_hot * 255
                        # cv2.imwrite(os.path.join(maskDir_case, case_id+'_'+class_name(j)+'before-one-hot.png'), mask_before_one_hot)


if __name__ == "__main__":
    opt = args
    print(opt)
    print(opt.mode)

    if opt.mode == 'test':
        inference(opt)