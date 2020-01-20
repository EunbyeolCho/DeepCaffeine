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

'''
#예진
의학영상처리 srcnn.py의 test_model 참고
미완성: 1)네트워크에서 나온 결과를 mask로 저장 2)traget과 비교하여 평가지표 구하는 부분 
'''

def class_name(num):
  cn = ['_background', 'Aortic Knob', 'Carina', 'DAO', 'LAA', 'Lt Lower CB', 'Pulmonary Conus', 'Rt Lower CB', 'Rt Upper CB']
  return cn[num]

def inference(opt):
  print()
  print('====Testing====')
  
  start_time = time.time()
  #total_loss = 0.0
  
  #load test data
  test_data_loader = get_test_data_loader(opt)

  print("test_dir is : {}".format(opt.test_dir))
  
  #output_dir 확인
  if not os.path.exists(opt.output_dir) :
    os.makedirs(opt.output_dir)
  
  #/data/volume에서 저장된 model 중 best model load
  _, net = load_model(opt, opt.volume_dir)
  loss_criterion = nn.MSELoss()
  
  if torch.cuda.device_count() > 1 and opt.multi_gpu : 
      print("Use" + str(torch.cuda.device_count()) + 'GPUs')
      net = nn.DataParallel(net)
  
  if opt.use_cuda :
      net = net.to(opt.device)
      loss_criterion = loss_criterion.to(opt.device)
  
  #test하기
  with torch.no_grad():
    total_num = 0
    
    #평가지표에 대한 변수
    sum_loss = 0
    avg_loss = 0
    
    for i, batch in enumerate(test_data_loader) :
      
      img, masks, filepath = batch[0], batch[1], batch[2]

      if opt.use_cuda :
        img = img.to(opt.device, dtype = torch.float)
        masks = masks.to(opt.device, dtype = torch.long)

      out = net(img) #예진:unet의 out은 어떤 형식?
      #채송: unet의 반환값 형식을 말하는거야??
      #out이 어디서 쓰이는 애야..???
      
      out = normalize(out)
      # out = out*255

      print("*****************************************")
      print(filepath)

      maskDir = opt.output_dir

      for b in range(opt.batch_size):
        #결과를 /data/ouput에 저장
        case_id = os.path.basename(filepath[b])[:-4]
        batch_img = out[b, :, :, :]
        batch_mask = one_hot(batch_img)
        print(case_id)

        for j in range(opt.num_class + 1):
          maskDir_case = os.path.join(maskDir, case_id)

          if j == 0: #background
            pass
          else : 
            if not os.path.exists(maskDir_case) :
              os.makedirs(maskDir_case)

            mask = batch_mask[j , :, :]
            # mask[mask>0.5] = 255
            mask = np.array(mask)

            # mask_before_one_hot = out[b, j, :, :]
            # mask_before_one_hot = np.array(mask_before_one_hot)
            # mask_before_one_hot = mask_before_one_hot * 255

            
            cv2.imwrite(os.path.join(maskDir_case, case_id+'_'+class_name(j)+'.png'), mask)
            # cv2.imwrite(os.path.join(maskDir_case, case_id+'_'+class_name(j)+'before-one-hot.png'), mask_before_one_hot)


if __name__ == "__main__":
  opt = args
  print(opt)
  print(opt.mode)
  
  if opt.mode == 'test' : 
    inference(opt)
