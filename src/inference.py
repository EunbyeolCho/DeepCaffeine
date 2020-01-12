import os
import torch
from options import args
import glob
import torch.nn as nn
from data_loader.data_loader import get_test_data_loader
from models.unet import UNet

'''
#예진
의학영상처리 srcnn.py의 test_model 참고
미완성: 1)네트워크에서 나온 결과를 mask로 저장 2)traget과 비교하여 평가지표 구하는 부분 
'''

def inference(opt):
  print()
  print('====Testing====')
  
  start_time = time.time()
  #total_loss = 0.0
  
  #load test data
  test_data_loader = get_test_data_loader(opt)
  print("test_dir is : {}".format(test_dir))
  
  #output_dir 확인
  if not os.path.exists(opt.output_dir) :
    os.makedirs(opt.output_dir)
  
  #/data/volume에서 저장된 model 중 best model load
  _, net = load_model(opt.checkpoint_dir)
  L2_criterion = nn.MSELoss()
  
  if torch.cuda.device_count() > 1 and opt.multi_gpu : 
      print("Use" + str(torch.cuda.device_count()) + 'GPUs')
      net = nn.DataParallel(net)
  
  if opt.use_cuda :
      net = net.to(opt.device)
      L2_criterion = L2_criterion.to(opt.device)
  
  #test하기
  with torch.no_grad():
    total_num = 0
    
    #평가지표에 대한 변수
    sum_loss = 0
    avg_loss = 0
    
    for i, batch in test_data_loader :
      
      img, masks, filepath = batch[0], batch[1], batch[2]

      if opt.use_cuda :
        img = img.to(opt.device, dtype = torch.float)
        masks = masks.to(opt.device, dtype = torch.float)

      out = net(img) #예진:unet의 out은 어떤 형식?
      #채송: unet의 반환값 형식을 말하는거야??
      #out이 어디서 쓰이는 애야..???
      
      model = pytorch_unet.UNet(num_class).to(device)
      #target과 비교하여 평가지표 구하기 -- traget 어디에?


      #결과를 /data/ouput에 저장
      maskDir_case = os.path.join(maskDir, case_id)
      _, mask = cv2.threshold(np.asarray(mask, dtype=unit8), 0, 1, cv2.THRESH_BINARY)
      cv2.imwrite(os.path.join(maskDir_case, case_id+'_'+class_name+'.png'), mask)

      #파일이름과 평가지표 프린트


if __name__ == "__main__":
  opt = args
  print(opt)

  if opt.mode is 'test' : 
    inference(opt)
