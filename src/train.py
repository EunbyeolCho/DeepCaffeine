import os
import torch
import random
from time import sleep
from options import args
import time
from data_load.data_loader import get_data_loader
from data_load.data_loader import normalize
from models import unet
import torch.nn as nn
import copy
from utils.saver import save_checkpoint
from tensorboardX import SummaryWriter
import utils.matrics as MATRICS
import numpy as np
from torch.optim import lr_scheduler


def set_loss(opt):
  if opt.loss == 'wce' :
    w = opt.loss_weight
    loss_weight = torch.FloatTensor([1, w , w, w, w, w, w, w, w])
    loss_criterion = nn.CrossEntropyLoss(weight = loss_weight)
  elif opt.loss =='dice' :
    loss_criterion = MATRICS.DCE_LOSS()
  elif opt.loss == 'ce' :
    loss_criterion = nn.CrossEntropyLoss()
  else : 
    raise ValueError('set the loss function (wce, ce, dice)')

  return loss_criterion

def trainer(opt, model, optimizer, data_loader, loss_criterion):

  print('====Training====')
  
  start_time = time.time()
  
  total_loss = 0.0
  
  for i, batch in enumerate(data_loader) :
    img, masks = batch[0], batch[1]

    if opt.use_cuda :
      img = img.to(opt.device, dtype = torch.float)
      #자연 : cross_entropy 때문에 mask type long으로 바꿈
      masks = masks.to(opt.device, dtype = torch.long)
      optimizer.zero_grad()

    out = model(img)

    if opt.loss == 'dice' :
      loss = loss_criterion.Get_total_DSC_loss(out, masks)
      # loss = torch.from_numpy(np.asarray(loss)).float().to('cuda')
      loss = loss.to('cuda')
    else :
      loss_criterion = loss_criterion.to(opt.device)
      loss = loss_criterion(out, masks)
    # print(loss)
    loss.backward()
    optimizer.step()

    total_loss +=loss.item()

  total_loss = total_loss/i

  print("***\nTraining %.2fs => Epoch[%d/%d] :: Loss : %.10f\n"%(time.time()-start_time, opt.epoch_num, opt.n_epochs, total_loss)) 
  
  return total_loss
  


def evaluator(opt, model, data_loader, loss_criterion):

  print('====Validation=====')

  start_time = time.time()

  total_loss = 0.0
  with torch.no_grad():
    for i, batch in enumerate(data_loader) :
    
      img, masks = batch[0], batch[1]
      
      if opt.use_cuda :
        img = img.to(opt.device, dtype = torch.float)
        masks = masks.to(opt.device, dtype = torch.long)
        optimizer.zero_grad()

      out = model(img)

      if opt.loss == 'dice' :
        loss = loss_criterion.Get_total_DSC_loss(out, masks)
        loss = loss.to('cuda')
      else :
        loss_criterion = loss_criterion.to(opt.device)
        loss = loss_criterion(out, masks)

      total_loss +=loss.item()
  
  total_loss = total_loss/i
  
  print("***\nValidation %.2fs => Epoch[%d/%d] :: Loss : %.10f\n"%(time.time()-start_time, opt.epoch_num, opt.n_epochs, total_loss)) 
  
  return total_loss
  
  
  
if __name__ == "__main__":
  
  opt = args
  print(opt)
  
  train_data_loader, valid_data_loader = get_data_loader(opt)
  
  if not os.path.exists(opt.log_dir) :
    os.makedirs(opt.log_dir)

  log_file = os.path.join(opt.log_dir, '%s_log.csv'%(opt.model))
  
  if opt.model == 'unet':
    net = unet.UNet(opt.num_class + 1) 
  
  loss_criterion = set_loss(opt)

  print(net)
  
  
  print('===> Setting GPU')
  print("CUDA Available", torch.cuda.is_available())
  
  if opt.use_cuda and torch.cuda.is_available():
    opt.use_cuda = True
    opt.device = 'cuda'
  else : 
    opt.use_cuda = False
    opt.device = 'cpu'

  if torch.cuda.device_count() > 1 and opt.multi_gpu : 
      print("Use" + str(torch.cuda.device_count()) + 'GPUs')
      net = nn.DataParallel(net)

  if opt.use_cuda :
      net = net.to(opt.device)
      # loss_criterion = loss_criterion.to(opt.device)
  
  print('===> Setting Optimizer')
  optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
  schedular = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr= 1e-5, verbose= True)

  best_loss = 1000.0
  old_train_loss = 1.0
  count = 0
  writer = SummaryWriter(log_dir = opt.log_dir)
  for epoch in range(opt.n_epochs):
    opt.epoch_num = epoch
    train_loss = trainer(opt, net, optimizer, train_data_loader, loss_criterion = loss_criterion)
    valid_loss = evaluator(opt, net, valid_data_loader, loss_criterion = loss_criterion)
    
    if opt.loss == 'dice':
      writer.add_scalar('DICELoss/train', train_loss, epoch)
      writer.add_scalar('DICELoss/valid', valid_loss, epoch)
    elif opt.loss == 'wce':
      writer.add_scalar('WCELoss/train', train_loss, epoch)
      writer.add_scalar('WCELoss/valid', valid_loss, epoch)
    else : 
      writer.add_scalar('Loss/train', train_loss, epoch)
      writer.add_scalar('Loss/valid', valid_loss, epoch)

    schedular.step(train_loss)
    delta = abs(old_train_loss - train_loss)
    
    if not opt.save_best:
      save_checkpoint(opt, net, epoch, valid_loss)

      if opt.save_best :
        if valid_loss < best_loss : 
          best_loss = valid_loss
          best_model_wts = copy.deepcopy(net.state_dict())
          #채송: main 함수 다 돌면 valid loss가 가장 좋은 model 저장하도록 하는 
    old_train_loss = train_loss
    if delta < old_train_loss//10:
      count+=1
    else:
      count = 0
      
    if count > 9:
      break
      
  if opt.save_best:
    save_checkpoint(opt, best_model_wts, epoch, valid_loss)

writer.close()
