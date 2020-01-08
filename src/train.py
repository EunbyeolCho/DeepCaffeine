import os
import tensorflow as tf
from time import sleep
from options import args
import time
from data_loader.data_loader import get_data_loader
from models import unet
import torch.nn as nn


def trainer(opt, model, optimizer, data_loader, loss_criterion):
  print('====Training====')
  
  start_time = time.time()

  total_loss = 0.0
  
  for i, batch in enumerate(data_loader) :
    
    img, masks = batch[0], batch[1]

    if opt.use_cuda :
      img = img.to(opt.device, dtype = torch.float)
      masks = masks.to(opt.device, dtype = torch.float)
      optimizer.zero_grad()

    out = model(img)

    loss = loss_criterion(out, masks)
    loss.backward()
    optimizer.step()

    total_loss +=loss.item()

  total_loss = total_loss/i

  print("***\nTraining %.2fs => Epoch[%d/%d] :: Loss : %.10f\n"%(time.time()-start_time, opt.epoch_num, opt.n_epochs, total_loss)) 
  
  return total_loss
  


def evaluator(opt, model, data_loader, loss_criterion):

  return total_loss



if __name__ == "__main__":

  opt = args
  print(opt)

  train_data_loader, valid_data_loader = get_data_loader(opt)
  
  if not os.path.exists(opt.log_dir)
    os.makedirs(opt.log_dir)

  log_file = os.path.join(opt.log_dir, '%s_log.csv'%(opt.model))

  if opt.model == 'unet' :
    #net = unet(opt)
    net = unet.UNet(6) #채송: 6은 num_classes! 
    
  L2_criterion = nn.MSELoss()
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
      L2_criterion = L2_criterion.to(opt.device)
  
  print('===> Setting Optimizer')
  optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))


  for epoch in range(opt.n_epochs):
    opt.epoch_num = epoch
    train_loss = trainer(opt, net, optimizer, train_data_loader, loss_criterion = L2_criterion)
    # valid_loss = evaluator(opt, net, valid_data_loader, loss_criterion = L2_criterion)
