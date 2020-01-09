from models import unet
import os
import glob
import torch

def load_model(checkpoint_dir):
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    checkpoint_list.sort()

    loss_list = list(map(lambda x: float(os.path.basename(x).split('_')[4][:-4]), checkpoint_list))
    best_loss_idx = loss_list.index(min(loss_list))
    checkpoint_path = checkpoint_list[best_loss_idx]

    if opt.model == 'unet' :
        net = unet.UNet(6) #채송: 6은 num_classes! 

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        
        n_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'].state_dict())
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_path, n_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        n_epoch = 0

    return n_epoch + 1, net