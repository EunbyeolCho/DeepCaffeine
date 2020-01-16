from models import unet
import os
import glob
import torch

def load_model(opt, checkpoint_dir):
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    checkpoint_list.sort()

    # 은별 : return n_epoch +1 이라고 했는데 n_epoch이 정의x
    n_epoch = 0

    loss_list = list(map(lambda x: float(os.path.basename(x).split('_')[4][:-4]), checkpoint_list))
    best_loss_idx = loss_list.index(min(loss_list))
    checkpoint_path = checkpoint_list[best_loss_idx]

    if opt.model == 'unet' :
        net = unet.UNet(opt.num_class + 1) 

    if os.path.isfile(checkpoint_path):
        # print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        
        n_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'].state_dict())
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_path, n_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        # 은별 : 0으로 바꿀 필요는 없을 것 같아
        # n_epoch = 0

    return n_epoch + 1, net