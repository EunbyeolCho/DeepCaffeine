import torch
import os

def save_checkpoint(opt, net, epoch, loss):

    checkpoint_dir = opt.weight_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir,  "models_epoch_%04d_loss_%.10f.pth"%(epoch, loss))

    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        state = {'epoch': epoch, 'net': net.module}
    else :
        state = {'epoch' : epoch, 'net': net}

    torch.save(state, checkpoint_dir)
    print("Checkpoint saved to {}".format(checkpoint_dir))
