""" import os
import glob
import numpy as np
import torch

def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2

    return (2. * intersect / union)

def choose_best_thr(opt, masks, out):
    print("choose best thresholde")

    dices = []
    thrs = np.arange(0.0, 1.0, 0.05)

    # masks = torch.from_numpy(masks).long()
    # preds = torch.from_numpy(out)
    print(type(masks))
    print(type(out))
    masks = masks.long()
    preds = out.long()

    masks_m = (masks>0.5).long()

    print(masks.shape)
    print(preds.shape)

    if opt.use_cuda and torch.cuda.is_available():
        use_cuda = True
        masks = masks.to('cuda')

    for th in thrs :
        print('Evaluating with threshold {}'.format(th))
        preds_m = (preds>th).long()

        if use_cuda :
            preds_m = preds_m.to('cuda')
        dice = dice_overall(preds_m, masks_m).mean()
        print('Dice coefficient is : {}'.format(dice))
        dices.append(dice)

    dices = np.array(dices)
    best_dice = dices.max()
    best_thrs = thrs[dices.argmax()]

    print('Best dice is {}'.format(best_dice))
    print('Best threshold is {}'.format(best_thrs))

    return best_thrs


def generate_mask(opt, out, best_thrs):

    out[out>best_thrs] = 1.0
    out[out<= best_thrs] = 0.0

    return out


     """