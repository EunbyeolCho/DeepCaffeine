import argparse
import os
import torch

ID = os.environ['ID']
ID = str(ID)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './data')))



train_dir = '/data/train'
log_dir = '/data/volume/logs'
test_dir = '/data/test'
output_dir = '/data/output'
volume_dir = '/data/volume'
weight_dir = '/data/volume/logs' + ID + '_final.hdf5'

#예진 local_test
# train_dir = './data/train'
# log_dir = './data/volume/logs'
# test_dir = './data/test'
# output_dir = './data/output'
# volume_dir = './data/volume'
# weight_dir = './data/volume/logs' + '_final.hdf5'


parser = argparse.ArgumentParser(description = 'HeLP Challenge 2019 Cardiovascular')

parser.add_argument('--train_dir', type = str, default = train_dir)
parser.add_argument('--log_dir', type = str, default = log_dir)
parser.add_argument('--test_dir', type = str, default = test_dir)
parser.add_argument('--output_dir', type = str, default = output_dir)
parser.add_argument('--volume_dir', type = str, default = volume_dir)
parser.add_argument('--weight_dir', type = str, default = weight_dir)


parser.add_argument('--use_cuda', type = bool, default = True)
parser.add_argument('--device', type = str, default = 'cpu',
                    help = 'cpu, cuda')
parser.add_argument('--seed', type = int, default = 1,
                    help = 'random seed 고정 위해서 넣음')
parser.add_argument('--multi-gpu', type = bool, default = True)


parser.add_argument('--mode', type = str, default = 'train',
                    help = 'train, test')
parser.add_argument('--model', type = str, default = 'unet',
                    help = 'unet, maskRcnn, deeplabv3p')
parser.add_argument('--n_epochs', type = int, default = 50,
                    help = 'max epochs')
parser.add_argument('--epoch_num', type = int, default = 0,
                    help = 'real time epoch')
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument("--batch_size", type = int, default = 32)
parser.add_argument('--b1', type = float, default = 0.9,
                    help = 'Adam : decay of first order momentum of gradient')
parser.add_argument('--b2', type = float, default = 0.999,
                    help = 'Adam : decay of second order momentum of gradient')
parser.add_argument('--loss_weight', type = float, default = 20)
parser.add_argument('--loss', type = str, default = 'dice',
                    help = 'bce, ce, wce, focal, dice')

# data loader argument
parser.add_argument('--img_size', type = int, default = 256,
                    help = '256, 512, ...')
parser.add_argument('--histo_equal', type = bool, default=True)
parser.add_argument('--augmentation', type = bool, default = False,
                    help = 'augmentation (flip)augmentation fro training set')
parser.add_argument('--num_class', type = int, default = 8)

#채송: save_best 추가
parser.add_argument('--save_best', type = bool, default = False, help = 'you can save only the best model')
parser.add_argument('--train_ratio', type =float, default = 0.8,
                    help = 'ratio of trainset/dataset, trainset : validset = train_ratio : 1-train_ratio')


args = parser.parse_args()

torch.manual_seed(args.seed)
