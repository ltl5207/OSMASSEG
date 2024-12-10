import argparse
import logging
import torchvision.transforms as transforms
import functools
from torch.nn import init
from packaging import version
import torch.nn.functional as F
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
from torch.nn.modules.loss import CrossEntropyLoss
from PIL import Image

#这个直接引进来吧
#from utils import test_single_volume
from models.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from collections import OrderedDict

#dataset在这里其实不是特别需要，一会看着办
#from datasets.dataset_syn40 import syn40_dataset, RandomGenerator

from PIL import Image


device = torch.device('cuda:0')
pre_process = 'resize_and_crop'
method = Image.BICUBIC
o = 286
osize = [o, o]
c = 256
nce_layers = [0,4,8,12,16]
num_patches = 256

#transunet的dice损失
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


#加载训练好的transunet模型
def create_and_load_TU(snapshot_path='',TRAIN_PARALLEL_FLAG=False):
    #snapshot_path = './transunet_epoch_149.pth'
    print(snapshot_path)
    vit_config = {'vit_name':'R50-ViT-B_16', 'num_classes':4, 'vit_patches_size':16, 'n_skip':3,
                  'img_size':224, }

    random_seed = 1234
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


    config_vit = CONFIGS_ViT_seg[vit_config['vit_name']]
    config_vit.n_classes = vit_config['num_classes']
    config_vit.n_skip = vit_config['n_skip']
    config_vit.patches.size = (vit_config['vit_patches_size'], vit_config['vit_patches_size'])
    if vit_config['vit_name'].find('R50') !=-1:
        config_vit.patches.grid = (int(vit_config['img_size']/vit_config['vit_patches_size']), int(vit_config['img_size']/vit_config['vit_patches_size']))
    net = ViT_seg(config_vit, img_size=vit_config['img_size'], num_classes=config_vit.n_classes).cuda()

    if TRAIN_PARALLEL_FLAG:
        new_a = OrderedDict()
        a = torch.load(snapshot_path)
        for k, v in a.items():
            name = k[7:]
            new_a[name] = v
        net.load_state_dict(new_a)
    else:
        net.load_state_dict(torch.load(snapshot_path))

    return net

def freeze(net):
    for param in net.parameters():
        param.requires_grad = False