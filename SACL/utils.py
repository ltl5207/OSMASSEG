from __future__ import print_function
import argparse
import logging
import torchvision.transforms as transforms
import functools
from torch.nn import init
from packaging import version
import torch.nn.functional as F
import os
import random
from medpy import metric
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

#这个直接引进来吧
#from utils import test_single_volume
from models.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from collections import OrderedDict
"""This module contains simple helper functions """

import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision





#TU用的
def create_TU(TRAIN_PARALLEL_FLAG=False, flag=1, num_classes=4):
    vit_config = {'vit_name': 'R50-ViT-B_16', 'num_classes': num_classes, 'vit_patches_size': 16, 'n_skip': 3,
                  'img_size': 224, }

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
    if vit_config['vit_name'].find('R50') != -1:
        config_vit.patches.grid = (int(vit_config['img_size'] / vit_config['vit_patches_size']),
                                   int(vit_config['img_size'] / vit_config['vit_patches_size']))

    if flag == 1:
        net = ViT_seg(config_vit, img_size=vit_config['img_size'], num_classes=config_vit.n_classes, flag=1).cuda()
    elif flag == 2:
        net = ViT_seg(config_vit, img_size=vit_config['img_size'], num_classes=config_vit.n_classes, flag=2).cuda()

    return net


def create_and_load_TU(snapshot_path='',TRAIN_PARALLEL_FLAG=False, flag=1,num_classes=7):
    #snapshot_path = './transunet_epoch_149.pth'
    vit_config = {'vit_name':'R50-ViT-B_16', 'num_classes':num_classes, 'vit_patches_size':16, 'n_skip':3,
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

    if flag==1:
        net = ViT_seg(config_vit, img_size=vit_config['img_size'], num_classes=config_vit.n_classes, flag=1).cuda()
    elif flag==2:
        net = ViT_seg(config_vit, img_size=vit_config['img_size'], num_classes=config_vit.n_classes, flag=2).cuda()

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


# 数据集中用scipy的mapcoordinate做变形，训练时变形用torch内置的gridsample
# 两个api变形能力没问题，但是grid的格式不同
# 这个函数用来把mapcoordinate用的grid的格式转为gridsample用的
# 针对二维图对应的grid
# 如果是三维数据，参考https://blog.csdn.net/weixin_42139669/article/details/114983071
# 但这篇帖子貌似对mapcoordinate格点格式的理解也有问题，如果做三维需要再改
def map2sample(d_field):
    #size_tensor = torch.tensor(d_field).size()
    size_tensor = d_field.clone().detach().size()
    #d_field = d_field.cpu().detach().numpy()
    #sample = generate_grid(size_tensor[1:-1])
    #sample = torch.from_numpy(sample).unsqueeze(0)

    #d_field = np.rollaxis(d_field, 0, 3)
    #grid = d_field.permute(1,2,0)
    grid = d_field.permute(2,1,0)
    #grid = torch.from_numpy(d_field)
    #grid[:, :, 0] = (grid[:, :, 0] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
    #grid[:, :, 1] = (grid[:, :, 1] - ((size_tensor[0] - 1) / 2)) / (size_tensor[0] - 1) * 2

    # grid_sample是归一化的分数形式，map_coordinates是位移的绝对值
    grid[:, :, 0] = ((2*grid[:, :, 0]) / (size_tensor[2])) - 1
    grid[:, :, 1] = ((2*grid[:, :, 1]) / (size_tensor[1])) - 1

    #grid[:, :, 0] = (grid[:, :, 0] - ((size_tensor[2]-1)/2)) / (size_tensor[2] - 1) *2
    #grid[:, :, 1] = (grid[:, :, 1] - ((size_tensor[1]-1)/2)) / (size_tensor[1] - 1) *2 #(2 * grid[:, :, 1]) / size_tensor[1] - 1

    #grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
    return grid


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


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1,0
    # 这个是后加的，因为会有有的切片在gt里就没有这个类的情况，
    # 如果是原来的三维体数据，那每张体数据一定都是每类都有的，不会有这个情况
    # 你用了二维的切片，所以要改
    # 如果gt和pred都0，就标-1，然后算的时候把-1忽略
    elif pred.sum() == 0 and gt.sum() == 0:
        return -1,0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() #调.numpy()直接转成ndarray
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]: #放缩再放回去
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


#下面是GAN用的

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls

def tensor2imgray(input_image,imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

        Parameters:
            input_image (tensor) --  the input image tensor array
            imtype (type)        --  the desired type of the converted numpy array
        """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        #image_numpy = (image_numpy + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        #要不然没法在visdom上显示
        if image_numpy.shape[0] == 1:
            # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        #image_numpy = (image_numpy + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(np.squeeze(image_numpy))
    #image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)
