'''
the new utils which is used by my new program
'''
import argparse
import logging
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
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from collections import OrderedDict
from datasets.dataset_syn40 import syn40_dataset, RandomGenerator
from PIL import Image
import torchvision.transforms as transforms

def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)

    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    #比方说，5类，矩阵3x3,值为1分对了7个像素，在5x5的混淆矩阵里，7应该写在(1,1)的位置(从0开始)
    #扁平化时就是在下标为6的位置，机器算出这个6，(1*5)+1，就是了


def per_class_dice(hist):
    list1 = np.diag(hist)
    list2 = []
    for i,tp in enumerate(list1):
        list2.append(tp*2 / (sum(hist[i,:])+sum(hist[:,i])))
    return list2


def cauculate_confusion_matrix_dice(args, pred_dir, label_dir, num_classes, net):
    hist = np.zeros((num_classes,num_classes))
    name_list = open(os.path.join(args.list_dir, 'test_vol'+'.txt')).read().splitlines()

    gt_imgs = [os.path.join(label_dir, x + ".png") for x in name_list]
    pred_imgs = [os.path.join(pred_dir, x + ".png") for x in name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 50 == 0:
            print('{:d} / {:d}: dice-{:0.2f}%'.format(
                    ind,
                    len(gt_imgs),
                    100*np.nanmean(per_class_dice(hist))
                )
            )

    return hist,per_class_dice(hist)


def generate_save_predresult(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            #这个就是加的
            transform = transforms.Compose([transforms.Normalize((0.5), (0.5))])
            input = transform(input)

            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:  # 放缩再放回去
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    for ind in range(prediction.shape[0]):
        imarr = prediction[ind]
        im = Image.fromarray(imarr.astype(np.uint8))
        im.save(os.path.join(test_save_path,case+'.png'))


def generate_save_predresult_for3channel(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        #for ind in range(image.shape[0]):
        slice = image
        x, y = slice.shape[1], slice.shape[2]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda() #删掉了一个squeeze(0)
        #这个就是加的
        transform = transforms.Compose([transforms.Normalize((0.5), (0.5))])
        input = transform(input)

        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:  # 放缩再放回去
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
        prediction[0] = pred
    for ind in range(prediction.shape[0]):
        imarr = prediction[ind]
        im = Image.fromarray(imarr.astype(np.uint8))
        im.save(os.path.join(test_save_path,case+'.png'))