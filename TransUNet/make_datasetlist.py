import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#小于1的时候分出来一部分做测试集
train_percent       = 1
trainval_percent    = 1
data_path      = '../data/ACDC70syn_099/'
data_path   = '../data/train9_my'
data_path = '../data/ACDC70syn_10deform_perref/'
list_path      = './lists/lists_ACDC70syn_099/'
list_path = './lists/lists_train9_my'
list_path = './lists/lists_ACDC70syn_10deform_perref'
#list_path      = './lists/lists_ACDC70syn_10deform_perref'
#data_path      = '/root/autodl-tmp/trans-unet/data/ACDC70syn_10deform_perref'

if __name__ == "__main__":
    random.seed(0)
    segfilepath     = os.path.join(data_path, 'Seg')
    saveBasePath    = list_path

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)
    num     = len(total_seg)
    list    = range(num)
    tr      = int(num*train_percent)
    train   = random.sample(list,tr)

    ftest       = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain      = open(os.path.join(saveBasePath, 'train.txt'), 'w')

    #写成txt文件
    for i in list:
        name = total_seg[i][:-4]+'\n'
        if i in train:
            ftrain.write(name)
        else:
            ftest.write(name)

    ftrain.close()
    ftest.close()
    print("Generate txt in ImageSets done.")
