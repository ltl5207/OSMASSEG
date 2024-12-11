import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#小于1的时候分出来一部分做测试集
train_percent       = 1
trainval_percent    = 1
data_path      = './datasets/ABmixed_acdc_10percentsup_doubledeform/'
list_path      = './lists/lists_acdc_10percentsup_doubledeform/'
os.makedirs(list_path, exist_ok=True)
if __name__ == "__main__":
    random.seed(0)
    segfilepath     = os.path.join(data_path, 'Seg')
    jpgfilepath     = os.path.join(data_path, 'JPEG')
    saveBasePath    = list_path

    temp_seg = os.listdir(segfilepath)
    temp_data = os.listdir(jpgfilepath)
    num     = len(temp_data)
    list    = range(num)
    #list    = random.sample(list, num)
    tr      = int(num*train_percent)
    train   = random.sample(list,tr)

    ftest       = None # open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain      = open(os.path.join(saveBasePath, 'train.txt'), 'w')

    #写成txt文件
    for i in list:
        name = temp_data[i][:-4]+'\n'
        if i in train:
            ftrain.write(name)
        else:
            ftest.write(name)
    if ftrain is not None:
        ftrain.close()
    if ftest is not None:
        ftest.close()
    print("Generate txt in ImageSets done.")
