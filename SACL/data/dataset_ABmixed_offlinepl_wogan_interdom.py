import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import utils as util
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
from scipy.ndimage.interpolation import map_coordinates
import random

#强变换给学生_合成集分割_TU1，所以需要if A: output: im_G_aS
#弱变换给教师_临床集分割_TU2，所以需要if B: output: im_aW

class dataset_ABmixed(Dataset):
    def __init__(self, opt, segopt='Seg', transform=None, G=None, TU1=None, device=None):
        self.G = G
        self.TU1 = TU1
        #
        # ~不要了，get_item里不用推理plabel了，直接复制detach的map1~
        self.opt = opt
        # self.root = opt.dataroot
        self.current_epoch = 0
        self.data_dir = opt.root_path
        self.transform = transform
        self.split = 'train' if opt.isTrain else 'test'
        self.sample_list = open(os.path.join(opt.list_dir, self.split + '.txt')).readlines()
        self.device = device
        self.segopt = segopt

    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, index):
        if self.split == 'train':
            slice_name = self.sample_list[index].strip('\n')
            domain_flag = slice_name.split('_')[-1]
            data_path = os.path.join(self.data_dir, 'JPEG', slice_name + '.jpg')
            im = Image.open(data_path)
            w, h = im.size
            im_arr = np.array(im).astype(np.float) / float(255)
            im_arr_zoomed = zoom(im_arr, (224/h, 224/w), order=3)
            #device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0]))
            #assert im.mode == 'L'
            label = Image.open(os.path.join(self.data_dir, self.segopt, slice_name + '.png'))
            label = np.array(label)
            label = zoom(label, (224/h, 224/w), order=0)

            '''if random.random() > 0.5:
                image, label = random_rot_flip(im_arr_zoomed, label)
            elif random.random() > 0.5:
                image, label = random_rotate(im_arr_zoomed, label)'''

            if domain_flag == 'A':
                im_aW, angle = weak_aug(im_arr_zoomed)

                d_field = strong_aug(input_pic=im_arr_zoomed, alpha=im_arr_zoomed.shape[1], sigma=im_arr_zoomed.shape[1] * 0.08)
                d_field = util.map2sample(torch.from_numpy(d_field))

                im_aS = F.grid_sample(torch.from_numpy(im_arr_zoomed).unsqueeze(0).unsqueeze(0), d_field.unsqueeze(0),
                                      padding_mode='zeros', mode='nearest', align_corners=True).squeeze().numpy()

            elif domain_flag == 'B':
                im_aW, angle = weak_aug(im_arr_zoomed)
                d_field = strong_aug(im_arr_zoomed, alpha=im_arr_zoomed.shape[1], sigma=im_arr_zoomed.shape[1]*0.08)
                d_field = util.map2sample(torch.from_numpy(d_field))
                im_aS = F.grid_sample(torch.from_numpy(im_arr_zoomed).unsqueeze(0).unsqueeze(0), d_field.unsqueeze(0),
                                      padding_mode='zeros', mode='nearest', align_corners=True).squeeze().numpy()

        sample = {'im_aW':im_aW, 'im_aS':im_aS, 'angle':angle, 'd_field':d_field, 'label':label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = slice_name
        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        #im_aW, im_aS, label
        im_aW, im_aS, label = sample['im_aW'], sample['im_aS'], sample['label']


        # 0号给合成tu1，1号给临床tu2
        x, y = im_aW.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            im_aW = zoom(im_aW, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            #im_aW[1] = zoom(im_aW[1], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            im_aS = zoom(im_aS, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            #im_aS[1] = zoom(im_aS[1], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        im_aW = torch.from_numpy(im_aW.astype(np.float32)).unsqueeze(0)
        im_aS = torch.from_numpy(im_aS.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        #sample = {'im_aW': im_aW, 'im_aS': im_aS, 'angle': angle, 'd_field': d_field, 'label': label}
        #sample = {'im_aW': im_aW, 'im_aS': im_aS, 'label': label.long(), 'angle': sample['angle'], 'd_field': sample['d_field']}
        sample['im_aW'], sample['im_aS'], sample['label'] = im_aW, im_aS, label.long()
        return sample


def get_tansform_forG(opt, method=Image.BICUBIC, convert=True):
    # opt沿用了GAN的参数
    # 重复给GAN的预处理还是用opt，不写死
    transform_list = []
    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, method))
    #不加flip
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)

#去你吗的map_coordinate, 老子不用了
#....
def strong_aug(input_pic, alpha, sigma):
    assert len(input_pic.shape) == 2
    shape = input_pic.shape
    random_state = np.random.RandomState()
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x,y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    #indices = np.reshape(x+dx, (-1,1)), np.reshape(y+dy, (-1,1))
    #result = map_coordinates(input_pic, indices).reshape(shape)
    #indices = np.array((x+dx,y+dy))
    indices = np.array((y+dy, x+dx))
    #result = map_coordinates(input_pic, indices)
    return indices



def weak_aug(image, angle=None):
    if angle == None:
        angle = np.random.randint(-90, 90)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    #label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, angle


def generate_grid(imgshape):
    '''
    生成适用于map_coordinates() 的通用网格（二维图）
    '''
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    #z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(y, x)), 0, 3)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid


#这个没用了，就当是备份，改完的要用的放在了util.py
def map2sample(flow):
    '''
    flow:batch_size,H,W,D,C(tensor)
    note:flow为位移场
    这个是三维图的变形场转换，要改
    '''
    size_tensor = flow.size()
    sample = generate_grid(size_tensor[1:-1])
    sample = torch.from_numpy(sample).unsqueeze(0)
    grid = flow + sample
    grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
    grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
    grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
    return grid

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label