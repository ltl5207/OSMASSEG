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
from scipy.ndimage.interpolation import map_coordinates


class dataset_acdc_testset(Dataset):
    def __init__(self, base_dir=None, split='test', list_dir=None, device=None, transform=None):
        # self.root = opt.dataroot
        self.current_epoch = 0
        self.data_dir = base_dir
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.device = device
        self.transform = transform
        self.otherpeoplegtdir = '/root/autodl-tmp/doubleTU/datasets/acdc_testset20_nocrop/Seg'

    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, index):
        if self.split == 'train':
            slice_name = self.sample_list[index].strip('\n')
            domain_flag = slice_name.split('_')[-1]
            data_path = os.path.join(self.data_dir, 'JPEG', slice_name + '.jpg')
            im = Image.open(data_path)
            im_arr = np.array(im).astype(np.float) / float(255)
            im_arr_zoomed = zoom(im_arr, (224/509,224/512), order=3)
            #device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0]))
            assert im.mode == 'L'

            if domain_flag == 'A':
                im3 = im.convert('RGB')
                transform_forG = get_tansform_forG(opt=self.opt)

                with torch.no_grad():
                    im3_aG_T = self.G(torch.unsqueeze(transform_forG(im3), 0).to(self.device))

                im_aG = np.squeeze(util.tensor2imgray(im3_aG_T))
                im_aG = Image.fromarray(im_aG).convert('L')
                im_aG = zoom(np.array(im_aG).astype(np.float) / float(255) , (224/256,224/256), order=3)
                im_aW, angle = weak_aug(im_arr_zoomed)
                im_aS, d_field = strong_aug(input_pic=im_aG, alpha=im_aG.shape[1], sigma=im_aG.shape[1]*0.08)

                label = self.TU1(torch.nn.functional.interpolate(im3_aG_T, (224,224), mode='bicubic').to(self.device))

                label = torch.argmax(torch.softmax(label, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

            elif domain_flag == 'B':
                im_aW, angle = weak_aug(im_arr_zoomed)
                im_aS, d_field = strong_aug(im_arr_zoomed, alpha=im_arr_zoomed.shape[1], sigma=im_arr_zoomed.shape[1]*0.08)
                label = Image.open(os.path.join(self.data_dir, 'Seg', slice_name+'.png'))
                label = np.array(label)
                label = zoom(label, (224 / 509, 224 / 512), order=0)
            sample = {'im_aW': im_aW, 'im_aS': im_aS, 'angle': angle, 'd_field': d_field, 'label': label}
        elif self.split == 'test':
            slice_name = self.sample_list[index].strip('\n')
            data_path = os.path.join(self.data_dir, 'JPEG', slice_name + '.jpg')
            label_path = os.path.join(self.data_dir, 'Seg', slice_name+'.png')
            im = Image.open(data_path)
            la = Image.open(label_path)
            im_arr = np.array(im)
            la_arr = np.array(la)
            x, y = im_arr.shape
            image = im_arr #zoom(im_arr, (224/x, 224/y), order=3)
            label = la_arr #zoom(la_arr, (224/x, 224/y), order=0)
            sample = {'image':image, 'label':label}

        elif self.split == 'gantest':
            slice_name = self.sample_list[index].strip('\n')
            # modified for gan test
            data_path = os.path.join(self.data_dir, slice_name + '.jpg')
            label_path = os.path.join(self.otherpeoplegtdir, slice_name+'.png')

            la = Image.open(label_path)
            la_arr = np.array(la)
            im = Image.open(data_path).convert('L').resize(la.size)
            im_arr = np.array(im).astype(np.float) / float(255)

            #x, y = im_arr.shape
            image = im_arr #zoom(im_arr, (224/im_arr.shape[0], 224/im_arr.shape[1]), order=3)
            label = la_arr #zoom(la_arr, (224/la_arr.shape[0], 224/la_arr.shape[1]), order=0)
            sample = {'image':image, 'label':label}

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

        #if random.random() > 0.5:
        #    image, label = random_rot_flip(image, label)
        #elif random.random() > 0.5:

        #image, label = random_rotate(image, label)
        x, y = im_aW.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            im_aW = zoom(im_aW, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            im_aS = zoom(im_aS, (self.output_size[0] / x, self.output_size[1] / y), order=3)
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


def strong_aug(input_pic, alpha, sigma):
    assert len(input_pic.shape) == 2
    shape = input_pic.shape
    random_state = np.random.RandomState()
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x,y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    #indices = np.reshape(x+dx, (-1,1)), np.reshape(y+dy, (-1,1))
    #result = map_coordinates(input_pic, indices).reshape(shape)
    indices = np.array((x+dx,y+dy))
    result = map_coordinates(input_pic, indices)
    return result,indices



def weak_aug(image):
    angle = np.random.randint(-20, 20)
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