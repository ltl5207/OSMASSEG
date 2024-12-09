import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-120, 120)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() <= 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class ACDC70syn_dataset(Dataset):
    def __init__(self, base_dir='', list_dir='', split='train', transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            #fordbg
            #print('slice_name is'+slice_name)
            #width 512, height 509

            data_path = os.path.join(self.data_dir, 'JPEG', slice_name+'.jpg')
            im = Image.open(data_path)
            image = np.array(im)
            image_arr = image.astype(np.float) / float(255)
            label = Image.open(os.path.join(self.data_dir, 'Seg', slice_name+'.png'))
            label_arr = np.array(label)

        elif self.split == "test":
            test_slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, 'JPEG', test_slice_name + '.jpg')
            im = Image.open(data_path)
            image = np.array(im)
            image_arr = image.astype(np.float) / float(255)
            label = Image.open(os.path.join(self.data_dir, 'Seg', test_slice_name + '.png'))
            label_arr = np.array(label)

        elif self.split == "gantest":
            test_slice_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "JPEG/" + test_slice_name + '.jpg'
            # 这是给测GAN翻译结果用的
            data_path = os.path.join(self.data_dir, test_slice_name + '.jpg')
            im = Image.open(data_path).convert('L')  # .resize([512,509])
            # im = Image.open(data_path)

            label = Image.open(
                os.path.join('../../data/ACDC20test_nocrop/Seg', test_slice_name + '.png'))
            label_arr = np.array(label)
            im = im.resize(label.size)
            image_arr = np.array(im)
            image_arr = image_arr.astype(np.float) / float(255)

        sample = {'image': image_arr, 'label': label_arr}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
