import torch
from tqdm import tqdm
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

from train_acdc_10sup import  load_and_freeze_TU
from train_acdc_10sup import  load_and_freeze_G

def get_tansform_forG(load_size, method=Image.BICUBIC):
    transform_list = []
    osize = [load_size, load_size]
    transform_list.append(transforms.Resize(osize, method))
    #不加flip
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)


# 也可以在线生成
if __name__ == '__main__':
    mode = 'wgan'
    #mode = 'wogan'
    device = torch.device('cuda:0')
    jpdir = './datasets/ABmixed_acdc_10percentsup/JPEG'
    pndir = './datasets/ABmixed_acdc_10percentsup/Seg'
    #pndir = '/home/pc/diskB/liTianLin/doubleTU/datasets/ABmixed_acdc/Seg_woGAN'
    with_gt_namelist = list(map(lambda x: x.split('.')[0], os.listdir(pndir)))
    wo_gt_namelist = list(set(map(lambda x: x.split('.')[0], os.listdir(jpdir))) - set(with_gt_namelist))
    flag = np.unique(np.array(list(map(lambda x: x.split('_')[-1], wo_gt_namelist))))

    assert len(flag)==1 and str(flag[0])=='A'

    if mode == 'wogan':
        TU = load_and_freeze_TU(freeze_flag=True, snapshot_path='./TU_acdc_10percent_ref_ep69.pth', num_classes=4)
        for _, name in enumerate(tqdm(wo_gt_namelist)):
            data_path = os.path.join(jpdir, name + '.jpg')
            im = Image.open(data_path)
            w, h = im.size
            im_arr = np.array(im).astype(np.float) / float(255)

            im_arr = zoom(np.array(im_arr).astype(np.float) / float(255), (224 / h, 224 / w), order=3)

            label1 = TU(torch.unsqueeze(torch.unsqueeze(torch.tensor(im_arr), 0), 0).to(device).float())
            label1 = torch.argmax(torch.softmax(label1, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
            label1_zoomed_back = zoom(label1, (h / 224, w / 224), order=0)
            label_final = Image.fromarray(label1_zoomed_back.astype(np.uint8))
            assert im.size == label_final.size
            label_final.save(os.path.join(pndir, name + '.png'))
    else:
        TU = load_and_freeze_TU(freeze_flag=True, snapshot_path='./TU_acdc_10percent_ref_ep69.pth', num_classes=4)
        G = load_and_freeze_G(pth_path='acdc_netG_10perv4_25.pth', device=torch.device('cuda:0'))

        for _, name in enumerate(tqdm(wo_gt_namelist)):
            data_path = os.path.join(jpdir, name + '.jpg')
            im = Image.open(data_path)
            w, h = im.size
            im_arr = np.array(im).astype(np.float) / float(255)
            #im_arr_zoomed = zoom(im_arr, (224 / 509, 224 / 512), order=3)
            assert im.mode == 'L'
            im3 = im.convert('RGB')
            transform_forG = get_tansform_forG(load_size=256)

            with torch.no_grad():
                im3_aG_T = G(torch.unsqueeze(transform_forG(im3), 0).to(device))

            im_aG = np.squeeze(util.tensor2imgray(im3_aG_T))
            im_aG = Image.fromarray(im_aG).convert('L')
            im_aG = zoom(np.array(im_aG).astype(np.float) / float(255), (224 / 256, 224 / 256), order=3)

            label1 = TU(torch.unsqueeze(torch.unsqueeze(torch.tensor(im_aG),0),0).to(device).float())
            label1 = torch.argmax(torch.softmax(label1,dim=1),dim=1).squeeze(0).cpu().detach().numpy()
            label1_zoomed_back = zoom(label1, (h / 224, w / 224), order=0)
            label_final = Image.fromarray(label1_zoomed_back.astype(np.uint8))
            assert im.size == label_final.size
            label_final.save(os.path.join(pndir,name+'.png'))



    '''
    import os                                       
    >>> namelist = os.listdir('.')                      
    >>> for _,name in enumerate(namelist):              
    ...     os.rename(name, name.split('.')[0]+'_B.jpg')
    ... 
    >>> exit()
    '''