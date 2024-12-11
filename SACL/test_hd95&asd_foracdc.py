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
from utils import test_single_volume
from models.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from IPython import embed
from scipy.ndimage.interpolation import zoom
import scipy
import pickle as pkl
from mesh_toolbox import *
import datetime
from PIL import Image
from data.dataset_pddca_testset9 import dataset_pddca_testset9
from data.dataset_acdc_testset import dataset_acdc_testset
import csv
from medpy import metric

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
        return data

def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./datasets/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='acdc_testset20', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--fold', type=int, default=None, help='fold')
args = parser.parse_args()

def _mask_transform(mask):
    # return torch.LongTensor(np.array(mask).astype('int32'))
    target = mask.astype('int32')
    target[target == 0] = -1
    return torch.from_numpy(target).long()
def new_inference(args, model, predict_path=None, gt_dir=None):
    model.eval()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    root = predict_path

    D, d_num = {}, 0
    predL = []
    annL = []
    for root, dirs, files in os.walk(root):
        for filename in sorted(files):
            predL.append(os.path.join(root, filename))
            if '_'.join(filename.split('_')[:-1]) not in D:
                D['_'.join(filename.split('_')[:-1])] = d_num
                d_num += 1

    for root, dirs, files in os.walk(gt_dir):
        for filename in sorted(files):
            annL.append(os.path.join(root, filename))

    print(D)
    A = np.zeros((d_num, args.num_classes))
    MSD = np.zeros((d_num, args.num_classes))
    AHD = np.zeros((d_num, args.num_classes))
    fz = np.zeros((d_num, args.num_classes))
    fm = np.zeros((d_num, args.num_classes))
    ####
    pred_vol, mask_vol = [], []
    pred_mask_vol = {}
    gt_mask_vol = {}
    for i in range(len(D)):
        pred_vol.append([])
        mask_vol.append([])
        for j in range(4):
            pred_vol[-1].append([])
            mask_vol[-1].append([])

    name_picnum_dict = {}
    for _, name in enumerate(predL):
        rawname = '_'.join(name.split('/')[-1].split('_')[:-1])
        if rawname not in name_picnum_dict:
            name_picnum_dict[rawname] = 1
        else:
            name_picnum_dict[rawname] += 1
    #ratio_dict = {}
    #resultsize = (256,256)
    for index in tqdm(range(len(predL))):
        pred = np.array(Image.open(predL[index]))
        ann = np.array(Image.open(annL[index]))
        h, w = pred.shape
        #pred = zoom(pred, (resultsize[0] / h, resultsize[1] / w), order=0)
        #ann = zoom(ann, (resultsize[0] / h, resultsize[1] / w), order=0)

        ann = _mask_transform(ann).cpu().numpy().astype(np.int8)

        raw_name = '_'.join(predL[index].split('/')[-1].split('_')[:-1])
        '''if raw_name not in ratio_dict:
            # size is square
            ratio_dict[raw_name] = resultsize[0]/h'''
        if raw_name not in pred_mask_vol:
            #pred_mask_vol[raw_name] = np.zeros((resultsize[0], name_picnum_dict[raw_name], resultsize[1]))
            #gt_mask_vol[raw_name] = np.zeros((resultsize[0], name_picnum_dict[raw_name], resultsize[1]))
            pred_mask_vol[raw_name] = np.zeros((pred.shape[0], name_picnum_dict[raw_name], pred.shape[1]))
            gt_mask_vol[raw_name] = np.zeros((pred.shape[0], name_picnum_dict[raw_name], pred.shape[1]))
            startoffset = int(predL[index].split('.')[-2].split('_')[-1])
            #print(startoffset)

        ann[ann==-1] = 0
        pred_mask_vol[raw_name][:, int(predL[index].split('.')[-2].split('_')[-1])-startoffset, :] = pred
        gt_mask_vol[raw_name][:, int(predL[index].split('.')[-2].split('_')[-1])-startoffset, :] = ann.astype(np.uint8)


    cnt = 0
    cnt2 = 0

    with open('./testfiles_spacing_acdc.pkl', 'rb') as f:
        spacingdict_fortestset = pkl.load(f)


    DSC = []
    ASD = []
    HD95 = []
    JAC = []
    l = list(pred_mask_vol.keys())
    for filename in tqdm(l):
        spacing = spacingdict_fortestset[filename]
        print(spacing)
        pred = pred_mask_vol[filename]
        gt = gt_mask_vol[filename]
        dsc = []
        asd = []
        hd95 = []
        jac = []
        for i in range(1,4):
            x = (pred == i)
            y = (gt == i)
            x[x>0] = 1
            y[y>0] = 1
            dsc.append(metric.binary.dc(x, y))
            asd.append(metric.binary.asd(x, y, spacing))
            hd95.append(metric.binary.hd95(x, y, spacing))
            jac.append(metric.binary.jc(x, y))
        DSC.append(dsc)
        ASD.append(asd)
        HD95.append(hd95)
        JAC.append(jac)

    d = np.asarray(DSC)
    a = np.asarray(ASD)
    h = np.asarray(HD95)
    j = np.asarray(JAC)
    print(np.mean(np.mean(d, axis=0)), np.mean(np.mean(a, axis=0)), np.mean(np.mean(h, axis=0)))


    dump_pickle({'dsc':d, 'asd':a, 'hd95':h, 'jaccard':j}, 'acdc10sup_best_240807_dah.pkl')
    embed()


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'pddca_testset9': {
            'Dataset': dataset_pddca_testset9,
            'volume_path': '/home/pc/diskB/liTianLin/doubleTU/datasets/pddca_testset9',
            'list_dir': '/home/pc/diskB/liTianLin/doubleTU/lists/lists_pddca_testset9',
            'num_classes': 7,
            'z_spacing': 1,
        },
        'acdc_testset20': {
            'Dataset': dataset_acdc_testset,
            'volume_path': '/home/pc/diskB/liTianLin/doubleTU/datasets/acdc_testset20',
            'list_dir': '/home/pc/diskB/liTianLin/doubleTU/lists/lists_acdc',
            'num_classes': 4,
            'z_spacing': 1,
        }
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot_final = '/root/autodl-tmp/doubleTU/checkpoints/TU_acdc_10persup/TU2/best_epoch_149.pth'

    pred_for_acdc_10sup = '/root/autodl-tmp/doubleTU/predictions/TU_acdc_10persup/epoch_149'
    #pred_for_acdc_10sup = '/root/autodl-tmp/doubleTU/predictions/TU_acdc_10persup_v2/epoch_149_doubledeform_best8836'

    #########################
    #########################
    snapshot = snapshot_final
    #########################
    #########################
    net.load_state_dict(torch.load(snapshot))
    print(snapshot)
    snapshot_name = snapshot.split('/')[-3]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    ##########################
    ##########################
    predict_path = pred_for_acdc_10sup
    ##########################
    ##########################

    #new_inference(args, net, predict_path, gt_dir='./datasets/acdc_testset20/Seg')
    new_inference(args, net, predict_path, gt_dir='./datasets/acdc_testset20_nocrop/Seg')
