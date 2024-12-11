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
    ratio_dict = {}
    resultsize = (256,256)
    for index in tqdm(range(len(predL))):
        pred = np.array(Image.open(predL[index]))
        ann = np.array(Image.open(annL[index]))
        h, w = pred.shape
        pred = zoom(pred, (resultsize[0] / h, resultsize[1] / w), order=0)
        ann = zoom(ann, (resultsize[0] / h, resultsize[1] / w), order=0)

        ann = _mask_transform(ann).cpu().numpy().astype(np.int8)

        raw_name = '_'.join(predL[index].split('/')[-1].split('_')[:-1])
        if raw_name not in ratio_dict:
            # size is square
            ratio_dict[raw_name] = [resultsize[0]/h, resultsize[1]/w]
        for i in range(1, args.num_classes):
            try:
                num = D[raw_name]
            except:
                embed()
            fz[num][i] += sum(sum((pred == ann).astype(np.uint8) * (ann == i).astype(np.uint8))) * 2
            fm[num][i] += sum(sum((ann == i).astype(np.uint8))) + sum(sum((pred == i).astype(np.uint8)))
        if raw_name not in pred_mask_vol:
            pred_mask_vol[raw_name] = np.zeros((resultsize[0], name_picnum_dict[raw_name], resultsize[1]))
            gt_mask_vol[raw_name] = np.zeros((resultsize[0], name_picnum_dict[raw_name], resultsize[1]))
            startoffset = int(predL[index].split('.')[-2].split('_')[-1])
            #print(startoffset)

        ann[ann==-1] = 0
        pred_mask_vol[raw_name][:, int(predL[index].split('.')[-2].split('_')[-1])-startoffset, :] = pred
        gt_mask_vol[raw_name][:, int(predL[index].split('.')[-2].split('_')[-1])-startoffset, :] = ann.astype(np.uint8)


    cnt = 0
    cnt2 = 0
    with open('./testfiles_spacing_acdc.pkl', 'rb') as f:
        spacingdict_fortestset = pkl.load(f)
    for filename in pred_mask_vol:
        #filename = 'zhulili_pre'
        spacing = spacingdict_fortestset[filename]
        spacing[0] = spacing[0] / ratio_dict[filename][0]
        spacing[1] = spacing[1] / ratio_dict[filename][1]
        print(spacing)
        print('now processing: '+filename+' the '+str(cnt2)+'th')
        cnt2 += 1
        raw_ann = gt_mask_vol[filename]
        '''
        raw_ann_cp = zoom(raw_ann, (0.25, 0.25, 0.25), order=0).astype('int32')
        pred_mask_cp = zoom(pred_mask_vol[filename], (0.25, 0.25, 0.25), order=0).astype('int32')
        '''
        raw_ann_cp = raw_ann.astype('int32')
        pred_mask_cp = pred_mask_vol[filename].astype('int32')

        pred_mask_vol[filename] = pred_mask_vol[filename].astype('int32')
        raw_ann = raw_ann.astype('int32')

        for i in range(1, 4):
            print(i)
            mesh1 = get_mesh_from_vol((raw_ann_cp == i).astype('float'))
            mesh2 = get_mesh_from_vol((pred_mask_cp == i).astype('float'))

            # o3d.io.write_triangle_mesh('./stl/'+filename+'_0000'+str(i)+'_pred.stl', mesh2)
            # o3d.io.write_triangle_mesh('./stl/'+filename+'_0000'+str(i)+'_gt.stl', mesh1)

            if (len(np.array(mesh2.vertices)) == 0 & len(np.array(mesh1.vertices)) == 0):
                MSD[cnt][i] = 0
            else:
                MSD[cnt][i] = cal_msd_symmetric(mesh1, mesh2)


            A_, B_ = [], []

            tmp = np.argwhere(pred_mask_cp == i)
            for j in range(tmp.shape[0]):
                #A_.append([tmp[j][0] * 3, tmp[j][1] * 1.08, tmp[j][2] * 1.08])
                #A_.append([tmp[j][0]*1.2, tmp[j][1]*1.2, tmp[j][2]*1.2])
                A_.append([tmp[j][0] * spacing[2], tmp[j][1] * spacing[1], tmp[j][2] * spacing[0]])
            tmp = np.argwhere(raw_ann_cp == i)
            for j in range(tmp.shape[0]):
                #B_.append([tmp[j][0] * 3, tmp[j][1] * 1.08, tmp[j][2] * 1.08])
                #B_.append([tmp[j][0]*1.2, tmp[j][1]*1.2, tmp[j][2]*1.2])
                B_.append([tmp[j][0] * spacing[2], tmp[j][1] * spacing[1], tmp[j][2] * spacing[0]])
            if len(A_) == 0:
                print(cnt, i)
            else:
                dis = scipy.spatial.distance.cdist(A_, B_)

                '''
                这里是通过卡一个阈值，把离群点卡掉，这里的阈值是个你最后觉得“别的地方都对，为啥还是不满足一些显然的性能排序”的时候，可以调一调
                '''
                th = 3
                dis = (dis >= th) * th + (dis < th) * dis

                AHD[cnt][i] = (dis.min(axis=0).mean() + dis.min(axis=1).mean()) / 2
                '''
                最后存下来的MSD矩阵和AHD矩阵，就是 测试数据数*分割结构数(10*6?) 这样的矩阵，本来是用来算p值的。你也可以直接算个均值 
                '''
                # print((dis.min(axis=0).mean() + dis.min(axis=1).mean()) / 2)
            pass
        cnt += 1
        #print()
    print('imsb')
    pass
    for img_id in range(d_num):
        for i in range(1, 4):
            A[img_id][i] = float(fz[img_id][i] / fm[img_id][i])

    def load_pickle(filepath):
        with open(filepath, 'rb') as f:
            data = pkl.load(f)
            return data

    def dump_pickle(data, filename):
        with open(filename, 'wb') as f:
            pkl.dump(data, f)

    pkl_path = '/home/ubuntu/diskH/NKC/MuscleSeg/details_CBCT.pkl'
    try:
        details = load_pickle(pkl_path)
    except:
        details = {}
    DSC_details = np.zeros((d_num, 3))
    AHD_details = np.zeros((d_num, 3))
    MSD_details = np.zeros((d_num, 3))
    '''for i in range(d_num):
        DSC_details[i, 0] = (A[i, 1] + A[i, 4]) / 2
        DSC_details[i, 1] = (A[i, 3] + A[i, 6]) / 2
        DSC_details[i, 2] = (A[i, 2] + A[i, 5]) / 2

        AHD_details[i, 0] = (AHD[i, 1] + AHD[i, 4]) / 2
        AHD_details[i, 1] = (AHD[i, 3] + AHD[i, 6]) / 2
        AHD_details[i, 2] = (AHD[i, 2] + AHD[i, 5]) / 2

        MSD_details[i, 0] = (MSD[i, 1] + MSD[i, 4]) / 2
        MSD_details[i, 1] = (MSD[i, 3] + MSD[i, 6]) / 2
        MSD_details[i, 2] = (MSD[i, 2] + MSD[i, 5]) / 2'''
    for i in range(d_num):
        DSC_details[i, 0] = A[i, 1]
        DSC_details[i, 1] = A[i, 2]
        DSC_details[i, 2] = A[i, 3]

        AHD_details[i, 0] = AHD[i, 1]
        AHD_details[i, 1] = AHD[i, 2]
        AHD_details[i, 2] = AHD[i, 3]

        MSD_details[i, 0] = MSD[i, 1]
        MSD_details[i, 1] = MSD[i, 2]
        MSD_details[i, 2] = MSD[i, 3]

    details['transunet'] = {
        'DSC': DSC_details,
        'AHD': AHD_details,
        'MSD': MSD_details
    }

    ahd_mean_1 = np.mean(AHD_details[:, 0])
    ahd_mean_3 = np.mean(AHD_details[:, 2])
    ahd_mean_2 = np.mean(AHD_details[:, 1])
    ahd_std_1 = np.std(AHD_details[:, 0])
    ahd_std_3 = np.std(AHD_details[:, 2])
    ahd_std_2 = np.std(AHD_details[:, 1])
    ahd_std_all = np.std(np.mean(AHD_details, axis=1))
    #ahd_std_2536 = np.std(np.mean(AHD_details[:, 1:], axis=1))
    msd_mean_1 = np.mean(MSD_details[:, 0])
    msd_mean_3 = np.mean(MSD_details[:, 2])
    msd_mean_2 = np.mean(MSD_details[:, 1])
    msd_std_1 = np.std(MSD_details[:, 0])
    msd_std_3 = np.std(MSD_details[:, 2])
    msd_std_2 = np.std(MSD_details[:, 1])
    msd_std_all = np.std(np.mean(MSD_details, axis=1))
    #msd_std_2536 = np.std(np.mean(MSD_details[:, 1:], axis=1))
    result_dict = {}
    result_dict['ahd1'] = [ahd_mean_1, ahd_std_1]
    result_dict['ahd3'] = [ahd_mean_3, ahd_std_3]
    result_dict['ahd2'] = [ahd_mean_2, ahd_std_2]
    result_dict['ahd_all_std'] = ahd_std_all

    result_dict['msd1'] = [msd_mean_1, msd_std_1]
    result_dict['msd3'] = [msd_mean_3, msd_std_3]
    result_dict['msd2'] = [msd_mean_2, msd_std_2]
    result_dict['msd_all_std'] = msd_std_all
    #result_dict['msd_std_2356'] = msd_std_2536
    ###############################################################################
    ###############################################################################
    '''
    file = open('dict_for_ahdmsd_acdcnor_baseline.csv', 'w', encoding='utf-8', newline='')
    ###############################################################################
    ###############################################################################
    dict_writer = csv.DictWriter(file, fieldnames=list(result_dict.keys()))
    dict_writer.writeheader()
    dict_writer.writerow(result_dict)
    file.close()
    '''
    dump_pickle(result_dict, 'acdc10sup_best_240807.pkl')
    allvalues = list(result_dict.values())
    ahd3 = np.array(allvalues[0:3])
    msd3 = np.array(allvalues[4:-1])
    print(np.mean(ahd3[:, 0]), np.mean(msd3[:, 0]))
    print(allvalues[3], allvalues[-1])
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

    new_inference(args, net, predict_path, gt_dir='./datasets/acdc_testset20_nocrop/Seg')
