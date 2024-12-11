import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom
from models.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from collections import OrderedDict
from PIL import Image
import csv
from data.dataset_ABmixed import dataset_ABmixed
from data.dataset_pddca_testset9 import dataset_pddca_testset9
from data.dataset_acdc_testset import dataset_acdc_testset
import torch.nn as nn
from IPython import embed

TRAIN_PARALLEL_FLAG = False#True#

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str, default='acdc_testset20_nocrop', help='experiment_name')
parser.add_argument('--num_classes', type=int, default=7, help='output channel of network')
parser.add_argument('--list_dir', type=str, default='./lists/lists_otherpeople10_full', help='list dir')
parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
# parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--is_savenii', type=bool, default=True, help='whether to save results during inference')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

def dict2csv(dic, filename):
    """
    将字典写入csv文件，要求字典的值长度一致。
    :param dic: the dict to csv
    :param filename: the name of the csv file
    :return: None
    """
    file = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):   # 将字典逐行写入csv
        dic1 = {key: dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)
    file.close()

def generate_save_predresult(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
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
                if x != patch_size[0] or y != patch_size[1]:  # 放缩再放回去
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    for ind in range(prediction.shape[0]):
        imarr = prediction[ind]
        im = Image.fromarray(imarr.astype(np.uint8))
        im.save(os.path.join(test_save_path,case+'.png'))


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
    #比方说，5类，图像3x3,值为1分对了7个像素，在5x5的混淆矩阵里，7应该写在(1,1)的位置(从0开始)
    #扁平化时就是在下标为6的位置，机器算出这个6，(1*5)+1，就是了


def per_class_dice(hist):
    list1 = np.diag(hist)
    list2 = []
    for i,tp in enumerate(list1):
        list2.append(tp*2 / (sum(hist[i,:])+sum(hist[:,i])))
    return list2


def calculate_confusion_matrix_dice(args, pred_dir, label_dir, num_classes, net):
    hist = np.zeros((num_classes,num_classes))
    name_list = open(os.path.join(args.list_dir, 'test'+'.txt')).read().splitlines()

    def name_extract_func(str1):
        return '_'.join(str1.split('/')[-1].split('_')[:-1])

    # 为了算方差，要计算每个人的混淆矩阵
    name_clean = np.unique(list(map(name_extract_func, name_list))).tolist()
    hist_list = []
    for i in range(len(name_clean)):
        hist_list.append(np.zeros((num_classes, num_classes)))
    hist_groupby_individual = dict(zip(name_clean, hist_list))

    gt_imgs = [os.path.join(label_dir, x + ".png") for x in name_list]
    pred_imgs = [os.path.join(pred_dir, x + ".png") for x in name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        name = name_extract_func(gt_imgs[ind])
        temp = fast_hist(label.flatten(), pred.flatten(), num_classes)
        hist_groupby_individual[name] += temp

        hist += temp  # (label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 50 == 0:
            print('{:d} / {:d}: dice-{:0.2f}%'.format(
                    ind,
                    len(gt_imgs),
                    #验证了，这个实验里用不用nanmean其实没区别
                    100*np.nanmean(per_class_dice(hist)[1:])
                    #100 * np.mean(per_class_dice(hist))
                )
            )
    dice_groupby_individual = {}
    for _, key in enumerate(hist_groupby_individual.keys()):
        dice_groupby_individual[key] = per_class_dice(hist_groupby_individual[key])

    dicematrix = np.array(list(dice_groupby_individual.values()))

    stdlist = []
    # for acdc
    for i in range(dicematrix.shape[1]):
        stdlist.append(np.std(dicematrix[:, i]))

    stdmean = np.std(np.mean(dicematrix[:, 1:], axis=1))
    return hist, per_class_dice(hist), dice_groupby_individual, stdlist, stdmean


def inference(args, model, test_save_path=None, gtdir=''):
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        if len(sampled_batch['image'].shape) < 4:
            sampled_batch['image'] = sampled_batch['image'].unsqueeze(0)
            sampled_batch['label'] = sampled_batch['label'].unsqueeze(0)
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        generate_save_predresult(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size], test_save_path=test_save_path, case=case_name)
    hist_matrix,dice,hist_matrix_groupby_individual,stdlist, stdmean = calculate_confusion_matrix_dice(args, test_save_path, gtdir, args.num_classes, model)
    for i in  range(0, args.num_classes):
        print('Dice of %d class is %f'%(i,dice[i]))
    performance = np.mean(dice[1:args.num_classes])

    logging.info('Testing performance in other person\'s testset is: mean_dice : %f ' % (performance))
    return dice[1:args.num_classes], performance, stdlist, stdmean

import pickle as pkl

def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pkl.dump(data, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
        return data

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
        'pddca_testset9':{
            'Dataset': dataset_pddca_testset9,
            'volume_path': '/home/pc/diskB/liTianLin/doubleTU/datasets/pddca_testset9',
            'list_dir': '/home/pc/diskB/liTianLin/doubleTU/lists/lists_pddca_testset9',
            'num_classes': 7,
            'z_spacing': 1,
        },

        'acdc_testset20_nocrop':{
            'Dataset': dataset_acdc_testset,
            'volume_path': './datasets/acdc_testset20_nocrop',
            'list_dir': './lists/lists_acdc_nocrop',
            'num_classes': 4,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    args.exp = 'TU_acdc_10persup_v2'

    snapshot_path = './checkpoints/TU_acdc_10persup_v2/TU2/'
    snapshot_path_wo_epnum = './checkpoints/TU_acdc_10persup_v2/TU2/epoch_'
    snapshot_suffix = '.pth'

    beginning_epoch = 0
    endding_epoch = 150
    epnum_list = list(np.linspace(beginning_epoch, endding_epoch, endding_epoch-beginning_epoch ,endpoint=False).astype(int))
    # epnum_list = [25]
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    result_dict = {}
    #epnum_list = ['149_doubledeform_best8836']
    for _, epnum in enumerate(tqdm(epnum_list)):
        snapshot = snapshot_path_wo_epnum + str(epnum) + snapshot_suffix
        print(snapshot)
        if TRAIN_PARALLEL_FLAG:
            new_a = OrderedDict()
            # 重新加载会覆盖，应该不需要重新初始化net
            a = torch.load(snapshot)
            for k,v in a.items():
                name = k[7:]
                new_a[name] = v
            net.load_state_dict(new_a)
        else:
            net.load_state_dict(torch.load(snapshot))
        snapshot_name = snapshot.split('/')[-1].split('.')[0]
        log_folder = './test_log/test_log_' + args.exp
        os.makedirs(log_folder, exist_ok=True)
        logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logger=logging.getLogger()
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(snapshot_name)

        #要存，看结果
        if args.is_savenii:
            args.test_save_dir = './predictions'
            test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
            os.makedirs(test_save_path, exist_ok=True)
        else:
            test_save_path = None
        print(os.path.join(args.volume_path, 'Seg'))
        dice, performance, stdlist, stdmean = inference(args, net, test_save_path, gtdir=os.path.join(args.volume_path, 'Seg'))

        result_dict['{}'.format(epnum)] = dice,performance, stdlist, stdmean
    #dict2csv(result_dict, './performance_dict_acdc10persup_6_only90rot_w_initialization_winterdomgan.csv')
    dump_pickle(result_dict, 'performance_dict_acdc10persup_lasttry_90rotnoflip_woweight_winitialization.pkl')
    embed()
