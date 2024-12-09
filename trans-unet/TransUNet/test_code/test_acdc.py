# this code is used to select the best performance of the translation model for promoting the pseudo-label quality
import sys
sys.path.append('../')
import random
from PIL import Image
import numpy as np
import logging
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from collections import OrderedDict
from IPython import embed
from datasets.dataset_synct40_v2_gan import synct40_v2_dataset, RandomGenerator
from datasets.dataset_synct40_gan import synct40_dataset, RandomGenerator
from datasets.dataset_syn40_gan import syn40_dataset,RandomGenerator
from datasets.dataset_newsyn40_gan import newsyn40_dataset,RandomGenerator
from datasets.dataset_syn40_chenxi import syn40_chenxi_dataset
from datasets.dataset_pddca_syn36 import pddca_syn36_dataset
from datasets.dataset_ACDC70syn import ACDC70syn_dataset
# need to be modified to the dir of the translation model(GAN)
sys.path.append('../../../translation/')
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import argparse

def load_and_save_results(destdir,
                          epoch,
                          checkpdir='',
                          name=''):
    '''
    destdir and epoch will be combined into the inference result path of an exact gan model
    modeldir is used to locate the gan models stored in pth file
    '''

    #extract the name
    #namepool = modeldir.split('/')
    #name = namepool[-2] if namepool[-1]=='' else namepool[-1]

    opt = TestOptions().parse()  # get test options
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    # hard-code some parameters for test
    opt.epoch = epoch
    opt.results_dir = destdir
    opt.checkpoints_dir = checkpdir
    #opt.gpu_ids = 0
    opt.dataroot = '../../../triple/datasets/acdc_testset20_nocrop'
    opt.name = name
    opt.CUT_mode = 'CUT'
    opt.phase = 'test'
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.display_winsize = 512
    # opt.netF_nc=512
    # opt.no_flip=True
    opt.num_test = 3000
    opt.gpu_ids = [0]
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)  # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML


def compute_dice_given_dir(image_dir,gtdir):
    TRAIN_PARALLEL_FLAG = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_path', type=str, default='../data/Synapse/test_vol_h5',help='root dir for validation volume data')  # for acdc volume_path=root_dir
    parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8,help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--dataset', type=str, default='ACDC70syn_10percent_ref_only20rot') # 'ACDC70syn_099_90rot')
    #parser.add_argument('--dataset', type=str, default='synct40_v2')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    args = parser.parse_args()

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
            im.save(os.path.join(test_save_path, case + '.png'))

    def fast_hist(a, b, n):
        # --------------------------------------------------------------------------------#
        #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
        # --------------------------------------------------------------------------------#
        k = (a >= 0) & (a < n)

        # --------------------------------------------------------------------------------#
        #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        #   返回中，写对角线上的为分类正确的像素点
        # --------------------------------------------------------------------------------#
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
        # 比方说，5类，矩阵3x3,值为1分对了7个像素，在5x5的混淆矩阵里，7应该写在(1,1)的位置(从0开始)
        # 扁平化时就是在下标为6的位置，机器算出这个6，(1*5)+1，就是了

    def per_class_dice(hist):
        list1 = np.diag(hist)
        list2 = []
        for i, tp in enumerate(list1):
            list2.append(tp * 2 / (sum(hist[i, :]) + sum(hist[:, i])))
        return list2

    def cauculate_confusion_matrix_dice(args, pred_dir, label_dir, num_classes, net):

        def name_extract_func(str1):
            return '_'.join(str1.split('/')[-1].split('_')[:-1])

        hist = np.zeros((num_classes, num_classes))
        name_list = open(os.path.join(args.list_dir, 'gantest' + '.txt')).read().splitlines()
        # 为了算方差，要计算每个人的混淆矩阵
        name_clean = np.unique(list(map(name_extract_func, name_list))).tolist()
        hist_list = []
        for i in range(len(name_clean)):
            hist_list.append(np.zeros((num_classes, num_classes)))
        hist_groupby_individual = dict(zip(name_clean, hist_list))
        gt_imgs = [os.path.join(label_dir, x + ".png") for x in name_list]
        pred_imgs = [os.path.join(pred_dir, x + ".png") for x in name_list]
        # flag = 1
        for ind in range(len(gt_imgs)):
            pred = np.array(Image.open(pred_imgs[ind]))
            label = np.array(Image.open(gt_imgs[ind]))
            name = name_extract_func(gt_imgs[ind])
            temp = fast_hist(label.flatten(), pred.flatten(), num_classes)
            # temp_avg = per_class_dice(temp)
            # hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
            hist_groupby_individual[name] += temp

            '''
            if ind == 0:
                per_dice = temp_avg

            else:
                per_dice.concatenate(temp_avg)
            '''

            hist += temp
            # 0，每人的方差
            # 1，每个数据（所有人的所有切片）的方差
            if ind > 0 and ind % 50 == 0:
                print('{:d} / {:d}: dice-{:0.2f}%'.format(
                    ind,
                    len(gt_imgs),
                    # 验证了，这个实验里用不用nanmean其实没区别
                    100 * np.nanmean(per_class_dice(hist)[1:])
                    # 100 * np.mean(per_class_dice(hist))
                )
                )

        dice_groupby_individual = {}
        for _, key in enumerate(hist_groupby_individual.keys()):
            dice_groupby_individual[key] = per_class_dice(hist_groupby_individual[key])

        dicematrix = np.array(list(dice_groupby_individual.values()))
        # for cbct and pddca
        '''dicematrix3 = np.zeros((dicematrix.shape[0], 3))
        dicematrix3[:, 0] = (dicematrix[:, 1] + dicematrix[:, 4]) / 2
        dicematrix3[:, 1] = (dicematrix[:, 2] + dicematrix[:, 5]) / 2
        dicematrix3[:, 2] = (dicematrix[:, 3] + dicematrix[:, 6]) / 2
        stdlist3 = [np.std(dicematrix3[:, 0]), np.std(dicematrix3[:, 1]), np.std(dicematrix3[:, 2])]
        stdlist6 = [np.std(dicematrix[:, 1]), np.std(dicematrix[:, 2]), np.std(dicematrix[:, 3]),
                    np.std(dicematrix[:, 4]), np.std(dicematrix[:, 5]), np.std(dicematrix[:, 6])]'''

        stdlist = []
        # for acdc
        for i in range(dicematrix.shape[1]):
            stdlist.append(np.std(dicematrix[:, i]))

        stdmean = np.std(np.mean(dicematrix[:, 1:], axis=1))
        # for cbct
        # return hist, per_class_dice(hist), dice_groupby_individual, stdlist3, stdlist6, stdmean
        # for acdc
        return hist, per_class_dice(hist), dice_groupby_individual, stdlist, stdmean

    def inference(args, model, test_save_path=None, gtdir='', split="gantest"):
        # 测本人时，用'../data/syn40/SegmentationClass'
        # 测其他人时，用'../data/otherpeople/SegmentationClass'
        # 测gan翻译完的时，用'../data/otherpeople_aftergan_3040100/SegmentationClass'
        db_test = args.Dataset(base_dir=args.volume_path, split=split, list_dir=args.list_dir)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        model.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):

            if len(sampled_batch['image'].shape) < 4:
                sampled_batch['image'] = sampled_batch['image'].unsqueeze(0)
                sampled_batch['label'] = sampled_batch['label'].unsqueeze(0)
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            generate_save_predresult(image, label, model, classes=args.num_classes,
                                     patch_size=[args.img_size, args.img_size], test_save_path=test_save_path,
                                     case=case_name)

        hist_matrix,dice,hist_matrix_groupby_individual, stdlist, stdlistmean = cauculate_confusion_matrix_dice(args, test_save_path, gtdir, args.num_classes, model)
        for i in range(0, args.num_classes):
            print('Dice of %d class is %f' % (i, dice[i]))
        #print('std is:' + str(stdlist))
        performance = np.mean(dice[1:args.num_classes])
        return dice[1:args.num_classes], performance, stdlist, stdlistmean
        #return hist_matrix,dice,performance

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

    #找分割模型也要用的，所以用哪个集训练的就指定哪个
    dataset_config = {
        'syn40': {
            'Dataset': syn40_dataset,
            #'volume_path': '../data/syn40/',
            #'volume_path': '../data/otherpeople/',
            'volume_path': image_dir, #'../data/otherpeople_aftergan_3040100/',
            #'list_dir': './lists/lists_syn40/',
            'list_dir': './lists/lists_otherpeople/',
            #'list_dir': './lists/lists_otherpeople_aftergan/',
            'num_classes': 7,
            'z_spacing': 1,
        },
        'newsyn40': {
            'Dataset': newsyn40_dataset,
            'volume_path': image_dir,
            'list_dir': './lists/lists_otherpeople10_full/',
            'num_classes': 7,
            'z_spacing': 1,
        },
        'syn40_chenxi': {
            'Dataset': syn40_chenxi_dataset,
            'volume_path': image_dir,
            'list_dir': './lists/lists_otherpeople10_full_new/',
            'num_classes': 7,
            'z_spacing': 1,
        },
        'pddca_syn36': {
            'Dataset': pddca_syn36_dataset,
            'volume_path': image_dir, #'../../data/pddca_testset9'
            'list_dir': '../lists/lists_pddca_testset9',
            'num_classes': 7,
            'z_spacing': 1,
        },
        'ACDC70syn': {
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1
        },
        'ACDC70syn_equalized': {
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1
        },
        'ACDC70syn_nor_from067_90rot': {
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1
        },
        'ACDC70syn_nor_from067': {
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1
        },
        'ACDC70syn_nor_from067_doublecheck':{
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1
        },
        'ACDC70syn_099_90rot': {
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1,
        },
        'ACDC70syn_099_90rot_2':{
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1,
        },
        'ACDC70syn_10percent_ref': {
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1,
        },
        'ACDC70syn_10percent_ref_std20rot':{
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1,
        },
        'ACDC70syn_10percent_ref_only20rot': {
            'Dataset': ACDC70syn_dataset,
            'volume_path': image_dir,
            'list_dir': '../lists/lists_ACDC20test',
            'num_classes': 4,
            'z_spacing': 1,
        }
    }
    dataset_name = args.dataset
    args.is_savenii = True
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    #if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(69))
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_' + str(69))
    print(snapshot)
    #这个是训练的时候用了多卡，多卡加载的时候带一个module前缀(module.transformer)，但是初始化的不带module前缀直接是transformer
    #那两个讲这个的帖子，存在csdn 花式bug 收藏夹了
    if TRAIN_PARALLEL_FLAG:
        new_a = OrderedDict()
        a = torch.load(snapshot)
        for k,v in a.items():
            name = k[7:]
            new_a[name] = v
        net.load_state_dict(new_a)
    else:
        net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)

    #要存，看结果
    if args.is_savenii:
        args.test_save_dir = '../predictions_acdc_10percent_newganv4_ononlyrot' #'../predictions_acdc_nocrop_ganv3'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    dice,performance,stdlist,stdlistmean = inference(args, net, test_save_path, gtdir=gtdir)
    return dice,performance,stdlist,stdlistmean


import csv


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

import pickle as pkl

def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pkl.dump(data, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
        return data


if __name__ == '__main__':
    destdir = '../../ganresult/acdc10percentsyn_ofnewv4'
    epochlist = np.linspace(1,200,200,endpoint=True, dtype=int)
    # epochlist= [3]
    name = 'real2syn_acdc_nocropv4'
    name = 'real2syn_acdc_10percentsyn_v4'
    checkpdir = '../../../triple/checkpoints/'
    gtdir = '../../data/ACDC20test_nocrop/Seg'
    performance_dict = {}
    #epochlist=[54]
    for _,epoch in enumerate(epochlist):
        load_and_save_results(destdir=destdir, epoch=epoch, name=name,checkpdir=checkpdir)
        result_dir = os.path.join(destdir, name, 'test_{}'.format(epoch), 'images', 'fake_B')
        dice, performance, stdlist, stdlistmean = compute_dice_given_dir(result_dir, gtdir=gtdir)
        performance_dict['{}'.format(epoch)]=(dice, performance, stdlist, stdlistmean)

    dict2csv(performance_dict, 'performances_per_model_acdc10percent_newv4_onlyrot.csv')
    dump_pickle(performance_dict, './performances_per_model_acdc10percent_newv4_onlyrot.pkl')
    embed()
    pass
