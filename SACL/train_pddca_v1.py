import torchvision.transforms as transforms
import os
import random
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import argparse
from models.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from collections import OrderedDict
import SimpleITK as sitk
from scipy.ndimage import zoom
from PIL import Image
import torch.backends.cudnn as cudnn
import functools
from torch.nn import init
from packaging import version
from copy import deepcopy


from models import networks
from utils import create_and_load_TU,freeze,map2sample,create_TU
from options.train_options import TrainOptions
from torch.nn.modules.loss import CrossEntropyLoss
from data.dataset_ABmixed_offlinepl import RandomGenerator,dataset_ABmixed
import logging
import math
from torch.nn import functional as F
from models.SRC import SRC_Loss
from models.hDCE import PatchHDCELoss

def load_and_freeze_TU(freeze_flag=False,snapshot_path='./pddca_netTU_from0727.pth'):
    TU = create_and_load_TU(snapshot_path=snapshot_path)
    if freeze_flag:
        freeze(TU)
    return TU

def load_and_freeze_G(pth_path='./pddcav1_netG_ep336.pth',device=torch.device('cpu')):
    norm_layer = networks.get_norm_layer(norm_type='instance')
    net = networks.ResnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False, no_antialias=False, no_antialias_up=False,
                          n_blocks=9)
    netG = networks.init_net(net, 'xavier', 0.02, [0], initialize_weights=('stylegan2' not in 'resnet_9blocks'))
    state_dict = torch.load(pth_path,
                            map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    netG.load_state_dict(state_dict)
    freeze(netG)
    return netG #.to('cpu')

# command: python ./train_pddca_v1.py --list_dir ./lists/lists_pddca --root_path ./datasets/ABmixed_pddca

if __name__ == '__main__':
    opt = TrainOptions().parse()
    base_lr = opt.base_lr
    gpu_ids = opt.gpu_ids
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    snapshot_path1 = './checkpoints/TU_pddcav1/TU1'
    snapshot_path2 = './checkpoints/TU_pddcav1/TU2'

    if opt.continue_train == False:
        TU1 = create_TU()
        TU2 = create_TU()
        TU = load_and_freeze_TU(freeze_flag=True)
    else:
        TU1 = load_and_freeze_TU(snapshot_path=os.path.join(snapshot_path1, 'epoch_' + str(opt.epoch_count) + '.pth'))
        TU2 = load_and_freeze_TU(snapshot_path=os.path.join(snapshot_path2, 'epoch_' + str(opt.epoch_count) + '.pth'))
        TU = load_and_freeze_TU(freeze_flag=True)

    netG_A2B = load_and_freeze_G(pth_path='./pddcav1_netG_ep336.pth',device=device)

    logging.basicConfig(filename=snapshot_path2 + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(opt))
    #base_lr = opt.base_lr
    num_classes = opt.num_classes
    batch_size = opt.batch_size * opt.n_gpu
    db_train = dataset_ABmixed(opt, transform=transforms.Compose([RandomGenerator(output_size=[opt.img_size, opt.img_size])]),
                               G=netG_A2B, TU1=TU, device=device)
    print("The length of train set is: {}".format(len(db_train)))

    if opt.continue_train == True:
        logging.info('successfully load from' + os.path.join(snapshot_path2, 'epoch_' + str(opt.epoch_count) + '.pth'))
        logging.info('and' + os.path.join(snapshot_path1, 'epoch_' + str(opt.epoch_count) + '.pth'))

    def worker_init_fn(worker_id):
        random.seed(opt.seed + worker_id)


    #trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if opt.n_gpu > 1:
        TU1 = nn.DataParallel(TU1)
        TU2 = nn.DataParallel(TU2)
    TU1.train()
    TU2.train()

    ce_loss = CrossEntropyLoss()
    ce_loss_sup1 = CrossEntropyLoss()
    ce_loss_sup2 = CrossEntropyLoss()
    criterionR = SRC_Loss(opt).to(device)
    criterionHDCE = PatchHDCELoss(opt).to(device)

    optimizerT1 = optim.SGD(TU1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizerT2 = optim.SGD(TU2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path2 + '/log')
    iter_num = 0
    max_epoch = opt.max_epochs
    max_iterations = opt.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    if opt.continue_train==False:
        iterator = tqdm(range(max_epoch), ncols=70)
    else:
        iterator = tqdm(range(opt.epoch_count + 1, max_epoch), ncols=70)
        iter_num = len(trainloader) * (opt.epoch_count + 1)
        lr_ = base_lr * (1.0 - (iter_num - 1) / max_iterations) ** 0.9
        for param_group in optimizerT1.param_groups:
            param_group['lr'] = lr_
        for param_group in optimizerT2.param_groups:
            param_group['lr'] = lr_


    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            im_aW_batch, im_aS_batch, label_batch = sampled_batch['im_aW'].cuda(), sampled_batch['im_aS'].cuda(), sampled_batch['label'].cuda()
            angle_batch, field_batch = sampled_batch['angle'], sampled_batch['d_field']
            outputs_TU1 = TU1(im_aS_batch)
            outputs_TU2 = TU2(im_aW_batch)
            re_transform_matrix = torch.zeros((outputs_TU2.shape[0], 2, 3), device=device, requires_grad=False)

            # 参考https://blog.51cto.com/u_15906550/5921646
            # 角度旋转逆变换，从角度算出torch需要的逆变换旋转格点
            for j in range(re_transform_matrix.shape[0]):
                re_transform_matrix[j] = torch.tensor([
                    [math.cos(-angle_batch[j]*math.pi/180), math.sin(angle_batch[j]*math.pi/180), 0],
                    [math.sin(-angle_batch[j]*math.pi/180), math.cos(-angle_batch[j]*math.pi/180), 0]])

            affine_grid = F.affine_grid(re_transform_matrix, outputs_TU2.shape, align_corners=True)
            # padding_mode取默认zeros会有问题
            outputs_TU2_reverse = F.grid_sample(outputs_TU2, affine_grid, align_corners=True, mode='nearest', padding_mode='border')

            outputs_TU2_deform = F.grid_sample(outputs_TU2_reverse, field_batch.cuda().float(), mode='nearest',align_corners=True)

            loss_ce = ce_loss(outputs_TU1, torch.argmax(torch.softmax(outputs_TU2_deform, dim=1), dim=1))

            label_batch_deform = F.grid_sample(label_batch.unsqueeze(1).float(), field_batch.cuda().float(), align_corners=True, mode='nearest', padding_mode='border')
            loss_ce_sup1 = ce_loss_sup1(outputs_TU1, label_batch_deform.squeeze().long())
            loss_ce_sup2 = ce_loss_sup2(outputs_TU2_reverse, label_batch)
            #loss_ce = ce_loss(outputs_TU2_deform, torch.argmax(torch.softmax(outputs_TU1, dim=1), dim=1))

            num_patches = 1024#2048
            #test2 = torch.softmax(test1.type(torch.int64), dim=3)

            softonehot_label = torch.softmax(F.one_hot(label_batch,num_classes=7).float(),dim=3) #.permute(0,3,1,2)
            #softonehot_label = torch.softmax(F.one_hot(label_batch, num_classes=7).type(torch.float32), dim=3)
            softonehot_label_flatten = softonehot_label.flatten(1,2)
            outputs_TU2_reverse_flatten = torch.softmax(outputs_TU2_reverse.permute(0,2,3,1),dim=3).flatten(1,2) #.permute(0,3,1,2).flatten(2,3)

            #类别采样，同一batch每个数据类别采样都不一样，要循环
            map_id = torch.zeros(label_batch.size()[0], num_patches).to(device).long()
            for i in range(label_batch.size()[0]):
                map_id_i = torch.nonzero(label_batch[i].flatten(0,1))
                if(len(map_id_i)>=num_patches):
                    #map_id_i = random.shuffle(map_id_i)
                    idid = torch.randperm(len(map_id_i), device=device)
                    idid = idid[0:num_patches]
                    #map_id_i = map_id_i[0:num_patches]
                    map_id_i = map_id_i[idid,:]
                elif(len(map_id_i) == 0):
                    # 以防还有伪标签全是背景的
                    map_id_i = torch.randperm(224*224, device=device)
                    map_id_i = map_id_i[:num_patches]
                else:
                    setdiff = set(np.arange(0, 224*224)) - set(torch.squeeze(map_id_i, 1).tolist())
                    setresult = random.sample(setdiff, num_patches - len(map_id_i))
                    map_id_i = torch.cat([torch.squeeze(map_id_i, 1), torch.tensor(list(setresult)).to(device)], 0)
                #map_id.append(torch.squeeze(map_id_i))
                map_id[i] = torch.squeeze(map_id_i)

            rowindex = torch.arange(start=0, end=softonehot_label_flatten.size()[0], step=1).unsqueeze(1).long()
            softonehot_label_flatten_s = softonehot_label_flatten[rowindex,map_id,:]
            outputs_TU2_reverse_flatten_s = outputs_TU2_reverse_flatten[rowindex,map_id,:]

            loss_SRC, weight_for_DCE = criterionR(outputs_TU2_reverse_flatten_s, softonehot_label_flatten_s, epoch=epoch_num)
            loss_HDCE = criterionHDCE(outputs_TU2_reverse_flatten_s, softonehot_label_flatten_s, weight_for_DCE).mean()


            total_L = 0.2*loss_ce + loss_ce_sup1 + loss_ce_sup2 + loss_HDCE*opt.lambda_HDCE + loss_SRC*opt.lambda_SRC

            optimizerT1.zero_grad()
            optimizerT2.zero_grad()
            total_L.backward()
            optimizerT1.step()
            optimizerT2.step()

            # TU1是student，TU2是被ema优化的teacher
            for param_stud, param_teach in zip(TU1.parameters(), TU2.parameters()):
                param_teach.data.copy_(0.001 * param_stud + 0.999 * param_teach)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizerT1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizerT2.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', total_L, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_ce_sup1: %f , loss_ce_sup2: %f, loss_R: %f, loss_dce: %f' \
                         % (iter_num, total_L.item(), loss_ce.item(), loss_ce_sup1.item(), loss_ce_sup2.item(),
                            loss_SRC.item(), loss_HDCE.item()))

            if iter_num % 20 == 0:

                # fordbg
                # print(iter_num,epoch_num)

                image_W = im_aW_batch[1, 0:1, :, :]
                image_S = im_aS_batch[1, 0:1, :, :]
                image_W = (image_W - image_W.min()) / (image_W.max() - image_W.min())
                image_S = (image_S - image_S.min()) / (image_S.max() - image_S.min())
                # 要查文件夹对应关系
                writer.add_image('train/Image_W', image_W, iter_num)
                writer.add_image('train/Image_S', image_S, iter_num)
                outputs = torch.argmax(torch.softmax(outputs_TU2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 保存TU1,TU2
        save_interval = 1  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        # if epoch_num in pool:
        if (epoch_num<=9) | ((epoch_num + 1) % save_interval == 0):
            save_mode_path = os.path.join(snapshot_path1, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(TU1.state_dict(), save_mode_path)
            save_mode_path = os.path.join(snapshot_path2, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(TU2.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path1, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(TU1.state_dict(), save_mode_path)
            save_mode_path = os.path.join(snapshot_path2, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(TU2.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()

    print("Training Finished!")