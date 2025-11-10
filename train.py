import torch
import random
import numpy as np
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
num_workers = 0

from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys
from torch.utils.data import DataLoader
from mytools import datasets, trans, utils, losses
from torchvision import transforms
from torch import optim
from natsort import natsorted
from network.model import Model
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*indexing.*")

def main():
    dataset = 'OASIS'
    # dataset = 'LPBA'
    # dataset = 'IXI'
    # dataset = 'Mindboggle'
    # dataset = 'Abdomen'
    channel_num = 8

    batch_size = 1
    max_epoch = 500
    cont_training = False

    if dataset=='OASIS':
        train_dir = '/home/user/hanfeiyang/dataset/OASIS/OASIS_L2R_2021_task03/All/'
        train_dir = glob.glob(train_dir + '*.pkl')
        val_dir = '/home/user/hanfeiyang/dataset/OASIS/OASIS_L2R_2021_task03/Test/'
        val_dir = glob.glob(val_dir + '*.pkl')
        atlas_dir = ''
        img_size = (160, 192, 224)
        trainDataset = datasets.OASISTrainDataset
        valDataset = datasets.OASISTestDataset
        lr = 0.0001
        weights = [1, 1]  # loss weights
        VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 31, 32, 33, 34, 35]  # 35
        criterion_sim = losses.NCC_vxm()  # NCC
        criterion_reg = losses.Grad3d(penalty='l2')  # smooth
        adaptive = {
            'num_k': 7,
            'tol': 1e-9,
            'm_iter': 1000,
        }
        iter = {
            '4': 2,
            '3': 2,
            '2': 2,
            '1': 2,
        }
    elif dataset=='LPBA':
        train_dir = '/home/user/hanfeiyang/dataset/LPBA/LPBA40_delineation/delineation_l_norm/train/'
        train_dir = glob.glob(train_dir + '*.gz')
        val_dir = '/home/user/hanfeiyang/dataset/LPBA/LPBA40_delineation/delineation_l_norm/test/'
        val_dir = glob.glob(val_dir + '*.gz')
        atlas_dir = {'img':'/home/user/hanfeiyang/dataset/LPBA/LPBA40_delineation/delineation_l_norm/fixed.nii.gz',
                     'seg':'/home/user/hanfeiyang/dataset/LPBA/LPBA40_delineation/label/S01.delineation.structure.label.nii.gz'}
        img_size = (160, 192, 160)
        trainDataset = datasets.LPBATrainDataset
        valDataset = datasets.LPBATestDataset
        lr = 0.0004
        weights = [1, 1]  # loss weights
        VOI_lbls = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                    61, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122,
                    161, 162, 163, 164, 165, 166, 181, 182]  # 56
        criterion_sim = losses.NCC_vxm()  # NCC
        criterion_reg = losses.Grad3d(penalty='l2')  # smooth
        adaptive = {
            'num_k': 7,
            'tol': 1e-9,
            'm_iter': 1000,
        }
        iter = {
            '4': 2,
            '3': 2,
            '2': 2,
            '1': 2,
        }
    elif dataset=='IXI':
        train_dir = '/home/user/hanfeiyang/dataset/IXI/IXI_data/Train/'
        train_dir = glob.glob(train_dir + '*.pkl')
        val_dir = '/home/user/hanfeiyang/dataset/IXI/IXI_data/Val/'
        val_dir = glob.glob(val_dir + '*.pkl')
        atlas_dir = '/home/user/hanfeiyang/dataset/IXI/IXI_data/atlas.pkl'
        img_size = (160, 192, 224)
        trainDataset = datasets.IXITrainDataset
        valDataset = datasets.IXITestDataset
        lr = 0.0001
        weights = [1, 1]  # loss weights
        VOI_lbls = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 31, 41, 42, 43, 46, 47, 49, 50, 51, 52,
                    53, 54, 60, 63]  # 30 follow transmorph
        criterion_sim = losses.NCC_vxm()  # NCC
        criterion_reg = losses.Grad3d(penalty='l2')  # smooth
        adaptive = {
            'num_k': 7,
            'tol': 1e-9,
            'm_iter': 1000,
        }
        iter = {
            '4': 2,
            '3': 2,
            '2': 2,
            '1': 2,
        }
    elif dataset == 'Mindboggle':
        train_basedir = '/home/user/hanfeiyang/dataset/Mindboggle/Mindboggle101_volumes/'
        train_basedir = glob.glob(train_basedir + 'NKI-*_volumes')
        train_dir = []
        for dir in train_basedir:
            dir = dir + '/*'
            dir = glob.glob(dir)
            for path in dir:
                path = path + '/'
                train_dir.append(path)
        train_dir = non_self_permutations(train_dir)

        val_basedir = '/home/user/hanfeiyang/dataset/Mindboggle/Mindboggle101_volumes/OASIS-TRT-20_volumes/*'
        val_basedir = glob.glob(val_basedir)
        val_dir = []
        for dir in val_basedir:
            dir = dir + '/'
            val_dir.append(dir)
        val_dir = non_self_permutations(val_dir)

        atlas_dir = ''
        img_size = (160, 192, 160)
        trainDataset = datasets.MindboggleTrainDataset
        valDataset = datasets.MindboggleTestDataset
        lr = 0.0001
        weights = [1, 1]  # loss weights
        VOI_lbls = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018,
                    1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002,
                    2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                    2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035]  # 62
        criterion_sim = losses.NCC_vxm()  # NCC
        criterion_reg = losses.Grad3d(penalty='l2')  # smooth
        adaptive = {
            'num_k': 7,
            'tol': 1e-9,
            'm_iter': 1000,
        }
        iter = {
            '4': 2,
            '3': 2,
            '2': 2,
            '1': 2,
        }
    elif dataset == 'Abdomen':
        train_dir = '/home/user/hanfeiyang/dataset/Abdomen/after-processing-abdomen_MR-CT/train/CT/data/'
        train_dir = glob.glob(train_dir + '*.npy')
        val_dir = '/home/user/hanfeiyang/dataset/Abdomen/after-processing-abdomen_MR-CT/test/CT/data/'
        val_dir = glob.glob(val_dir + '*.npy')
        atlas_dir = ''
        img_size = (192, 160, 192)
        trainDataset = datasets.AbdomenTrainDataset
        valDataset = datasets.AbdomenTestDataset
        lr = 0.0002
        weights = [1, 1]  # loss weights
        VOI_lbls = [1, 2, 3, 4]  # 4
        criterion_sim = losses.MutualInformation()  # MI
        criterion_reg = losses.Grad3d(penalty='l2')  # smooth
        adaptive = {
            'num_k': 7,
            'tol': 1e-9,
            'm_iter': 1000,
        }
        iter = {
            '4': 2,
            '3': 2,
            '2': 2,
            '1': 2,
        }
    '''
    Initialize model
    '''
    model = Model(channel_num=channel_num, adaptive=adaptive, iter=iter)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        save_dir = '{}_'.format(dataset) + 'experiment_time/'

        epoch_start = 353
        model_dir = 'experiments/'+save_dir
        updated_lr = lr
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        save_dir = '{}_{}/'.format(dataset, datetime.now().strftime("%Y%m%d%H%M"))
        if not os.path.exists('experiments/' + save_dir):
            os.makedirs('experiments/' + save_dir)
        if not os.path.exists('logs/' + save_dir):
            os.makedirs('logs/' + save_dir)
        sys.stdout = utils.Logger('logs/' + save_dir)

        epoch_start = 0
        updated_lr = lr

    '''
    Initialize training
    '''
    img_type = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    seg_type = transforms.Compose([trans.NumpyType((np.int16, np.int16))])
    train_set = trainDataset(data_path=train_dir, atlas_dir=atlas_dir, img_type=img_type, seg_type=seg_type, img_size=img_size)
    val_set = valDataset(data_path=val_dir, atlas_dir=atlas_dir, img_type=img_type, seg_type=seg_type, img_size=img_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    print("weights: [0]_{} [1]_{}".format(weights[0], weights[1]))
    print("lr: {}".format(lr))
    best_dsc = float('-inf')
    writer = SummaryWriter(log_dir='logs/'+save_dir)

    for epoch in range(epoch_start, max_epoch):
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        print('====================', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '====================')
        print('Training Starts')
        for data in tqdm(train_loader, desc="Epoch {:03d}".format(epoch), ncols=100):
            idx += 1
            model.train()

            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            output, flow = model(x, y)

            loss_sim = criterion_sim(output, y) * weights[0]
            loss_reg = criterion_reg(flow) * weights[1]
            loss = loss_sim + loss_reg

            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('Epoch {} loss {:.4f} lr {:.6f}'.format(epoch, loss_all.avg, current_lr))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                output, flow = model(x, y)
                def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long(), VOI_lbls=VOI_lbls)
                eval_dsc.update(dsc.item(), x.size(0))
                print('dsc', dsc.item())
        print('dsc_avg {:.4f}  dsc_std {:.4f}'.format(eval_dsc.avg, eval_dsc.std))
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            },
            save_dir='experiments/'+save_dir,
            filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg)
        )

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        loss_all.reset()
        del def_out, output, flow
        torch.cuda.empty_cache()
    print('====================', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '====================')
    writer.close()

def non_self_permutations(dir_list):
    result = []
    for i in dir_list:
        for j in dir_list:
            if i != j:
                result.append({'x': i, 'y': j})
    return result

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
