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

import os, glob
import sys
from torch.utils.data import DataLoader
from mytools import datasets, trans, utils
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from network.model import Model
import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*indexing.*")

def main():
    channel_num = 8
    dataset = 'OASIS'
    # dataset = 'LPBA'
    # dataset = 'IXI'
    # dataset = 'Mindboggle'
    # dataset = 'Abdomen'
    experiment_time = '2025'

    if dataset == 'OASIS':
        test_dir = '/home/user/hanfeiyang/dataset/OASIS/OASIS_L2R_2021_task03/Test/'
        test_dir = glob.glob(test_dir + '*.pkl')
        atlas_dir = ''
        img_size = (160, 192, 224)
        testDataset = datasets.OASISTestDataset
        VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 31, 32, 33, 34, 35]  # 35
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
        test_dir = '/home/user/hanfeiyang/dataset/LPBA/LPBA40_delineation/delineation_l_norm/test/'
        test_dir = glob.glob(test_dir + '*.gz')
        atlas_dir = {'img':'/home/user/hanfeiyang/dataset/LPBA/LPBA40_delineation/delineation_l_norm/fixed.nii.gz',
                     'seg':'/home/user/hanfeiyang/dataset/LPBA/LPBA40_delineation/label/S01.delineation.structure.label.nii.gz'}
        img_size = (160, 192, 160)
        testDataset = datasets.LPBATestDataset
        VOI_lbls = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                    61, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122,
                    161, 162, 163, 164, 165, 166, 181, 182]  # 56
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
        test_dir = '/home/user/hanfeiyang/dataset/IXI/IXI_data/Test/'
        test_dir = glob.glob(test_dir + '*.pkl')
        atlas_dir = '/home/user/hanfeiyang/dataset/IXI/IXI_data/atlas.pkl'
        img_size = (160, 192, 224)
        testDataset = datasets.IXITestDataset
        VOI_lbls = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 31, 41, 42, 43, 46, 47, 49, 50, 51, 52,
                    53, 54, 60, 63]  # 30 follow transmorph
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
        test_basedir = '/home/user/hanfeiyang/dataset/Mindboggle/Mindboggle101_volumes/OASIS-TRT-20_volumes/*'
        test_basedir = glob.glob(test_basedir)
        test_dir = []
        for dir in test_basedir:
            dir = dir + '/'
            test_dir.append(dir)
        test_dir = non_self_permutations(test_dir)

        atlas_dir = ''
        img_size = (160, 192, 160)
        testDataset = datasets.MindboggleTestDataset
        VOI_lbls = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018,
                    1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035, 2002,
                    2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                    2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035]  # 62
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
        test_dir = '/home/user/hanfeiyang/dataset/Abdomen/after-processing-abdomen_MR-CT/test/CT/data/'
        test_dir = glob.glob(test_dir + '*.npy')
        atlas_dir = ''
        img_size = (192, 160, 192)
        testDataset = datasets.AbdomenTestDataset
        VOI_lbls = [1, 2, 3, 4]  # 4
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

    model_folder = '{}_{}/'.format(dataset, experiment_time)
    model_dir = 'experiments/' + model_folder
    model_idx = -1
    sys.stdout = utils.Logger('logs/' + model_folder, file_name=natsorted(os.listdir(model_dir))[model_idx].split('.pth.tar')[0])

    '''
    Initialize model
    '''
    model = Model(channel_num=channel_num, adaptive=adaptive, iter=iter)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    Initialize testing
    '''
    img_type = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    seg_type = transforms.Compose([trans.NumpyType((np.int16, np.int16))])
    test_set = testDataset(data_path=test_dir, atlas_dir=atlas_dir, img_type=img_type, seg_type=seg_type, img_size=img_size)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    times = []
    eval_dsc = utils.AverageMeter()
    print('====================', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '====================')
    print('Testing Starts')
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            start = time.time()
            output, flow = model(x, y)
            times.append(time.time()-start)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            dsc = utils.dice_val_VOI(def_out.long(), y_seg.long(), VOI_lbls=VOI_lbls)
            eval_dsc.update(dsc.item(), x.size(0))
            print('dsc', dsc.item())

    print(times)
    print('dsc_avg {:.4f}'.format(eval_dsc.avg))
    print('====================', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '====================')


def non_self_permutations(dir_list):
    result = []
    for i in dir_list:
        for j in dir_list:
            if i != j:
                result.append({'x': i, 'y': j})
    return result


if __name__ == '__main__':
    GPU_iden = 0
    torch.cuda.set_device(GPU_iden)
    main()
