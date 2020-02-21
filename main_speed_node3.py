# from opts import parse_opts
# from mean import get_mean, get_std
from spatial_transforms import (Compose, ToTensor, Normalize, MultiScaleCornerCrop, RandomHorizontalFlip)
# from target_transforms import ClassLabel, VideoID

import cv2
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import torch.utils.data as data
from glob import glob
import pandas as pd
import numpy as np

import csv
import os
import random
import sys
from torch import optim
from torch.optim import lr_scheduler
import time
import torchvision
from PIL import Image
from torch.utils.data import Subset, ConcatDataset

import matplotlib.pyplot as plt

#####
print(os.getcwd())
import sys
from os import path

sys.path.append(path.join("../../comma2k19/notebooks"))
sys.path.append(path.join("../../comma2k19/notebooks/openpilot"))
from openpilot.tools.lib.framereader import FrameReader

####

# from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='/root/data/ActivityNet',
        type=str,
        help='Root directory path of data')

    parser.add_argument(
        '--video_path',
        # default="../../klumblr_bbox_new_data/*/*/*/video.hevc",
        # default='../../data_all/video/choi/20190404/AlwaysMovie/alwa_20190404_110322_F.MP4',
        default='../../../klumblr_bbox_new_data/*/*/*/video.hevc',
        #         default='../../../dataset/comma2k19/Chunk_1/b0c9d2329ad1606b_2018-07-27--06-03-57/*/video.hevc',
        # default='../../../dataset/comma2k19/Chunk_1/*/*/video.hevc',
        # default='../../../dataset/comma2k19/*/*/*/video.hevc',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_path',
        # default="../../klumblr_bbox_new_data/*/*/*/global_pose/frame_velocities",
        default="../../../klumblr_bbox_new_data/*/*/*/global_pose/frame_velocities",
        #         default='../../../dataset/comma2k19/Chunk_1/b0c9d2329ad1606b_2018-07-27--06-03-57/*/global_pose/frame_velocities',
        # default='../../../dataset/comma2k19/Chunk_1/*/*/global_pose/frame_velocities',
        # default='../../../dataset/comma2k19/*/*/*/global_pose/frame_velocities',
        type=str,
        help='Annotation file path')

    parser.add_argument(
        '--result_path',
        default='results2/results12',
        #         default='results2/results5/demo',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='Comma2k19',
        type=str,
        help='Used dataset Comma2k19')
    parser.add_argument(
        '--n_classes',
        default=16,
        type=int,
        help=
        'Number of classes (16 frame speeds)'
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=400,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales',
        default=0,
        type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.0,
        type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='corner',
        type=str,
        help=
        'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    )
    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')

    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument(
        '--mean_dataset',
        default='activitynet',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')

    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--std_norm',
        action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--batch_size', default=20, type=int, help='Batch Size')

    parser.add_argument(
        '--n_epochs',
        default=20,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=3,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--resume_path',
        # default='',
        default='./results2/results10/save_11_1137.pth',
        #         default='./results2/results0/save_0_1137.pth',
        #         default='./results2/results6/save_6_974.pth', # batch_size : 20
        #         default='./results2/results5/save_5_974.pth', # batch_size : 20
        #         default='./results2/results4/save_4_853.pth', # batch_size : 20
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--scale_in_test',
        default=1.0,
        type=float,
        help='Spatial scale in test')
    parser.add_argument(
        '--crop_position_in_test',
        default='c',
        type=str,
        help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=0,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--norm_value',
        default=255.0,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=16):
        print("ResNet configuration: ")
        print("block: ", block)
        print("layers: ", layers)
        print("sample_size: ", sample_size)
        print("sample_duration: ", sample_duration)
        print("shortcut_type: ", shortcut_type)
        print("num_classes: ", num_classes)
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)

        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)

        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)

        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)

        last_duration = int(math.ceil(sample_duration / 16))

        last_size = int(math.ceil(sample_size / 32))

        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class resnet():

    def resnet18(**kwargs):
        """Constructs a ResNet-18 model.
        """

        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        return model


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def generate_model(opt):
    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        # from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

        if not opt.no_cuda:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)

            if opt.pretrain_path:
                print('loading pretrained model {}'.format(opt.pretrain_path))
                pretrain = torch.load(opt.pretrain_path)
                assert opt.arch == pretrain['arch']

                model.load_state_dict(pretrain['state_dict'])

                if opt.model == 'densenet':
                    model.module.classifier = nn.Linear(
                        model.module.classifier.in_features, opt.n_finetune_classes)
                    model.module.classifier = model.module.classifier.cuda()
                else:
                    model.module.fc = nn.Linear(model.module.fc.in_features,
                                                opt.n_finetune_classes)
                    model.module.fc = model.module.fc.cuda()

                parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
                return model, parameters
        #         print("summary")
        #         summary(model, (3, 16, 112, 112))
        return model, model.parameters()


def normalize(image):
    return image - [104.00699, 116.66877, 122.67892]


def crop_image(image):
    h, w, _ = image.shape

    h_bottom = h - int(h * 0.10)
    h_top = int(h * 0.30)

    w_crop_size = int(w * 0.15)

    return image[h_top:h_bottom, w_crop_size:(w - w_crop_size)]


class Comma2k19_nanopen(data.Dataset):
    def __init__(self,
                 video_path_all,
                 annotation_path_all,

                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16):

        ####
        label_list_2k19 = glob(annotation_path_all)
        video_list_2k19 = glob(video_path_all)

        self.vframes = []
        self.subvideos_per_video = []
        self.timesteps = sample_duration
        # self.data_path = {"videos": video_list_2k19,
        #                   "speed": label_list_2k19}
        sum_f = 0
        sum_sq_f = 0
        temp_label_list_2k19 = []
        temp_video_list_2k19 = []

        for i in range(len(label_list_2k19)):
            # if i<0:continue
            # if i >0:break
            label_path = label_list_2k19[i]
            video_path = video_list_2k19[i]
            label_temp = self.read_ylabel2k19(label_path)
            sum_f += np.sum(label_temp)
            sum_sq_f += np.sum(label_temp * label_temp)

            self.vframes.append(len(label_temp))
            self.subvideos_per_video.append(len(label_temp) // self.timesteps)
            temp_video_list_2k19.append(video_path)
            temp_label_list_2k19.append(label_path)

        label_list_2k19 = temp_label_list_2k19
        video_list_2k19 = temp_video_list_2k19

        self.data_path = {"videos": video_list_2k19,
                          "speed": label_list_2k19}
        # mean, var, std of speed targets
        self.mean_speed = (sum_f * 1.0 / np.sum(self.vframes)).astype(np.float32)
        self.var_speed = (sum_sq_f * 1.0 / np.sum(self.vframes) - self.mean_speed * self.mean_speed).astype(np.float32)
        self.std_speed = np.sqrt(self.var_speed)

        print("mean: ", self.mean_speed)
        print("var: ", self.var_speed)
        print("std: ", self.std_speed)

        # mean, var, std =  https://github.com/cardwing/Codes-for-Steering-Control/blob/master/steering-control/3d_resnet_lstm.py
        self.real_subvideos_per_epoch = int(np.sum(self.subvideos_per_video))  # number of clips

        ####
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        print("--init--")
        print("video path:", video_path_all)
        print("annotation path:", annotation_path_all)
        print("sample_duration:", self.timesteps)
        print("number of videos: ", len(video_list_2k19))
        print("real_subvideos_per_epoch:", self.real_subvideos_per_epoch)
        print("--continue--")
        self.video_index = 0
        self.frame_idx = 0

    def read_ylabel2k19(self, y_label_path):
        train_y = np.linalg.norm(np.load(y_label_path), axis=1)

        return train_y

    def __getitem__(self, sub_video_index):
        # get a random clip by sub_video_index (random number based on data len)
        ###
        index_count = 0
        frame_idx = 0
        video_index = 0
        # get clip location which are video index and frame index w.r.t its video

        for i in range(len(self.subvideos_per_video)):
            temp = index_count + self.subvideos_per_video[i]
            if sub_video_index < temp:
                video_index = i
                frame_idx = (sub_video_index - index_count) * self.timesteps

                break
            else:
                index_count = temp
        self.video_index = video_index
        self.frame_idx = frame_idx

        # get video path from video index
        # print()
        # print("sub_video_index:", sub_video_index)
        # print("video_inex: ", video_index)
        # print("frame index:", frame_idx)
        # print()
        video_path = self.data_path['videos'][video_index]
        label_path = self.data_path['speed'][video_index]

        # get annotation
        labels = (self.read_ylabel2k19(
            label_path) - self.mean_speed) * 1.0 / self.std_speed  # speed of each frame ~ 1200
        # get video and clip (also crop and resize clip)
        cap = cv2.VideoCapture(video_path)
        # cap.set(1, frame_idx)

        for i in range(frame_idx - 1):
            ret, frame = cap.read()

        cur_video_seq = []
        cur_label_seq = []
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()

        for seq_idx in range(self.timesteps):
            ret, frame = cap.read()
            if ret:
                # Handling Image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).convert('RGB')
                label = labels[frame_idx + seq_idx]
                if self.spatial_transform is not None:
                    frame = self.spatial_transform(frame)
                    # img = frame
                    # plt.imshow(img.transpose(0,2).transpose(0,1))
                    # plt.title("test data")
                    # plt.show()
                else:
                    frame = torchvision.transforms.ToTensor()(frame)  # (w,h,d)->(d,w,h)

                cur_video_seq.append(frame)
                cur_label_seq.append(label)

            else:
                return None

        clip = cur_video_seq
        target = torch.from_numpy(np.array(cur_label_seq, dtype=np.float32))
        # print("target",target)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  ###
        # # print("clip shape: ",np.shape(clip))
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return clip, target, self.video_index, self.frame_idx, video_path, label_path, self.mean_speed, self.std_speed

    def __len__(self):
        return self.real_subvideos_per_epoch


class Comma2k19(data.Dataset):
    def __init__(self,
                 video_path_all,
                 annotation_path_all,

                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16):

        ####
        label_list_2k19 = glob(annotation_path_all)
        video_list_2k19 = glob(video_path_all)

        self.vframes = []
        self.subvideos_per_video = []
        self.timesteps = sample_duration
        # self.data_path = {"videos": video_list_2k19,
        #                   "speed": label_list_2k19}
        sum_f = 0
        sum_sq_f = 0
        temp_label_list_2k19 = []
        temp_video_list_2k19 = []

        for i in range(len(label_list_2k19)):
            # if i<0:continue
            # if i >0:break
            label_path = label_list_2k19[i]
            video_path = video_list_2k19[i]
            label_temp = self.read_ylabel2k19(label_path)
            sum_f += np.sum(label_temp)
            sum_sq_f += np.sum(label_temp * label_temp)

            self.vframes.append(len(label_temp))
            self.subvideos_per_video.append(len(label_temp) // self.timesteps)
            temp_video_list_2k19.append(video_path)
            temp_label_list_2k19.append(label_path)

        label_list_2k19 = temp_label_list_2k19
        video_list_2k19 = temp_video_list_2k19

        self.data_path = {"videos": video_list_2k19,
                          "speed": label_list_2k19}
        # mean, var, std of speed targets
        self.mean_speed = (sum_f * 1.0 / np.sum(self.vframes)).astype(np.float32)
        self.var_speed = (sum_sq_f * 1.0 / np.sum(self.vframes) - self.mean_speed * self.mean_speed).astype(np.float32)
        self.std_speed = np.sqrt(self.var_speed)

        print("mean: ", self.mean_speed)
        print("var: ", self.var_speed)
        print("std: ", self.std_speed)

        # mean, var, std =  https://github.com/cardwing/Codes-for-Steering-Control/blob/master/steering-control/3d_resnet_lstm.py
        self.real_subvideos_per_epoch = int(np.sum(self.subvideos_per_video))  # number of clips

        ####
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        print("--init--")
        print("video path:", video_path_all)
        print("annotation path:", annotation_path_all)
        print("sample_duration:", self.timesteps)
        print("number of videos: ", len(video_list_2k19))
        print("real_subvideos_per_epoch:", self.real_subvideos_per_epoch)
        print("--continue--")
        self.video_index = 0
        self.frame_idx = 0

    def read_ylabel2k19(self, y_label_path):
        train_y = np.linalg.norm(np.load(y_label_path), axis=1)

        return train_y

    def __getitem__(self, sub_video_index):

        # get a random clip by sub_video_index (random number based on data len)
        ###
        index_count = 0
        frame_idx = 0
        video_index = 0
        # get clip location which are video index and frame index w.r.t its video

        for i in range(len(self.subvideos_per_video)):
            temp = index_count + self.subvideos_per_video[i]
            if sub_video_index < temp:
                video_index = i
                frame_idx = (sub_video_index - index_count) * self.timesteps

                break
            else:
                index_count = temp
        self.video_index = video_index
        self.frame_idx = frame_idx

        # get video path from video index
        # print()
        # print("sub_video_index:", sub_video_index)
        # print("video_inex: ", video_index)
        # print("frame index:", frame_idx)
        # print()
        video_path = self.data_path['videos'][video_index]
        label_path = self.data_path['speed'][video_index]

        # get annotation
        labels = (self.read_ylabel2k19(
            label_path) - self.mean_speed) * 1.0 / self.std_speed  # speed of each frame ~ 1200
        # get video and clip (also crop and resize clip)
        '''
        cap = cv2.VideoCapture(video_path)
        # cap.set(1, frame_idx)

        for i in range(frame_idx - 1):
            ret, frame = cap.read()
            
            
        '''
        fr = FrameReader(video_path)

        cur_video_seq = []
        cur_label_seq = []
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()

        for seq_idx in range(self.timesteps):
            #             ret, frame = cap.read()
            frame_index = frame_idx + seq_idx
            frame = fr.get(frame_index, pix_fmt='rgb24')[0]

            if True:
                # Handling Image
                #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).convert('RGB')
                label = labels[frame_idx + seq_idx]
                if self.spatial_transform is not None:
                    frame = self.spatial_transform(frame)
                    # img = frame
                    # plt.imshow(img.transpose(0,2).transpose(0,1))
                    # plt.title("test data")
                    # plt.show()
                else:
                    frame = torchvision.transforms.ToTensor()(frame)  # (w,h,d)->(d,w,h)

                cur_video_seq.append(frame)
                cur_label_seq.append(label)

            else:
                return None

        clip = cur_video_seq
        target = torch.from_numpy(np.array(cur_label_seq, dtype=np.float32))
        # print("target",target)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  ###
        # # print("clip shape: ",np.shape(clip))
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return clip, target, self.video_index, self.frame_idx, video_path, label_path, self.mean_speed, self.std_speed

    def __len__(self):
        return self.real_subvideos_per_epoch


class agilesoda(data.Dataset):
    def __init__(self,
                 data_path="../../script/1022_total_df.csv",

                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16):

        ####

        label_df = pd.read_csv(data_path)
        label_list = list(label_df["Speeds"])
        video_list = list(label_df["Videos"])

        self.vframes = []
        self.subvideos_per_video = []
        self.timesteps = sample_duration
        # self.data_path = pd.DataFrame({"videos": video_list,
        #                                "speed": label_list})

        sum_f = 0
        sum_sq_f = 0
        temp_video_list = []
        temp_label_list = []

        for i in range(len(video_list)):
            # if i < 0:continue
            # if i >0 : break
            label_path = label_list[i]
            video_path = video_list[i]

            label_temp = self.read_ylabel("../" + label_path)
            temp_cap = cv2.VideoCapture("../" + video_path)
            len_label = len(label_temp)
            len_video = temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # print("len_label:",len_label)
            # print("len_video:", len_video)

            if len_video != len_label:
                continue
            else:
                temp_video_list.append(video_path)
                temp_label_list.append(label_path)
                # print("checked video:",video_path)

            self.vframes.append(len(label_temp))
            self.subvideos_per_video.append(len(label_temp) // self.timesteps)
            sum_f += np.sum(label_temp)
            sum_sq_f += np.sum(label_temp * label_temp)

        self.data_path = pd.DataFrame({"videos": temp_video_list,
                                       "speed": temp_label_list})
        # mean, var, std of speed targets
        self.mean_speed = (sum_f * 1.0 / np.sum(self.vframes)).astype(np.float32)
        self.var_speed = (sum_sq_f * 1.0 / np.sum(self.vframes) - self.mean_speed * self.mean_speed).astype(np.float32)
        self.std_speed = np.sqrt(self.var_speed)
        print("mean: ", self.mean_speed)
        print("var: ", self.var_speed)
        print("std: ", self.std_speed)
        self.real_subvideos_per_epoch = int(np.sum(self.subvideos_per_video))  # number of clips

        ####
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        print("--init--")
        # print("video path:", video_path)
        # print("annotation path:", annotation_path)
        print("sample_duration:", self.timesteps)
        print("number of videos: ", len(temp_video_list))
        print("real_subvideos_per_epoch:", self.real_subvideos_per_epoch)
        print("--continue--")
        self.video_index = 0
        self.frame_idx = 0

    def read_ylabel(self, y_label_path):
        train_y = []
        f = open(y_label_path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            train_y.append(float(line))
        # train_y = pd.DataFrame(train_y)
        # train_y.columns = ['real']
        train_y = np.array(train_y)
        return train_y

    def read_ylabel2k19(self, y_label_path):
        train_y = pd.DataFrame(np.linalg.norm(np.load(y_label_path), axis=1))
        train_y.columns = ['real']
        return train_y

    def __getitem__(self, sub_video_index):
        # get a random clip by sub_video_index (random number based on data len)
        ###
        index_count = 0
        frame_idx = 0
        video_index = 0
        # get clip location which are video index and frame index w.r.t its video

        for i in range(len(self.subvideos_per_video)):
            temp = index_count + self.subvideos_per_video[i]
            if sub_video_index < temp:
                video_index = i
                frame_idx = (sub_video_index - index_count) * self.timesteps

                break
            else:
                index_count = temp
        self.video_index = video_index
        self.frame_idx = frame_idx

        # get video path from video index
        # print()
        # print("sub_video_index:", sub_video_index)
        # print("video_inex: ", video_index)
        # print("frame index:", frame_idx)
        # print()
        video_path = "../" + self.data_path['videos'][video_index]
        label_path = "../" + self.data_path['speed'][video_index]

        # get annotation
        # labels = self.read_ylabel(label_path)  # speed of each frame ~ 1200
        labels = (self.read_ylabel(label_path) - self.mean_speed) / self.std_speed  # speed of each frame ~ 1200

        # get video and clip (also crop and resize clip)
        cap = cv2.VideoCapture(video_path)

        # video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        #         print("video path:",video_path)
        #         print("label path:",label_path)
        # print("Number of frames: ", video_length)
        cap.set(1, frame_idx)

        #         for i in range(frame_idx - 1):
        #             ret, frame = cap.read()

        cur_video_seq = []
        cur_label_seq = []
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()

        for seq_idx in range(self.timesteps):
            ret, frame = cap.read()

            if ret:
                # Handling Image

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).convert('RGB')

                label = labels[frame_idx + seq_idx]
                if self.spatial_transform is not None:

                    frame = self.spatial_transform(frame)
                    # img = frame
                    # plt.imshow(img.transpose(0,2).transpose(0,1))
                    # plt.title("test data")
                    # plt.show()
                else:
                    frame = torchvision.transforms.ToTensor()(frame)  # (w,h,d)->(d,w,h)

                cur_video_seq.append(frame)
                cur_label_seq.append(label)

            else:
                return None
        # ret,img =   cap.read()
        # plt.imshow(img )
        # plt.title("test data")
        # plt.show()
        clip = cur_video_seq
        target = torch.from_numpy(np.array(cur_label_seq, dtype=np.float32))
        # print("target",target)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  ###
        # # print("clip shape: ",np.shape(clip))
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return clip, target, self.video_index, self.frame_idx, video_path, label_path, self.mean_speed, self.std_speed

    def __len__(self):
        return self.real_subvideos_per_epoch


def get_agilesoda_dataset(opt=None, spatial_transform=None, temporal_transform=None,
                          target_transform=None):
    # assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']

    training_data = agilesoda(
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform)

    return training_data


def get_dataset(opt, spatial_transform, temporal_transform,
                target_transform):
    # assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']

    training_data = Comma2k19(
        opt.video_path,
        opt.annotation_path,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform)

    return training_data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def im_show(img):
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()


def demo_epoch(epoch, demo_loader, model, criterion, demo_logger, opt):
    print('demo at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end_time = time.time()

    # print("video path: ",demo_video_path.video_path)
    # cap  = cv2.VideoCapture(demo_video_path.video_path)

    # print("video size: ({},{})".format(int(height),int(width)))
    # ret,video = cap.read()

    # height = video.shape[0]
    # width = video.shape[1]

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cur_video_path = ''
    cur_label_path = ''
    height = 874
    width = 116
    out = None
    #     frame_index=-1
    start = False
    cur_video_index = -1
    cur_frame_index = -1
    for i, (inputs, targets2, video_index, frame_index, video_path, label_path, data_mean, data_std) in enumerate(
            demo_loader):

        # break
        # if i >1:
        #     break
        data_time.update(time.time() - end_time)
        inputs.float().mul_(2.0).sub_(1.0)
        ntargets = targets2
        ntargets = ntargets.cuda()
        inputs = Variable(inputs)
        ntargets = Variable(ntargets)

        noutputs = model(inputs)

        loss = criterion(noutputs, ntargets)

        losses.update(loss.data, inputs.size(0))
        # accuracies.update(acc, inputs.size(0))

        data_std = data_std[0]
        data_mean = data_mean[0]
        # show_loss = criterion(targets*data_std+data_mean,outputs*data_std+data_mean)

        # print("mse gpu: ", show_loss.data)

        targets = ntargets.mul(data_std).add(data_mean)
        outputs = noutputs.mul(data_std).add(data_mean)
        # targets = targets.data.item()
        # outputs = outputs.data.item()

        batch_time.update(time.time() - end_time)

        end_time = time.time()
        batch_size = len(inputs)

        for j in range(batch_size):
            #             if j ==0:
            #                 print("frame index:", frame_index[j].numpy(), " video index:", video_index[j].numpy())
            # print()

            train_y = np.linalg.norm(np.load(label_path[j]), axis=1)

            #             print("target load from file",train_y[frame_index[j] : frame_index[j]+16])
            #             print("target: ", targets[j])

            if frame_index[j].numpy() == 0 and start == False:
                start = True
            else:
                pass

            if video_path[j] != cur_video_path and cur_label_path != label_path[j]:
                if out != None:
                    print("release video {}".format(cur_video_index))
                    out.release()

                cur_video_path = video_path[j]
                cur_label_path = label_path[j]
                cur_video_index = video_index[j].numpy()
                cur_frame_index = frame_index[j].numpy()
                #                 print("video path:", video_path[j])
                #                 print("label_path:", label_path[j])
                cap = cv2.VideoCapture(cur_video_path)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(opt.result_path + "/demo{}.mp4".format(video_index[j]), fourcc, 20,
                                      (int(width) * 2 // 3, int(height) * 2 // 3))

            for k in range(16):

                ret, video = cap.read()

                image = video

                tar = targets[j, k]
                outp = outputs[j, k]
                mae = np.abs(np.diff([targets[j, k], outputs[j, k]]))[0]

                # plt.imshow(image)
                # plt.show()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # plt.imshow(image)
                # plt.show()
                # print("video shape: ",np.shape(video))

                cv2.putText(image, 'True : {:5.3f}'.format(tar), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                cv2.putText(image, 'Pred : {:5.3f}'.format(outp), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.putText(image, 'Error: {:5.3f}'.format(mae), (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                image = cv2.resize(image, (int(width) * 2 // 3, int(height) * 2 // 3), interpolation=Image.BILINEAR)
                out.write(image.astype(np.uint8))
                if (k == 0 and j == 0):
                    print(" video index:", video_index[j].numpy(), "frame:", k + j * 16 + i * batch_size * 16,
                          "target :", tar.data.cpu().numpy(), "output :", outp.data.cpu().numpy(), "mae :",
                          mae.data.cpu().numpy())
                    demo_logger.log(
                        {'video_index': cur_video_index, 'frame_index': cur_frame_index + k,
                         'target': targets[j, k].data.cpu().numpy(),
                         'prediction': outputs[j, k].data.cpu().numpy(), 'loss': losses.avg.data.cpu().numpy()})
                # cv2.imwrite('./demo/b{}c{}f{}.jpg'.format(i,j, k), image)

        # if i % (int(len(demo_loader) / 10)) == 0:
        #     print('Eval_Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
        #         epoch,
        #         i + 1,
        #         len(demo_loader),
        #         batch_time=batch_time,
        #         data_time=data_time,
        #         loss=losses))

    print("release video {}".format(cur_video_index))
    if out != None:
        out.release()

    # 'epoch', 'step', 'target', 'prediction', 'loss'

    return losses.avg


def sum_square_erorr(inputs, targets):
    se = np.sum(np.square(targets - inputs))
    return se


def sum_absolute_erorr(inputs, targets):
    # print("input shape",np.shape(inputs))
    # print("targets shape",np.shape(targets))
    ae = np.sum(np.abs(targets - inputs))
    return ae


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    # batch_logger = Logger(
    #     os.path.join(opt.result_path, 'val_batch_%d.log' % (epoch)),
    #     ['epoch', 'batch', 'iter', 'loss', 'loss_mae_val', 'loss_mae_avg', 'loss_mse_val', 'loss_mse_avg', 'lr'])

    val_batch_logger = Logger(
        os.path.join(opt.result_path, 'val_batch_%d.log' % (epoch)),
        ['epoch', 'index', 'loss_mae_val', 'loss_mae_avg', 'loss_mse_val', 'loss_mse_avg', 'target', 'output'])
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_mae = AverageMeter()
    losses_mse = AverageMeter()

    end_time = time.time()
    print()
    log_step = 10
    break_step = 400
    for i, (inputs, targets2, video_index, frame_index, video_path, label_path, data_mean, data_std) in enumerate(
            data_loader):
        if i == break_step: break
        # plt.imshow(np.squeeze(inputs[0].permute(1, 2, 3, 0)[0].numpy()))
        # plt.show()
        # continue
        data_time.update(time.time() - end_time)
        inputs.float().mul_(2.0).sub_(1.0)
        ntargets = targets2
        ntargets = ntargets.cuda()
        inputs = Variable(inputs)
        ntargets = Variable(ntargets)

        noutputs = model(inputs)

        loss = criterion(noutputs, ntargets)

        losses.update(loss.data, inputs.size(0))
        # accuracies.update(acc, inputs.size(0))

        data_std = data_std[0]
        data_mean = data_mean[0]
        # show_loss = criterion(targets*data_std+data_mean,outputs*data_std+data_mean)

        # print("mse gpu: ", show_loss.data)

        targets = ntargets.mul(data_std).add(data_mean)
        outputs = noutputs.mul(data_std).add(data_mean)

        # targets = ntargets.data.cpu() * data_std + data_mean
        # outputs = noutputs.data.cpu() * data_std + data_mean

        # tar = targets.data.cpu().numpy().flatten()
        # out = outputs.data.cpu().numpy().flatten()
        # mae_err = sum_absolute_erorr(tar, out) / len(tar)
        # mse_err = sum_square_erorr(tar, out) / len(tar)
        # losses_mae.update(mae_err, len(tar))
        # losses_mse.update(mse_err, len(tar))

        mae_err = criterion2(outputs, targets)
        mse_err = criterion(outputs, targets)
        losses_mae.update(mae_err.data, targets.size(0))
        losses_mse.update(mse_err.data, targets.size(0))

        # print("mse cpu: ",mse_err)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % log_step == 0:
            # print("num frames: ", len(tar))

            print('Epoch: [{0}][{1}/{2}] \t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'mae {mae.val:.4f} ({mae.avg:.4f}) \t'
                  'mse {mse.val:.4f} ({mse.avg:.4f}) \t'
                  'target {3:.4f}    output {4:.4f} \t'
                  'normed target {5}  normed output {6:.4f} \t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader), targets[0][0].data, outputs[0][0].data,
                ntargets[0][0].data, noutputs[0][0].data,
                loss=losses,
                mae=losses_mae,
                mse=losses_mse,
                batch_time=batch_time,
                data_time=data_time)
            )

            # print(losses_mse.avg.item())
            # batch_logger.log({
            #     'epoch': epoch,
            #     'batch': i + 1,
            #     'iter': (epoch) * len(data_loader) + (i + 1),
            #     'loss': losses.val.data.item(),
            #     'loss_mae_val': losses_mae.val.data.item() ,
            #     'loss_mae_avg': losses_mae.avg.data.item(),
            #     'loss_mse_val': losses_mse.val.data.item(),
            #     'loss_mse_avg': losses_mse.avg.data.item(),
            #     'lr': optimizer.param_groups[0]['lr']
            # })
            val_batch_logger.log(
                {'epoch': epoch,
                 'index': i,
                 'loss_mae_val': losses_mae.val.data.item(),
                 'loss_mae_avg': losses_mae.avg.data.item(),
                 'loss_mse_val': losses_mse.val.data.item(),
                 #                  'loss_mse_avg': losses_mae.avg.data.item(),
                 'loss_mse_avg': losses_mse.avg.data.item(),
                 'target': targets[0][0].data.item(),
                 'output': outputs[0][0].data.item()})

    print("[validation]  epoch: ", epoch, "losses", losses.avg.data.item(), 'loss_mae:', losses_mae.avg.data.item(),
          'loss_mse:',
          losses_mse.avg.data.item())
    logger.log({'epoch': epoch, 'loss': losses.avg.data.item(),
                'loss_mae': losses_mae.avg.data.item(),
                'loss_mse': losses_mse.avg.data.item(), })

    return losses.avg


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger):
    # batch_logger = Logger(
    #     os.path.join(opt.result_path, 'train_batch_%d.log'%(epoch)),
    #     ['epoch', 'batch', 'iter', 'loss', 'loss_mae_val', 'loss_mae_avg', 'loss_mse_val', 'loss_mse_avg', 'lr'])
    train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch_%d.log' % (epoch)),
        ['epoch', 'batch', 'loss', 'loss_mae_val', 'loss_mae_avg', 'loss_mse_val', 'loss_mse_avg', 'target', 'output',
         'lr'])
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()
    losses_mae = AverageMeter()
    losses_mse = AverageMeter()
    end_time = time.time()
    break_ratio = 6
    record_step = 10
    count = 0
    print()
    for i, (inputs, targets2, video_index, frame_index, video_path, label_path, data_mean, data_std) in enumerate(
            data_loader):
        #         if i == 2000: break
        # break
        # print("--train mean:",data_mean)
        # print("step:",i)
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:

            inputs.float().mul_(2.0).sub_(1.0)
            ntargets = targets2
            ntargets = ntargets.cuda()
            inputs = Variable(inputs)
            ntargets = Variable(ntargets)

            noutputs = model(inputs)

            loss = criterion(noutputs, ntargets)

            losses.update(loss.data, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_std = data_std[0]
            data_mean = data_mean[0]
            # show_loss = criterion(targets*data_std+data_mean,outputs*data_std+data_mean)

            # print("mse gpu: ", show_loss.data)

            targets = ntargets.mul(data_std).add(data_mean)
            outputs = noutputs.mul(data_std).add(data_mean)

            # targets = ntargets.data.cpu() * data_std + data_mean
            # outputs = noutputs.data.cpu() * data_std + data_mean

            # tar = targets.data.cpu().numpy().flatten()
            # out = outputs.data.cpu().numpy().flatten()
            # mae_err = sum_absolute_erorr(tar, out) / len(tar)
            # mse_err = sum_square_erorr(tar, out) / len(tar)
            # losses_mae.update(mae_err, len(tar))
            # losses_mse.update(mse_err, len(tar))

            mae_err = criterion2(outputs, targets)
            mse_err = criterion(outputs, targets)
            losses_mae.update(mae_err.data, targets.size(0))
            losses_mse.update(mse_err.data, targets.size(0))

            # print("mse cpu: ",mse_err)
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if i % record_step == 0:
                # print("num frames: ", len(tar))

                print('Epoch: [{0}][{1}/{2}] \t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'mae {mae.val:.4f} ({mae.avg:.4f}) \t'
                      'mse {mse.val:.4f} ({mse.avg:.4f}) \t'
                      'target {3:.4f}    output {4:.4f} \t'
                      'normed target {5}  normed output {6:.4f} \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(data_loader), targets[0][0].data, outputs[0][0].data,
                    ntargets[0][0].data, noutputs[0][0].data,
                    loss=losses,
                    mae=losses_mae,
                    mse=losses_mse,
                    batch_time=batch_time,
                    data_time=data_time)
                )

                # print(losses_mse.avg.item())
                # batch_logger.log({
                #     'epoch': epoch,
                #     'batch': i + 1,
                #     'iter': (epoch) * len(data_loader) + (i + 1),
                #     'loss': losses.val.data.item(),
                #     'loss_mae_val': losses_mae.val.data.item(),
                #     'loss_mae_avg': losses_mae.avg.data.item(),
                #     'loss_mse_val': losses_mse.val.data.item(),
                #     'loss_mse_avg': losses_mse.avg.data.item(),
                #     'lr': optimizer.param_groups[0]['lr']
                # })
                train_batch_logger.log({
                    'epoch': epoch,
                    'batch': i,
                    'loss': losses.val.data.item(),
                    'loss_mae_val': losses_mae.val.data.item(),
                    'loss_mae_avg': losses_mae.avg.data.item(),
                    'loss_mse_val': losses_mse.val.data.item(),
                    'loss_mse_avg': losses_mse.avg.data.item(),
                    'target': targets[0][0].data.item(),
                    'output': outputs[0][0].data.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })

            if i % (len(data_loader) // break_ratio) == 0 and i > 0:

                save_file_path = os.path.join(opt.result_path,
                                              'save_{}_{}.pth'.format(epoch, i))

                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
                count = count + 1
                if count == 1: break

    print("[train]  epoch: ", epoch, "losses", losses.avg.data.item(), 'loss_mae:', losses_mae.avg.data.item(),
          'loss_mse:', losses_mse.avg.data.item())
    epoch_logger.log({
        'epoch': epoch + 1,
        'loss': losses.avg.data.item(),
        'loss_mae': losses_mae.avg.data.item(),
        'loss_mse': losses_mse.avg.data.item(),
        'lr': optimizer.param_groups[0]['lr']
    })


#     save_file_path = os.path.join(opt.result_path,
#                                   'save_{}.pth'.format(epoch))

#     states = {
#         'epoch': epoch + 1,
#         'arch': opt.arch,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     torch.save(states, save_file_path)


if __name__ == '__main__':

    opt = parse_opts()
    #     random.seed(opt.manual_seed)
    #     np.random.seed(opt.manual_seed)
    #     torch.manual_seed(opt.manual_seed)
    #     torch.cuda.manual_seed(opt.manual_seed)

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    # opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    # opt.std = get_std(opt.norm_value)

    print("opt: ")
    print(opt)

    # opt.video_path = "../../dataset/comma2k19/*/*/*/global_pose/frame_velocities"
    # opt.annotation_path = "../../dataset/comma2k19/*/*/*/video.hevc"

    begin_epoch = 0
    end_epoch = opt.n_epochs
    opt.scales = [opt.initial_scale]
    # spatial_transform = None
    # crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c'])
    norm_method = Normalize([0, 0, 0], [1, 1, 1])

    model, parameters = generate_model(opt)

    print(model)
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
        criterion2 = criterion2.cuda()
    if not opt.no_train:
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value),
            norm_method
        ])

        # spatial_transform =None
        temporal_transform = None
        target_transform = None

        # target_transform = ClassLabel()

        # speed_data = get_agilesoda_dataset(opt=None, spatial_transform=spatial_transform,
        #                                    temporal_transform=temporal_transform, target_transform=target_transform)
        speed_data = get_dataset(opt, spatial_transform,
                                 temporal_transform, target_transform)

        num_data = speed_data.real_subvideos_per_epoch
        indices = range(num_data)

        cur_train_index = 0
        cur_val_index = int(num_data - 0.1 * num_data)

        print("number of training clips: ", cur_val_index)
        print("batch size: ", opt.batch_size)

        # cur_val_video_index =
        dataset_train_indices = indices[cur_train_index:cur_val_index]
        # dataset_val_indices = indices[cur_val_index:num_data]

        dataset_train = Subset(speed_data, indices=dataset_train_indices)
        # dataset_test= Subset(speed_data,indices=dataset_val_indices)

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)

        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'loss_mae', 'loss_mse', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)

    if not opt.no_val:
        spatial_transform = Compose([
            crop_method,
            # CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value),
            norm_method
        ])
        temporal_transform = None
        target_transform = None

        ###
        test_speed_data = get_dataset(opt, spatial_transform,
                                      temporal_transform, target_transform)

        # test_speed_data = get_agilesoda_dataset(opt=None, spatial_transform=spatial_transform,
        #                                         temporal_transform=temporal_transform,
        #                                         target_transform=target_transform)

        num_data = test_speed_data.real_subvideos_per_epoch
        indices = range(num_data)

        cur_train_index = 0
        cur_val_index = int(num_data - 0.1 * num_data)

        print("number of testing clips: ", int(0.1 * num_data))

        # cur_val_video_index =
        dataset_val_indices = indices[cur_val_index:num_data]

        dataset_test = Subset(test_speed_data, indices=dataset_val_indices)

        val_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'loss_mae', 'loss_mse'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        begin_epoch = opt.begin_epoch
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

        optimizer.param_groups[0]['lr'] = 0.000001
        # new learning rate

    print("begin_epoch:", begin_epoch)
    print("end_epoch:", end_epoch)
    print("batch_size: ", opt.batch_size)
    print("learning rate: ", optimizer.param_groups[0]['lr'])
    # run
    demo = False
    if demo == True:

        demo_logger = Logger(
            os.path.join(opt.result_path, 'demo.log'), ['video_index', 'frame_index', 'target', 'prediction', 'loss'])

        # demo_epoch(0, train_loader, model, criterion, demo_logger)
        demo_epoch(0, val_loader, model, criterion, demo_logger, opt)



    else:
        print("begin_epoch:", begin_epoch)
        for i in range(begin_epoch, end_epoch):

            if not opt.no_train:
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger)
            if not opt.no_val:
                validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)

            if not opt.no_train and not opt.no_val:
                scheduler.step(validation_loss)

# # '''
#     fig = plt.figure(figsize=(10,10))
#     for i, (inputs, targets2, video_index, frame_index, video_path, label_path,data_mean,data_std) in enumerate(train_loader):
#         print('i: ',i)
#
#         fig.add_subplot(3,2,i%6+1)
#
#         img = np.squeeze(inputs[0].permute(1, 2, 3, 0)[0].numpy())*255.0
#         plt.imshow(img.astype(np.int32()))
#         print("img shape: ", np.shape(img))
#         print("speed: ", targets2[0][0])
#         if i %6==5:
#             print('inputs[0][0]: ', inputs[0][0])
#             plt.show()
#             # break
#
#
#             fig = plt.figure(figsize=(10,10))

# fig = plt.figure(figsize=(10, 10))
# for i ,(inputs,targets ) in enumerate(val_loader):
#     print('i: ',i)
#
#     ax = fig.add_subplot(3,2,i%6+1)
#     ax.title.set_text('%d'%i)
#     img = np.squeeze(inputs[0].permute(1, 2, 3, 0)[0].numpy())
#     plt.imshow(img.astype(np.int32()))
#     print("img shape: ", np.shape(img))
#     print("speed: ", targets[0])
#     if i %6==5:
#         plt.show()
#         fig = plt.figure(figsize=(10, 10))
#         # break
#
#

#     for i, (inputs, targets2, video_index, frame_index, video_path, label_path, data_mean, data_std) in enumerate(val_loader):
#         if i ==3: break
#         if i ==0:
#             print("input shape",np.shape(inputs))
#         for j in range (5): # clips

#             for k in range (5): # frams
#                 print(i)
#                 print(k)

#                 img = np.squeeze(inputs[j].permute(1, 2, 3, 0)[k].numpy())

#                 img = ((img+1.0)*1.0/2.0)*255.0
#                 print(img.flatten()[:10])
#                 cv2.imwrite('./img2/b{}c{}f{}.jpg'.format(i,j,k), img)

# '''


'''
results file
resume file
learng rate
'''
