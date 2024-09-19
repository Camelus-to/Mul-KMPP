import os

import torch.nn as nn
import torch.nn.functional as F
import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def get_roi_feature(feature_map_s, feature_map_m, template_s, template_m):
    roi_feature = list()

    special_ROI = [21, 22, 35, 36, 41, 42, 75, 76, 79, 80]

    # 90 ROIs
    for i in range(1, 91):

        if i not in special_ROI:

            roi_template = (template_s == i).astype(np.uint8)[np.newaxis, :, :, :]
            feature = np.sum(roi_template * feature_map_s, axis=(2, 3, 4)) / np.sum(roi_template)
            roi_feature.append(feature)
        else:
            roi_template = (template_m == i).astype(np.uint8)[np.newaxis, :, :, :]
            feature = np.sum(roi_template * feature_map_m, axis=(2, 3, 4)) / np.sum(roi_template)
            roi_feature.append(np.concatenate((feature, feature), axis=1))

    selected_elements = [[feature[i] for feature in roi_feature] for i in range(len(roi_feature[0]))]
    roi_features = np.array(selected_elements)
    return roi_features


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(inChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(outChannels)
        self.conv2 = nn.Conv3d(outChannels, outChannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm3d(outChannels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class DeConvLayer(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DeConvLayer, self).__init__()
        self.deconv = nn.ConvTranspose3d(inChannels, outChannels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(outChannels)

    def forward(self, x):
        out = F.relu(self.bn(self.deconv(x)))
        return out


class ConvAutoEncoder(nn.Module):
    def __init__(self, nChannels):
        super(ConvAutoEncoder, self).__init__()
        # encoder
        self.Conv1 = ConvBlock(1, nChannels)
        self.Conv2 = ConvBlock(nChannels, nChannels * 2)
        self.Conv3 = ConvBlock(nChannels * 2, nChannels * 4)
        self.Conv4 = ConvBlock(nChannels * 4, nChannels * 8)
        self.AvgPool = nn.AvgPool3d(kernel_size=2, stride=2)

        self.deConv1 = DeConvLayer(nChannels * 8, nChannels * 4)
        self.Conv5 = ConvBlock(nChannels * 4, nChannels * 4)
        self.deConv2 = DeConvLayer(nChannels * 4, nChannels * 2)
        self.Conv6 = ConvBlock(nChannels * 2, nChannels * 2)
        self.deConv3 = DeConvLayer(nChannels * 2, nChannels)
        self.Conv7 = ConvBlock(nChannels, nChannels)
        self.Conv8 = nn.Conv3d(nChannels, 1, kernel_size=1)

    def forward(self, x):
        feature = self.AvgPool(self.Conv1(x))
        feature = self.AvgPool(self.Conv2(feature))
        feature = self.AvgPool(self.Conv3(feature))
        feature = self.deConv1(self.Conv4(feature))
        feature = self.deConv2(self.Conv5(feature))
        feature = self.deConv3(self.Conv6(feature))
        out = self.Conv8(self.Conv7(feature))
        return out


class Feature_Extraction(nn.Module):
    def __init__(self, nChannels):
        super(Feature_Extraction, self).__init__()
        self.Conv1 = ConvBlock(1, nChannels)
        self.Conv2 = ConvBlock(nChannels, nChannels * 2)
        self.Conv3 = ConvBlock(nChannels * 2, nChannels * 4)
        self.Conv4 = ConvBlock(nChannels * 4, nChannels * 8)  # 增加通道数目，是原来得4倍，然后输出的通道是原来的8倍
        self.AvgPool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        feature = self.Conv1(x)
        feature = self.AvgPool(feature)
        feature = self.Conv2(feature)
        feature = self.AvgPool(feature)
        out_m = self.Conv3(feature)
        feature = self.AvgPool(out_m)
        out_s = self.Conv4(feature)
        return out_s, out_m


