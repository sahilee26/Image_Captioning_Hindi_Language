from scipy import ndimage
from collections import Counter

import numpy as np
import pandas as pd
import hickle
import pickle
import os
import json

from PIL import Image
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


"""Reference Link: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # Batch normalization reduces the amount by what the hidden unit values shift around (covariance shift)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                weight_init.constant_(m.weight, 1)
                weight_init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

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

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f, encoding='latin1')
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, 2)
        print ('Saved %s..' %path)

def prepend(list, str): 
    # Using format() 
        str += '{0}'
        list = [str.format(i) for i in list] 
        return(list)


def main():

    batch_size = 100
    max_length = 15
    word_count_threshold = 1
     
    model=resnet34()
    orthogonal_model=torch.load('model_best.pth.tar', map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for keys, v in orthogonal_model['state_dict'].items():
        name = keys[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    preprocess_image = transforms.Compose([
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                normalize,
            ])

    image_dir = '/Data/santosh_1821cs03/Santosh/val_feature_extract/image/%2014_resized/'
    
    for split in ['train_hindi', 'val_hindi', 'test_hindi']:
        annotations = load_pickle('/Data/santosh_1821cs03/Santosh/val_feature_extract/data_hindi/%s/%s.annotations.pkl' % (split, split))
        save_path = './%s.features_orthogonal_cnn_49X512.hkl' % (split)
        image_path = list(annotations['file_name'].unique())
        n_examples = len(image_path)
        image_path=prepend(image_path, '/Data/santosh_1821cs03/Santosh/val_feature_extract/')
        all_feats = np.ndarray([n_examples, 49, 512], dtype=np.float32) #change here according to layer chosen

        for start, end in zip(range(0, n_examples, batch_size),
                              range(batch_size, n_examples + batch_size, batch_size)):
            image_batch_file = image_path[start:end]
            image_batch = torch.stack(list(map(lambda x: preprocess_image(Image.open(x).convert('RGB')), image_batch_file)))

            feats=model(image_batch)
            feats=feats.permute(0, 3, 2, 1)
            feats=feats.detach().numpy()
            all_feats[start:end, :] = feats.reshape(-1,49,512)
            print ("Processed %d %s features.." % (end, split))

        hickle.dump(all_feats, save_path)
        print ("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()