# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision

num_classes = 200

def my_model():
    def __init__(self):
        super(my_model,self).__init__()
        vgg_model = torchvision.models.vgg16(pretrained=True)
        self.conv1_1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.conv1_2 = nn.Sequential(*list(vgg_model.features.children())[4:9])
        self.conv1_3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        num_features = self.classifier[6].in_features
        self.fc1_1 = nn.Linear(num_features, num_classes)
        self.fc1_2 = nn.Linear(num_features, 4)

        self.conv2_1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.conv2_2 = nn.Sequential(*list(vgg_model.features.children())[4:9])
        self.conv2_3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        self.fc2_1 = nn.Linear(num_features, num_classes)

    def forward(self,x):
        out1_1 = self.conv1_1(x)
        out1_2 = self.conv1_2(out1_1)
        out1_3 = self.conv1_3(out1_2)
        label1 = self.fc1_1(out1_3)
        bbox = self.fc1_2(out1_3)

        y = x[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]

        out2_1 = self.conv2_1(y)
        out2_2 = self.conv2_2(out2_1)
        out2_3 = self.conv2_3(out2_2)
        label2 = self.fc1(out2_3)

        return bbox, label1, label2


