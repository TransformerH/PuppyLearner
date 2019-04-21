# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pdb
num_classes = 200

class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        vgg_model = torchvision.models.vgg16(pretrained=True)
        # for params in vgg_model.parameters():
        #     params.requires_grad = False
        print(vgg_model)

        self.conv1 = nn.Sequential(*list(vgg_model.features.children()))
        #pdb.set_trace()
        #self=vgg_model
        num_features = vgg_model.classifier[0].in_features
        print("num features: ", num_features)
        self.fc1_1 = nn.Linear(num_features, num_classes)
        self.fc1_2 = nn.Linear(num_features, 4)

        self.conv2 = nn.Sequential(*list(vgg_model.features.children()))
        self.fc2_1 = nn.Linear(num_features, num_classes)

    def forward(self,x):

        out1 = self.conv1(x)
        out1 = out1.view([4, 25088])
        label1 = self.fc1_1(out1)
        bbox = self.fc1_2(out1)

        print("bbox: ", bbox)
        y = x[int(bbox[0]):int(bbox[0]+bbox[2]), int(bbox[1]):int(bbox[1]+bbox[3])]
        y = F.upsample(y, (224, 224))

        out2 = self.conv2(y)
        out2 = out2.view([4, 25088])
        label2 = self.fc1(out2)

        return bbox, label1, label2


