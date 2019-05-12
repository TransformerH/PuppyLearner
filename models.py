# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import pdb
num_classes = 120

class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        vgg_model = torchvision.models.vgg16(pretrained=True)
        # for params in vgg_model.parameters():
        #     params.requires_grad = False
        print(vgg_model)

        self.conv1 = nn.Sequential(*list(vgg_model.features.children()))

        num_features = vgg_model.classifier[0].in_features
        self.fc1_1_1 = nn.Linear(num_features, 4096)
        self.fc1_1_2 = nn.Linear(4096, num_classes)

        self.fc1_2_1 = nn.Linear(num_features, 1024)
        self.fc1_2_2 = nn.Linear(1024, 4)

        self.conv2 = nn.Sequential(*list(vgg_model.features.children()))
        self.fc2_1 = nn.Linear(num_features, 1024)
        self.fc2_2 = nn.Linear(1024, num_classes)

    def forward(self, x, batch_size):

        out1 = self.conv1(x)
        out1 = out1.view([batch_size, 25088])
        out1_1 = F.relu(self.fc1_1_1(out1))
        label1 = self.fc1_1_2(out1_1)

        out1_2 = F.tanh(self.fc1_2_1(out1))
        bbox = F.sigmoid(self.fc1_2_2(out1_2))

        bbox = torch.mul(bbox, 224)


        y = []
        boxes = []
        for i in range(batch_size):
            if(int(bbox[i][2])<=int(bbox[i][0]) and int(bbox[i][3])<=int(bbox[i][1])):
                part = x[i][:, int(bbox[i][0]): 224, int(bbox[i][1]): 224]
                box = [int(bbox[i][0]), int(bbox[i][1]), 224, 224]
            elif(int(bbox[i][2])<=int(bbox[i][0])):
                part = x[i][:, int(bbox[i][0]):224, int(bbox[i][1]):int(bbox[i][3])]
                box = [int(bbox[i][0]), int(bbox[i][1]), 224, int(bbox[i][3])]
            elif(int(bbox[i][3])<=int(bbox[i][1])):
                part = x[i][:, int(bbox[i][0]):int(bbox[i][2]), int(bbox[i][1]):224]
                box = [int(bbox[i][0]), int(bbox[i][1]), int(bbox[i][2]), 224]
            else:
                part = x[i][:, int(bbox[i][0]):int(bbox[i][2]), int(bbox[i][1]):int(bbox[i][3])]
                box = [int(bbox[i][0]), int(bbox[i][1]), int(bbox[i][2]), int(bbox[i][3])]
            part = torch.unsqueeze(part, dim=0)
            part = F.upsample(part, (224, 224), mode='bilinear', align_corners=False)
            part = torch.squeeze(part, dim=0)
            y.append(part)
            boxes.append(np.asarray(box))

        y = torch.stack(y, dim=0)
        boxes = np.asarray(boxes)

        out2 = self.conv2(y)
        out2 = out2.view([batch_size, 25088])
        out2 = F.relu(self.fc2_1(out2))
        label2 = self.fc2_2(out2)

        #del y, part, x, out1

        return boxes, label1, label2


