# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
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
        self.fc1_1_1 = nn.Linear(num_features, 4096)
        self.fc1_1_2 = nn.Linear(4096, num_classes)

        self.fc1_2_1 = nn.Linear(num_features, 1024)
        self.fc1_2_2 = nn.Linear(1024, 4)

        self.conv2 = nn.Sequential(*list(vgg_model.features.children()))
        self.fc2_1 = nn.Linear(num_features, 1024)
        self.fc2_2 = nn.Linear(1024, num_classes)

    def forward(self,x):

        out1 = self.conv1(x)
        out1 = out1.view([4, 25088])
        out1_1 = F.relu(self.fc1_1_1(out1))
        label1 = self.fc1_1_2(out1_1)

        out1_2 = F.tanh(self.fc1_2_1(out1))
        bbox = F.sigmoid(self.fc1_2_2(out1_2))

        bbox = torch.mul(bbox, 224)

        p1 = torch.tensor(
            x[0][:, int(bbox[0][0]):int(bbox[0][0] + bbox[0][2]),
            int(bbox[0][1]):int(bbox[0][1] + bbox[0][3])])
        p2 = torch.tensor(
            x[1][:, int(bbox[1][0]):int(bbox[1][0] + bbox[1][2]),
            int(bbox[1][1]):int(bbox[1][1] + bbox[1][3])])
        p3 = torch.tensor(
            x[2][:, int(bbox[2][0]):int(bbox[2][0] + bbox[2][2]),
            int(bbox[2][1]):int(bbox[2][1] + bbox[2][3])])
        p4 = torch.tensor(
            x[3][:, int(bbox[3][0]):int(bbox[3][0] + bbox[3][2]),
            int(bbox[3][1]):int(bbox[3][1] + bbox[3][3])])

        #p1 = torch.tensor(p1)
        p1 = torch.unsqueeze(p1, dim=0)
        p1 = F.upsample(p1, (224, 224), mode='bilinear',align_corners=False)
        p1 = torch.squeeze(p1, dim=0)

        #p2 = torch.tensor(p2)
        p2 = torch.unsqueeze(p2, dim=0)
        p2 = F.upsample(p2, (224, 224), mode='bilinear',align_corners=False)
        p2 = torch.squeeze(p2, dim=0)

        #p3 = torch.tensor(p3)
        p3 = torch.unsqueeze(p3, dim=0)
        p3 = F.upsample(p3, (224, 224), mode='bilinear',align_corners=False)
        p3 = torch.squeeze(p3, dim=0)

        #p4 = torch.tensor(p4)
        p4 = torch.unsqueeze(p4, dim=0)
        p4 = F.upsample(p4, (224, 224), mode='bilinear',align_corners=False)
        p4 = torch.squeeze(p4, dim=0)

        y = [p1, p2, p3, p4]

        y = torch.stack(y, dim=0)


        out2 = self.conv2(y)
        out2 = out2.view([4, 25088])
        out2 = F.relu(self.fc2_1(out2))
        label2 = self.fc2_2(out2)

        del y, out2, p1, p2, p3, p4, x, out1

        return bbox, label1, label2


