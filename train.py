# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

from load import load_datasets
import models
from GIoU import bbox_loss
from PuppyDetection import extract_object
import torch.nn.functional as F
import pdb

margin = 0.05


def choose_device(use_cuda=True):
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return device


def start(batch_size, n_epochs, learning_rate):
    plot_path = "./plot"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    device = choose_device()
  #  device = "cuda"
    my_model = models.my_model()
    my_model = my_model.to(device=device)
    #extract_object = PuppyDetection.extract_object()
    #extract_object = extract_object.to(device=device)

    # Load dataset
    train_data, test_data, classes = load_datasets()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    criterion = nn.NLLLoss()
    criterion = criterion.to(device=device)
    optimizer = optim.SGD(my_model.parameters(), lr=learning_rate)

    # start training
    def train(n_epochs):
        least_loss = 999
        loss_over_time = []

        my_model.train()

        for epoch in range(n_epochs):
            running_loss = 0.0

            for batch_i, data in enumerate(train_loader):
                # get the input images and their corresponding labels


#################################  read boxes list from file  #################################
                dog_boxes = []
                dog_head_boxes = []
                
                original_images, labels, fileNames = data


                ##############to delete###########
                '''save_box_path = os.getcwd() + "/boxes/"
                print(save_box_path)
                try:
                    os.mkdir(save_box_path)
                except OSError:
                    print("Can't create folder")
                dog_box, dog_head_box = extract_object(original_images)
                # save to the file
                print("len: ", len(fileNames))
                for i in range(len(fileNames)):
                    save_fileName = fileNames[i].split('.')[0]
                    print("save_fileName: " + save_fileName)
                    save_file_path = os.path.join(save_box_path, save_fileName + '.txt')
                    with open(save_file_path, 'w') as f:
                        for item in dog_box[i]:
                            f.write("%s " % item)
                        f.write("\n")
                        for item in dog_head_box[i]:
                            f.write("%s " % item)
                    f.close()'''
                ############to delete##########




              #  original_images = original_images.cuda()
              #   original_images = original_images
                save_box_path = os.getcwd() + "/boxes/"
                for i in range(len(fileNames)):
                    save_fileName = fileNames[i].split('.')[0]
                    save_file_path = os.path.join(save_box_path, save_fileName + '.txt')
                    with open(save_file_path) as f:
                        line = f.readlines()
                    dog = list(map(int, line[0].split(' ')[:-1]))
                    dog_head = list(map(int, line[1].split(' ')[:-1]))
                    dog_boxes.append(dog)
                    dog_head_boxes.append(dog_head)
                dog_head_boxes = torch.tensor(dog_head_boxes)

                print("read test end")
#################################  read boxes list from file  #################################

                inputs = []
                for i in range(len(dog_boxes)):
                    input = torch.tensor(original_images[i][:, int(dog_boxes[i][0]):int(dog_boxes[i][0] + dog_boxes[i][2]),
                                          int(dog_boxes[i][1]):int(dog_boxes[i][1] + dog_boxes[i][3])])
                    input = torch.unsqueeze(input, dim=0)
                    input = F.upsample(input, (224, 224), mode='bilinear', align_corners=False)
                    input = torch.squeeze(input, dim=0)
                    inputs.append(input)
                inputs = torch.stack(inputs, dim=0)

                '''inputs = torch.squeeze(inputs, 1)
                print("size: ", inputs.size())'''
                if(device == "cuda"):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                # forward pass to get outputs
                #pdb.set_trace()

                bbox, pred1, pred2 = my_model(inputs)

                # calculate the loss
                loss1 = criterion(pred1, labels)

                loss2 = criterion(pred2, labels)

                loss3, _ = bbox_loss(bbox, dog_head_boxes)
                loss3 = torch.tensor(loss3)

                temp1 = pred1[:, labels]
                temp1 = temp1.cpu().detach().numpy()
                temp2 = pred2[:, labels]
                temp2 = temp2.cpu().detach().numpy()
                loss4 = []
                for i in range(batch_size):
                    loss4.append(max(0, temp1[i][i] - temp2[i][i] + margin))
                #pdb.set_trace()
                loss4 = np.mean(loss4)
         #       loss4 = torch.tensor(loss4).cuda()
                loss4 = torch.tensor(loss4)
                loss3 = torch.mean(loss3)
                # loss3= torch.mean(loss3).cuda()
                #pdb.set_trace()

                loss = loss1 + loss2 + loss3 + loss4
                print(loss)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print loss statistics
                # to convert loss into a scalar and add it to running_loss, we use .item()
                #running_loss += loss.item()
                running_loss += loss

                if batch_i % 45 == 44:    # print every 45 batches
                    avg_loss = running_loss/45
                    # record and print the avg loss over the 100 batches
                    loss_over_time.append(avg_loss)
                    print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, avg_loss))
                    running_loss = 0.0
                break #revised, not right
            if epoch % 10 == 0:      # save every 10 epochs
                if(loss < least_loss):
                    torch.save(my_model.state_dict(), 'checkpoint.pt')
                    least_loss = loss

        print('Finished Training')
        return loss_over_time

    #pdb.set_trace()
    training_loss = train(n_epochs)
    

    # visualize the loss as the network trained
    fig = plt.figure()
    plt.plot(45*np.arange(len(training_loss)), training_loss)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xlabel('Number of Batches', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.ylim(0, 5.5) # consistent scale
    plt.tight_layout()
    if plot_path:
        plt.savefig(os.path.join(plot_path, "Loss_Over_Time"))
        print("saved")
    else:
        plt.show()
    plt.clf()

    # initialize tensor and lists to monitor test loss and accuracy
    if(device == "cuda"):
        test_loss = torch.zeros(1).cuda()
    else:
        test_loss = torch.zeros(1)
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    # set the module to evaluation mode
    state = torch.load("checkpoint.pt")
    my_model.load_state_dict(state)
    my_model.eval()

    for batch_i, data in enumerate(test_loader):
        dog_boxes = []
        dog_head_boxes = []

        original_images, labels, fileNames = data

        ##############to delete###########
        '''save_box_path = os.getcwd() + "/boxes/"
        print(save_box_path)
        try:
            os.mkdir(save_box_path)
        except OSError:
            print("Can't create folder")
        dog_box, dog_head_box = extract_object(original_images)
        # save to the file
        print("len: ", len(fileNames))
        for i in range(len(fileNames)):
            save_fileName = fileNames[i].split('.')[0]
            print("save_fileName: " + save_fileName)
            save_file_path = os.path.join(save_box_path, save_fileName + '.txt')
            with open(save_file_path, 'w') as f:
                for item in dog_box[i]:
                    f.write("%s " % item)
                f.write("\n")
                for item in dog_head_box[i]:
                    f.write("%s " % item)
            f.close()'''
        ############to delete##########

        #  original_images = original_images.cuda()
        #   original_images = original_images
        save_box_path = os.getcwd() + "/boxes/"
        for i in range(len(fileNames)):
            save_fileName = fileNames[i].split('.')[0]
            save_file_path = os.path.join(save_box_path, save_fileName + '.txt')
            with open(save_file_path) as f:
                line = f.readlines()
            dog = list(map(int, line[0].split(' ')[:-1]))
            dog_head = list(map(int, line[1].split(' ')[:-1]))
            dog_boxes.append(dog)
            dog_head_boxes.append(dog_head)
        dog_head_boxes = torch.tensor(dog_head_boxes)

        print("read test end")
        #################################  read boxes list from file  #################################

        inputs = []
        for i in range(len(dog_boxes)):
            input = torch.tensor(original_images[i][:, int(dog_boxes[i][0]):int(dog_boxes[i][0] + dog_boxes[i][2]),
                                 int(dog_boxes[i][1]):int(dog_boxes[i][1] + dog_boxes[i][3])])
            input = torch.unsqueeze(input, dim=0)
            input = F.upsample(input, (224, 224), mode='bilinear', align_corners=False)
            input = torch.squeeze(input, dim=0)
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0)

        if(device == "cuda"):
            inputs, labels = inputs.cuda(), labels.cuda()

        # forward pass to get outputs
        bbox, pred1, pred2 = my_model(inputs)

        # calculate the loss
        loss1 = criterion(pred1, labels)

        loss2 = criterion(pred2, labels)

        loss3, _ = bbox_loss(bbox, dog_head_boxes)
        loss3 = torch.tensor(loss3)

        temp1 = pred1[:, labels]
        temp1 = temp1.cpu().detach().numpy()
        temp2 = pred2[:, labels]
        temp2 = temp2.cpu().detach().numpy()

        loss4 = []
        for i in range(batch_size):
            loss4.append(max(0, temp1[i][i] - temp2[i][i] + margin))
        loss4 =torch.tensor(loss4)
        #loss4 = np.mean(loss4)
        # loss4 = torch.mean(loss4).cuda()
        loss4 = torch.mean(loss4)

        # loss3= torch.mean(loss3).cuda()
        loss3 = torch.mean(loss3)

        loss = loss1 + loss2 + loss3 + loss4

        if(device == "cuda"):
            test_loss = test_loss + ((torch.ones(1).cuda() / (batch_i + 1)) * (loss.data - test_loss))
        else:
            test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))

        _, result2 = torch.max(pred2.data, 1)
        correct = np.squeeze(result2.eq(labels.data.view_as(result2)))

        # calculate test accuracy for *each* object class
        # we get the scalar value of correct items for a class, by calling `correct[i].item()`
        for l, c in zip(labels.data, correct):
            class_correct[l] += c.item()
            class_total[l] += 1
        break #revised, not right

    print('Test Loss: {:.6f}\n'.format(test_loss.cpu().numpy()[0]))

    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %30s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))


    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))



