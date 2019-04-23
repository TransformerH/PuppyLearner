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

margin = 0.05


def choose_device(use_cuda=True):
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return device


def start(batch_size, n_epochs, learning_rate):
    plot_path = "./plot"

    device = choose_device()
    my_model = models.my_model()
    my_model = my_model.to(device=device)

    # Load dataset
    train_data, test_data, classes = load_datasets()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    criterion = nn.NLLLoss()
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
                original_images, labels = data
                inputs, red_bbox = extract_object(original_images)
                inputs = torch.squeeze(inputs, 1)
                print("size: ", inputs.size())
                if(device == "cuda"):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                # forward pass to get outputs
                bbox, pred1, pred2 = my_model(inputs)

                # calculate the loss
                loss1 = criterion(pred1, labels)

                loss2 = criterion(pred2, labels)

                loss3, _ = bbox_loss(bbox, red_bbox)
                loss3 = torch.tensor(loss3)

                temp1 = pred1[:, labels]
                temp1 = temp1.detach().numpy()
                temp2 = pred2[:, labels]
                temp2 = temp2.detach().numpy()
                loss4 = []
                for i in range(batch_size):
                    loss4.append(max(0, temp1[i][i] - temp2[i][i] + margin))
                loss4 = torch.tensor(loss4)

                loss = loss1 + loss2 + loss3 + loss4
                print(loss)

                loss.mean().backward()
                optimizer.step()

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
            if epoch % 10 == 0:      # save every 10 epochs
                if(loss < least_loss):
                    torch.save(my_model.state_dict(), 'checkpoint.pt')
                    least_loss = loss

        print('Finished Training')
        return loss_over_time


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
        original_images, labels = data
        inputs, red_bbox = extract_object(original_images)
        inputs = torch.squeeze(inputs, 1)
        if(device == "cuda"):
            inputs, labels = inputs.cuda(), labels.cuda()

        # forward pass to get outputs
        bbox, pred1, pred2 = my_model(inputs)

        # calculate the loss
        loss1 = criterion(pred1, labels)

        loss2 = criterion(pred2, labels)

        loss3, _ = bbox_loss(bbox, red_bbox)
        loss3 = torch.tensor(loss3)

        temp1 = pred1[:, labels]
        temp1 = temp1.detach().numpy()
        temp2 = pred2[:, labels]
        temp2 = temp2.detach().numpy()

        loss4 = []
        for i in range(batch_size):
            loss4.append(max(0, temp1[i][i] - temp2[i][i] + margin))
        loss4 =torch.tensor(loss4)

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

    # Visualize Sample Results (Runs until a batch contains a )
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    misclassification_found = False
    while(not misclassification_found):
        fig.clf()
        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        if(device == "cuda"):
            images, labels = images.cuda(), labels.cuda()
        # get predictions
        preds = np.squeeze(my_model(images).data.max(1, keepdim=True)[1].cpu().numpy())
        images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
            ax.imshow(images[idx])
            if preds[idx]==labels[idx]:
                ax.set_title("{}".format(classes[preds[idx]], classes[labels[idx]]), color="green")
            else:
                ax.set_title("({})\n{}".format(classes[labels[idx]], classes[preds[idx]]), color="red", pad=.4)
                misclassification_found = True
    if plot_path:
        plt.savefig(os.path.join(plot_path, "Results Visualization"))
    else:
        plt.show()
    plt.clf()



