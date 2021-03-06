import os
import cv2
import numpy as np
import os.path as osp
import matplotlib.cm as cm
import torch
#import torch.hub
from torchvision import models, transforms
import torch.nn as nn
import pdb
import torch.nn.functional as F

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable():
    classes = []
    with open("synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    #print("raw_image shape: ", raw_image.shape)
    # extract dog from colormap
    temp = cmap.astype(np.float)
    """TODO: tune the below parameters"""
    rgb_lower1 = np.array([0, 0, 35], dtype='uint8')
    rgb_upper1 = np.array([255, 255, 255], dtype='uint8')
    mask1 = cv2.inRange(temp, rgb_lower1, rgb_upper1)
    contours1, _ = cv2.findContours(mask1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour1 = sorted(contours1, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(contour1)
    dog = [x,y,w,h]
    # dog = raw_image[:, y:y+h, x:x+w]
    # dog = torch.unsqueeze(dog, 0)
    # dog = F.upsample(dog, (224, 224), mode="bilinear", align_corners=False)

    rgb_lower2 = np.array([120, 240, 100], dtype='uint8')
    rgb_upper2 = np.array([140, 255, 130], dtype='uint8')
    mask2 = cv2.inRange(temp, rgb_lower2, rgb_upper2)
    if (mask2 is not None):
        contours2, _ = cv2.findContours(mask2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour2 = sorted(contours2, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(contour2)
    dog_face = [x, y, w, h]
    #dog_face = torch.tensor(dog_face)
    ####
    return dog, dog_face


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def extract_object(original_image, cuda = 1):
    """
        Generate Grad-CAM at different layers of ResNet-152
        """
    output_dir = "./output"

    # device = get_device(cuda)
    device = torch.device("cuda" if cuda else "cpu")

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
  #  device = "cuda"
  #   model.to(device)
    model.eval()

    target_layer = "layer4"
    target_class = 243  # "bull mastif"


    # Images
    # images = original_image.to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(original_image)
#    ids_ = torch.LongTensor([[target_class]] * len(original_image)).to(device)
#     pdb.set_trace()
    gcam.backward(ids=ids)

    # Grad-CAM
    dogs = []
    red_bboxs = []
    regions = gcam.generate(target_layer=target_layer)
    for j in range(len(original_image)):
        dog, red_bbox = save_gradcam(gcam=regions[j, 0], raw_image=original_image[j])
        dogs.append(dog)
        red_bboxs.append(red_bbox)
        print(red_bbox)
    # tensor_dogs = torch.stack(dogs)
    # tensor_red_bbox = torch.stack(red_bboxs)
    #del gcam,model,dogs,red_bboxs
    return dogs, red_bboxs

