import os
import cv2
import numpy as np
import os.path as osp
import matplotlib.cm as cm
import torch
import torch.hub
from torchvision import models, transforms

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
    with open("test/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    print(image_path)
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0

    # extract dog from colormap
    temp = cmap.astype(np.float)
    """TODO: tune the below parameters"""
    rgb_lower = np.array([0, 0, 35], dtype='uint8')
    rgb_upper = np.array([255, 255, 255], dtype='uint8')
    mask = cv2.inRange(temp, rgb_lower, rgb_upper)
    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(raw_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    dog = raw_image[y:y+h, x:x+w]
    ####

    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

    return dog


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


def get_image_paths(image_folder):
    image_paths = []
    for filename in os.listdir(image_folder):
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            continue
        image_paths.append(os.path.join(image_folder, filename))
    return image_paths


def extract_object(image_folder, output_dir, cuda = None):
    """
        Generate Grad-CAM at different layers of ResNet-152
        """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    target_layer = "layer4"
    target_class = 243  # "bull mastif"

    image_paths = get_image_paths(image_folder)

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    print("Generating Grad-CAM @{}".format(target_layer))

    # Grad-CAM
    dogs = []
    regions = gcam.generate(target_layer=target_layer)
    for j in range(len(images)):
        dog = save_gradcam(
            filename=osp.join(
                output_dir,
                "{}-{}-gradcam-{}-{}.png".format(
                    j, "resnet152", target_layer, classes[target_class]
                ),
            ),
            gcam=regions[j, 0],
            raw_image=raw_images[j],
        )

        dogs.append(dog)

    return dogs

