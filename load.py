from torchvision import transforms
from stanford_dogs_data import dogs
from os.path import join, expanduser

root = expanduser("")
imagesets = join(root, 'DATASETS', 'IMAGE')


def load_datasets():
    input_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4069, 0.3683, 0.3585],
                             std=[0.2583, 0.2415, 0.2412])
    ])

    train_dataset = dogs(root=imagesets,
                             train=True,
                             cropped=False,
                             transform=input_transforms,
                             download=True)
    test_dataset = dogs(root=imagesets,
                            train=False,
                            cropped=False,
                            transform=input_transforms,
                            download=True)

    classes = train_dataset.classes

    print("Training set stats:")
    train_dataset.stats()
    print("Testing set stats:")
    test_dataset.stats()

    return train_dataset, test_dataset, classes
