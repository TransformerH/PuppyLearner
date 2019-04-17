import cv2
import argparse
from PuppyDetection import extract_object
from train import start


#image_folder = "data/Images"
image_folder = "test"
output_folder = "test_result"
train_data = "list/train_list.mat"
test_data = "list/test_list.mat"


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DML Training')
    parser.add_argument('--batch_size', help='Batch_size', default=64, type=int)
    parser.add_argument('--n_epochs', help='Number of Epochs', default=5, type=int)
    parser.add_argument('--learning_rate', help='Learning Rate', default=0.01, type=float)
    parser.add_argument('--saved_epoch', help='epoch of saved model', default=None, type=int)
    args = parser.parse_args()
    return args


def main(args):
    is_train = True

    dogs = extract_object(image_folder, output_folder)
    for dog in dogs:
        cv2.imshow("dog", dog)
        cv2.waitKey(0)
    #if(is_train == True):
    #    start(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
