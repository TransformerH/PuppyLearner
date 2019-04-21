import argparse
from train import start


def parse_args():
    parser = argparse.ArgumentParser(description='Puppy Learner')
    parser.add_argument('--batch_size', help='Batch_size', default=64, type=int)
    parser.add_argument('--n_epochs', help='Number of Epochs', default=50, type=int)
    parser.add_argument('--learning_rate', help='Learning Rate', default=0.01, type=float)
    parser.add_argument('--saved_epoch', help='epoch of saved model', default=False, type=int)
    args = parser.parse_args()
    return args


def main():
    '''args = parse_args()
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    saved_epoch = args.saved_epoch'''

    batch_size = 4
    n_epochs = 2
    learning_rate = 0.001
    saved_epoch = 1

    start(batch_size, n_epochs, learning_rate, saved_epoch)


if __name__ == '__main__':
    main()
