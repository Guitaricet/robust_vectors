import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/42bin_haber',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    args = parser.parse_args()
    return args

class VisGraph():
    pass


if __name__ == '__main__':
    args = get_args()