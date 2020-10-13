import argparse

from database import get_training_data
from experiment import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--iteration-times', default=1000, type=int)
parser.add_argument('-g', '--gpu-num', default=8, type=int)
parser.add_argument('-n', '--individual-num', default=50, type=int)
parser.add_argument('-a', '--allocation', action='store_true')
parser.add_argument('-s', '--image', action='store_true')
args = parser.parse_args()

image_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']
action_names = ['tsn', 'tsm', 'slowonly', 'slowfast', 'r2plus1d', 'i3d']


def main():
    job_names = image_names if args.image else action_names
    gpu_num = args.gpu_num
    data = get_training_data(job_names, gpu_num)

    sequential_execution(job_names, gpu_num, data)

    parallel_execution(job_names, gpu_num, data)

    optimus_execution(job_names, gpu_num, data)

    ga_execution(job_names, gpu_num, data, args)


if __name__ == '__main__':
    main()
