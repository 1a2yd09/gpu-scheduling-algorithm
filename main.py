import argparse

from database import obtaining_training_data
from experiment import *
from genetic_algorithm import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--iteration-times', default=1000, type=int)
parser.add_argument('-g', '--gpu-nums', default=8, type=int)
parser.add_argument('-n', '--individual-nums', default=50, type=int)
parser.add_argument('-a', '--allocation', action='store_true')
args = parser.parse_args()

job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']

gpu_nums = args.gpu_nums
job_gpus = list(range(1, gpu_nums + 1))
td = obtaining_training_data(job_names, job_gpus)


def main():
    sequential_execution(job_names, gpu_nums, td)

    parallel_execution(job_names, gpu_nums, td)

    optimus_execution(job_names, gpu_nums, td)

    ga_execution(job_names, gpu_nums, td, args)


if __name__ == '__main__':
    main()
