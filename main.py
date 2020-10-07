import argparse

from database import obtaining_training_data
from genetic_algorithm import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--iteration-times', default=1000, type=int)
parser.add_argument('-g', '--gpu-nums', default=8, type=int)
parser.add_argument('-n', '--individual-nums', default=50, type=int)
parser.add_argument('-a', '--allocation', action='store_true')
args = parser.parse_args()

job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']

job_nums = len(job_names)
gpu_nums = args.gpu_nums
individual_nums = args.individual_nums
job_orders = list(range(1, job_nums + 1))
job_gpus = list(range(1, gpu_nums + 1))


def main(iteration_times: int):
    td = obtaining_training_data(job_names, job_gpus)
    new_group = [init_individual(job_nums, job_orders) for _ in range(individual_nums)]
    calculation_process(job_names, gpu_nums, new_group, td, args.allocation)

    for _ in range(iteration_times):
        # 注意选择过程返回的是一个和原始个体信息相同但ID不同的个体，这样后续对个体的修改不会影响到原始种群个体:
        after_group = selection(new_group)
        cross_over(after_group)
        mutation_process(after_group, job_orders)
        # 交叉、变异后个体信息产生变化，需要重新计算个体适应度:
        calculation_process(job_names, gpu_nums, after_group, td, args.allocation)
        new_group = preferential_admission(new_group, after_group)

    print(new_group[0].all_completion_time)
    for i in new_group[0].solution:
        for j in i:
            print(j)
        print('=' * 100)
    print(new_group[0].allocation)


if __name__ == '__main__':
    main(iteration_times=args.iteration_times)
