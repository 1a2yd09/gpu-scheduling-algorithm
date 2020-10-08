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

job_nums = len(job_names)
gpu_nums = args.gpu_nums
individual_nums = args.individual_nums
# JOB可被分配的顺序:
job_orders = list(range(1, job_nums + 1))
# JOB可被分配的GPU数量:
job_gpus = list(range(1, gpu_nums + 1))


def main():
    td = obtaining_training_data(job_names, job_gpus)
    new_group = [init_individual(job_nums, job_orders) for _ in range(individual_nums)]
    calculation_process(job_names, gpu_nums, new_group, td, args.allocation)

    for _ in range(args.iteration_times):
        # 注意选择过程返回的是一个和原始个体信息相同但ID不同的个体，这样后续对个体的修改不会影响到原始种群个体:
        after_group = selection(new_group)
        cross_over(after_group)
        mutation_process(after_group, job_orders, job_nums)
        # 交叉、变异后个体信息产生变化，需要重新计算个体适应度:
        calculation_process(job_names, gpu_nums, after_group, td, args.allocation)
        new_group = preferential_admission(new_group, after_group)

    sequential_execution_time(job_names, gpu_nums, td)

    parallel_execution_time(job_names, gpu_nums, td)

    optimus_execution_time(job_names, gpu_nums, td)

    ga_optimum_individual = new_group[0]
    ga_optimum_solution = ga_optimum_individual.solution
    ga_optimum_time = ga_optimum_individual.all_completion_time
    print('=' * 100)
    print('ga solution:')
    print(f'ga execution time: {round(ga_optimum_time / 60)}minutes.')
    print(f'ga utilization rate: {solution_utilization_rate(ga_optimum_solution, ga_optimum_time, gpu_nums)}%.')
    for jg in new_group[0].solution:
        for jb in jg:
            jb.completion_time = round(jb.completion_time, 3)
            if jb.after_job:
                jb.after_job.completion_time = round(jb.after_job.completion_time, 3)
            print(jb)
        print('-' * 100)
    print('=' * 100)


if __name__ == '__main__':
    main()
