from database import obtaining_training_data
from genetic_algorithm import *

job_nums = 5
gpu_nums = 8
individual_nums = 50
job_orders = list(range(1, job_nums + 1))
job_gpus = list(range(1, gpu_nums + 1))
job_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def main(iteration_times: int):
    td = obtaining_training_data(job_names, job_gpus)
    new_group = [init_individual(job_nums, job_orders, job_gpus) for _ in range(individual_nums)]
    calculation_process(job_names, gpu_nums, new_group, td)

    for _ in range(iteration_times):
        # 注意选择过程返回的是一个和原始个体信息相同但ID不同的个体，这样后续对个体的修改不会影响到原始种群个体:
        after_group = selection(new_group)
        cross_over(after_group)
        mutation_process(after_group, job_orders, job_gpus)
        # 交叉、变异后个体信息产生变化，需要重新计算个体适应度:
        calculation_process(job_names, gpu_nums, after_group, td)
        new_group = preferential_admission(new_group, after_group)

    print(new_group[0])


if __name__ == '__main__':
    main(iteration_times=1000)
