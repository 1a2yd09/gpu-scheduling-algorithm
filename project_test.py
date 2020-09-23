from database import obtaining_training_data
from genetic_algorithm import *

job_nums = 5
gpu_nums = 8
individual_nums = 10
job_orders = list(range(1, job_nums + 1))
job_gpus = list(range(1, gpu_nums + 1))
job_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

td = obtaining_training_data(job_names, job_gpus)

init_group = [init_individual(job_nums, job_orders, job_gpus) for _ in range(individual_nums)]
calculation_process(job_names, gpu_nums, init_group, td)
print(f'初始: {init_group}')
after_group = selection(init_group)
print(f'选择: {after_group}')
cross_over(after_group)
print(f'交叉: {after_group}')
mutation_process(after_group, job_orders, job_gpus)
calculation_process(job_names, gpu_nums, after_group, td)
print(f'变异: {after_group}')
