import unittest

from database import get_training_data
from experiment import sequential_execution, parallel_execution, cal_individual_plan, optimus_execution, ga_execution
from genetic_algorithm import *


class MyTestCase(unittest.TestCase):
    def test_database(self):
        job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']
        training_data = get_training_data(job_names, 8)
        for i in training_data:
            print(i)
            print(training_data[i])

    def test_ga(self):
        job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']
        max_gpu_num = 8
        training_data = get_training_data(job_names, max_gpu_num)
        job_nums = len(job_names)
        job_orders = list(range(1, job_nums + 1))
        individual = Individual([random.choice(job_orders) for _ in range(job_nums)])
        print(individual)
        cal_individual_plan(individual, max_gpu_num, job_names, training_data)
        utilization_rate = individual.plan.cal_utilization_rate(max_gpu_num)
        print(f'总时间: {individual.plan.total_time}.')
        print(f'利用率: {utilization_rate}%.')
        print(f'方案:')
        for slice_list in individual.plan.plan:
            for ts in slice_list:
                print(ts)
            print('=' * 100)

    def test_iter_ga(self):
        job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']
        max_gpu_num = 8
        training_data = get_training_data(job_names, max_gpu_num)
        plan = ga_execution(job_names, max_gpu_num, training_data)
        utilization_rate = plan.cal_utilization_rate(max_gpu_num)
        print(f'总时间: {plan.total_time}.')
        print(f'利用率: {utilization_rate}%.')
        print(f'方案:')
        for slice_list in plan.plan:
            for ts in slice_list:
                print(ts)
            print('=' * 100)

    def test_seq(self):
        job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']
        max_gpu_num = 8
        training_data = get_training_data(job_names, max_gpu_num)
        plan = sequential_execution(job_names, max_gpu_num, training_data)
        utilization_rate = plan.cal_utilization_rate(max_gpu_num)
        print(f'总时间: {plan.total_time}.')
        print(f'利用率: {utilization_rate}%.')
        print(f'方案:')
        for slice_list in plan.plan:
            for ts in slice_list:
                print(ts)
            print('=' * 100)

    def test_parallel(self):
        job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']
        max_gpu_num = 4
        training_data = get_training_data(job_names, max_gpu_num)
        plan = parallel_execution(job_names, max_gpu_num, training_data)
        utilization_rate = plan.cal_utilization_rate(max_gpu_num)
        print(f'总时间: {plan.total_time}.')
        print(f'利用率: {utilization_rate}%.')
        print(f'方案:')
        for slice_list in plan.plan:
            for ts in slice_list:
                print(ts)
            print('=' * 100)

    def test_optimus(self):
        job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']
        max_gpu_num = 8
        training_data = get_training_data(job_names, max_gpu_num)
        plan = optimus_execution(job_names, max_gpu_num, training_data)
        utilization_rate = plan.cal_utilization_rate(max_gpu_num)
        print(f'总时间: {plan.total_time}.')
        print(f'利用率: {utilization_rate}%.')
        print(f'方案:')
        for slice_list in plan.plan:
            for ts in slice_list:
                print(ts)
            print('=' * 100)

    def test_something(self):
        job_names = ['alexnet', 'resnet50', 'resnext50', 'seresnet101', 'googlenet', 'vgg16', 'densenet201']
        max_gpu_num = 8
        data = get_training_data(job_names, max_gpu_num)
        new_group = [init_individual(job_names) for _ in range(10)]
        for i in new_group:
            print(i)
            cal_individual_plan(i, max_gpu_num, job_names, data, False)
            i.plan.print_plan()


if __name__ == '__main__':
    unittest.main()
