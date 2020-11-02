import argparse

from database import get_training_data
from experiment import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--iteration-times', default=1000, type=int)
parser.add_argument('-g', '--gpu-num', default=8, type=int)
parser.add_argument('-n', '--individual-num', default=50, type=int)
parser.add_argument('-c', '--category', default='image', type=str)
args = parser.parse_args()


def main():
    job_names = []
    with open(f'./job_name_data/{args.category}.txt', 'r') as f:
        for job_name in f:
            job_names.append(job_name.strip())
    gpu_num = args.gpu_num
    data = get_training_data(job_names, gpu_num)
    # 顺序调度:
    sequential_execution(job_names, gpu_num, data)
    # 并行调度:
    parallel_execution(job_names, gpu_num, data)
    # Optimus调度:
    optimus_execution(job_names, gpu_num, data)
    # 遗传算法调度(不考虑利用时间片):
    ga_execution(job_names, gpu_num, data, args, False)
    # 遗传算法调度(考虑利用时间片):
    ga_execution(job_names, gpu_num, data, args, True)


if __name__ == '__main__':
    main()
