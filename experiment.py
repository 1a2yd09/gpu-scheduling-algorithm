from typing import List, Dict, Any

from entity import Job


def sequential_execution_time(job_names: List[str],
                              gpu_nums: int,
                              training_data: Dict[str, Dict[int, Dict[str, Any]]]):
    """
    顺序执行的总完成时间。

    :param job_names: JOB名称数组。
    :param gpu_nums: 最大可用GPU数量。
    :param training_data: 记录了epoch次数和时间的数据字典。
    :return: 完成时间。
    """
    job_list = []
    for i in range(len(job_names)):
        job = Job(job_names[i],
                  i + 1,
                  gpu_nums,
                  training_data[job_names[i]][gpu_nums]['epoch_num'],
                  training_data[job_names[i]][gpu_nums]['epoch_time'])
        job.completion_time = job.epoch_num * job.epoch_time
        job_list.append(job)

    all_completion_time = 0
    for job in job_list:
        all_completion_time += job.completion_time

    print('=' * 100)
    print('seq solution:')
    print(f'seq execution time: {round(all_completion_time / 60)}minutes.')
    for job in job_list:
        job.completion_time = round(job.completion_time, 3)
        print(job)
    print('=' * 100)


def parallel_execution_time(job_names: List[str],
                            gpu_nums: int,
                            training_data: Dict[str, Dict[int, Dict[str, Any]]]):
    job_list = []
    for job_name in job_names:
        job = Job(job_name, 1, 1, training_data[job_name][1]['epoch_num'], training_data[job_name][1]['epoch_time'])
        job.completion_time = job.epoch_num * job.epoch_time
        job_list.append(job)

    if len(job_names) <= gpu_nums:
        remain_gpu_nums = gpu_nums - len(job_list)
        while remain_gpu_nums > 0:
            job = max(job_list, key=lambda j: j.completion_time)
            job.gpu_num += 1
            job.epoch_num = training_data[job.name][job.gpu_num]['epoch_num']
            job.epoch_time = training_data[job.name][job.gpu_num]['epoch_time']
            job.completion_time = job.epoch_num * job.epoch_time
            remain_gpu_nums -= 1
        job_list.sort(key=lambda j: j.completion_time)
        print('=' * 100)
        print('pa solution:')
        print(f'pa execution time: {round(job_list[-1].completion_time / 60)}minutes.')
        print(f'pa utilization rate: {solution_utilization_rate([job_list], job_list[-1].completion_time, gpu_nums)}%.')
        for job in job_list:
            job.completion_time = round(job.completion_time, 3)
            print(job)
        print('=' * 100)
    else:
        job_list.sort(key=lambda j: j.completion_time)


def optimus_execution_time(job_names: List[str],
                           gpu_nums: int,
                           training_data: Dict[str, Dict[int, Dict[str, Any]]]):
    if gpu_nums > len(job_names):
        job_list = []
        for job_name in job_names:
            job = Job(job_name, 1, 1, training_data[job_name][1]['epoch_num'], training_data[job_name][1]['epoch_time'])
            job.completion_time = job.epoch_num * job.epoch_time
            job_list.append(job)

        remain_gpu_nums = gpu_nums - len(job_list)
        while remain_gpu_nums > 0:
            utility_list = []
            for job in job_list:
                plus_epoch_time = training_data[job.name][job.gpu_num + 1]['epoch_time']
                utility = (job.epoch_time - plus_epoch_time) / (job.gpu_num + 1)
                utility_list.append(utility)
            job = job_list[utility_list.index(max(utility_list))]
            job.gpu_num += 1
            job.epoch_num = training_data[job.name][job.gpu_num]['epoch_num']
            job.epoch_time = training_data[job.name][job.gpu_num]['epoch_time']
            job.completion_time = job.epoch_num * job.epoch_time
            remain_gpu_nums -= 1
        job_list.sort(key=lambda j: j.completion_time)
        print('=' * 100)
        print('op solution:')
        print(f'op execution time: {round(job_list[-1].completion_time / 60)}minutes.')
        print(f'op utilization rate: {solution_utilization_rate([job_list], job_list[-1].completion_time, gpu_nums)}%.')
        for job in job_list:
            job.completion_time = round(job.completion_time, 3)
            print(job)
        print('=' * 100)


def solution_utilization_rate(job_group_list: List[List[Job]],
                              all_completion_time: float,
                              gpu_nums: int) -> float:
    """
    计算调度方案的资源利用率。

    :param job_group_list: 解决方案。
    :param all_completion_time: 解决方案的总体完成时间。
    :param gpu_nums: 最大资源数。
    :return: 资源利用率
    """
    # 记录调度方案中未利用的资源单位，计算公式为时间乘以资源数:
    all_unused_resource = 0
    for job_group in job_group_list:
        # 当前分组中的最大完成时间:
        group_max_completion_time = job_group[-1].completion_time
        for job in job_group:
            execution_time = 0
            # 计算当前JOB和后续JOB（如果有）的执行时间:
            execution_time += job.completion_time
            if job.after_job:
                execution_time += job.after_job.completion_time
            # 未被利用的时间等于最大完成时间减去当前执行时间:
            unused_time = group_max_completion_time - execution_time
            # 将未利用时间乘以所占有的资源数得到未利用的资源单位:
            all_unused_resource += unused_time * job.gpu_num

    return round((all_completion_time * gpu_nums - all_unused_resource) / (all_completion_time * gpu_nums) * 100, 3)
