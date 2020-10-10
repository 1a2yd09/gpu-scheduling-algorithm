from typing import List, Dict, Any

from entity import Job


def utilization_rate(job_group_list: List[List[Job]],
                     all_completion_time: float,
                     gpu_nums: int) -> float:
    """
    计算调度方案的资源利用率。

    :param job_group_list: 调度方案。
    :param all_completion_time: 调度方案的总体完成时间。
    :param gpu_nums: 最大资源数。
    :return: 资源利用率。
    """
    # 记录调度方案中未利用的资源单位，计算公式为时间乘以资源数:
    all_unused_resource = 0
    for job_group in job_group_list:
        # 当前分组中的最大完成时间:
        group_max_completion_time = max(job_group, key=lambda j: j.completion_time).completion_time
        for job in job_group:
            execution_time = 0
            # 计算当前JOB和后续JOB（如果有）的执行时间之和:
            execution_time += job.completion_time
            if job.after_job:
                execution_time += job.after_job.completion_time
            # 未被利用的时间等于最大完成时间减去该执行时间:
            unused_time = group_max_completion_time - execution_time
            # 将未利用时间乘以所占有的资源数得到未利用的资源单位:
            all_unused_resource += unused_time * job.gpu_num
    # 用总资源单位扣除未利用的资源单位并除以总资源单位得到资源利用率:
    return round((all_completion_time * gpu_nums - all_unused_resource) / (all_completion_time * gpu_nums) * 100, 3)


def sequential_execution(job_names: List[str],
                         gpu_nums: int,
                         training_data: Dict[str, Dict[int, Dict[str, Any]]]):
    """
    顺序执行的总完成时间及其调度方案。

    :param job_names: JOB名称数组。
    :param gpu_nums: 最大可用GPU数量。
    :param training_data: 记录了epoch次数和时间的数据字典。
    """
    job_list = []
    for i in range(len(job_names)):
        # JOB按顺序并按照最大可用GPU数量进行训练:
        job = Job(job_names[i],
                  i + 1,
                  gpu_nums,
                  training_data[job_names[i]][gpu_nums]['epoch_num'],
                  training_data[job_names[i]][gpu_nums]['epoch_time'])
        job.completion_time = job.epoch_num * job.epoch_time
        job_list.append(job)
    # 顺序执行的总完成时间等于每个JOB的完成时间之和:
    all_completion_time = 0
    for job in job_list:
        all_completion_time += job.completion_time

    print('=' * 100)
    print(f'顺序执行时间: {round(all_completion_time / 60)}minutes.')
    print('顺序执行方案:')
    for job in job_list:
        job.completion_time = round(job.completion_time, 3)
        print(job)
    print('=' * 100)


def parallel_execution(job_names: List[str],
                       gpu_nums: int,
                       training_data: Dict[str, Dict[int, Dict[str, Any]]]):
    """
    并行执行的调度方案及其总体完成时间。

    :param job_names: JOB名称数组。
    :param gpu_nums: 最大可用GPU数量。
    :param training_data: 训练数据。
    """
    job_list = []
    for job_name in job_names:
        # 每个JOB预先分配1个GPU数量的资源进行训练:
        job = Job(job_name, 1, 1, training_data[job_name][1]['epoch_num'], training_data[job_name][1]['epoch_time'])
        job.completion_time = job.epoch_num * job.epoch_time
        job_list.append(job)
    # 如果资源总数大于等于JOB数量，则所有JOB一次性执行完毕:
    if len(job_names) <= gpu_nums:
        remain_gpu_nums = gpu_nums - len(job_list)
        while remain_gpu_nums > 0:
            # 按照当前完成时间大小分配原则分配剩余的资源:
            job = max(job_list, key=lambda j: j.completion_time)
            job.gpu_num += 1
            job.epoch_num = training_data[job.name][job.gpu_num]['epoch_num']
            job.epoch_time = training_data[job.name][job.gpu_num]['epoch_time']
            job.completion_time = job.epoch_num * job.epoch_time
            remain_gpu_nums -= 1
        # 按照JOB完成时间对调度方案进行排序，其中最大完成时间就是调度方案的总完成时间。
        job_list.sort(key=lambda j: j.completion_time)
        all_completion_time = job_list[-1].completion_time
        print('=' * 100)
        print('完成时间分配原则:')
        print(f'并行执行时间: {round(all_completion_time / 60)}minutes.')
        print(f'并行利用率: {utilization_rate([job_list], all_completion_time, gpu_nums)}%.')
        print('并行解决方案:')
        for job in job_list:
            job.completion_time = round(job.completion_time, 3)
            print(job)
        print('=' * 100)
    else:
        job_list.sort(key=lambda j: j.completion_time, reverse=True)
        job_group_list = []
        for i in range(gpu_nums):
            job_group = {'job_group': [job_list[i]], 'completion_time': job_list[i].completion_time}
            job_group_list.append(job_group)
        for job in job_list[gpu_nums:]:
            minimum_time_group = min(job_group_list, key=lambda jgd: jgd['completion_time'])
            minimum_time_group['job_group'].append(job)
            minimum_time_group['completion_time'] += job.completion_time
        completion_time = max(job_group_list, key=lambda jgd: jgd['completion_time'])['completion_time']
        new_job_group_list = []
        for jgd in job_group_list:
            new_job_group_list.append(jgd['job_group'])
        print('=' * 100)
        print('拼接时间分配原则:')
        print(f'并行执行时间: {round(completion_time / 60)}minutes.')
        unused_resource = 0
        for jgd in job_group_list:
            unused_resource += 1 * (completion_time - jgd['completion_time'])
        used_resource = completion_time * gpu_nums
        print(f'并行利用率: {round((used_resource - unused_resource) / used_resource * 100, 3)}%.')
        print('并行解决方案:')
        for jg in new_job_group_list:
            for job in jg:
                job.completion_time = round(job.completion_time, 3)
                print(job)
            print('-' * 100)
        print('=' * 100)


def optimus_execution(job_names: List[str],
                      gpu_nums: int,
                      training_data: Dict[str, Dict[int, Dict[str, Any]]]):
    """
    Optimus 根据边际效用资源分配原则得到调度方案。

    :param job_names: JOB名称数组。
    :param gpu_nums: 最大可用GPU数量。
    :param training_data: 训练数据。
    """
    if gpu_nums > len(job_names):
        job_list = []
        for job_name in job_names:
            # 每个JOB预先分配1个GPU数量的资源进行训练:
            job = Job(job_name, 1, 1, training_data[job_name][1]['epoch_num'], training_data[job_name][1]['epoch_time'])
            job.completion_time = job.epoch_num * job.epoch_time
            job_list.append(job)

        remain_gpu_nums = gpu_nums - len(job_list)
        # 根据边际效用规则进行剩余资源的分配:
        while remain_gpu_nums > 0:
            utility_list = []
            for job in job_list:
                # 假设将该资源给予当前JOB，获取相应的epoch时间:
                plus_epoch_time = training_data[job.name][job.gpu_num + 1]['epoch_time']
                # 将减少的时间除以占用的资源数得到该JOB的边际效用:
                utility = (job.epoch_time - plus_epoch_time) / job.gpu_num
                utility_list.append(utility)
            # 获取最大边际效用对应的JOB，将当前资源赋予它:
            job = job_list[utility_list.index(max(utility_list))]
            job.gpu_num += 1
            job.epoch_num = training_data[job.name][job.gpu_num]['epoch_num']
            job.epoch_time = training_data[job.name][job.gpu_num]['epoch_time']
            job.completion_time = job.epoch_num * job.epoch_time
            remain_gpu_nums -= 1
        job_list.sort(key=lambda j: j.completion_time)
        print('=' * 100)
        all_completion_time = job_list[-1].completion_time
        print(f'Optimus 执行时间: {round(all_completion_time / 60)}minutes.')
        print(f'Optimus 利用率: {utilization_rate([job_list], all_completion_time, gpu_nums)}%.')
        print('Optimus 调度方案:')
        for job in job_list:
            job.completion_time = round(job.completion_time, 3)
            print(job)
        print('=' * 100)
