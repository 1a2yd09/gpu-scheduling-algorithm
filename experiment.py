import itertools
from typing import List, Dict

from entity import Job, TrainingData, TimeSlice, Plan, Individual
from genetic_algorithm import init_individual, selection, cross_over, mutation_process, preferential_admission


def init_job_list(job_names: List[str],
                  job_orders: List[int],
                  job_gpus: List[int],
                  data: Dict[str, Dict[int, TrainingData]]) -> List[Job]:
    job_list = []
    for job_name, job_order, gpu_num in zip(job_names, job_orders, job_gpus):
        job = Job(job_name, job_order, gpu_num, data[job_name][gpu_num].epoch_num, data[job_name][gpu_num].epoch_time)
        job.cal_time()
        job_list.append(job)
    job_list.sort(key=lambda j: (j.order, j.completion_time))
    return job_list


def init_plan(final_group_list: List[List[Job]], max_gpu_num: int) -> Plan:
    plan = []
    for group in final_group_list:
        slice_list = []
        for job in group:
            slice_list.append(TimeSlice([job], job.gpu_num, job.completion_time))
        plan.append(slice_list)
    return Plan(plan, max_gpu_num)


def maximum_allocation(max_gpu_num: int,
                       job_list: List[Job],
                       data: Dict[str, Dict[int, TrainingData]]):
    remain_gpu_num = max_gpu_num - len(job_list)
    while remain_gpu_num > 0:
        max_time_job = max(job_list, key=lambda j: j.completion_time)
        max_time_job.add_gpu(1, data)
        remain_gpu_num -= 1
    job_list.sort(key=lambda j: j.completion_time)


def cal_individual_plan(individual: Individual,
                        max_gpu_num: int,
                        job_names: List[str],
                        data: Dict[str, Dict[int, TrainingData]]):
    job_list = init_job_list(job_names, individual.orders, [1] * len(job_names), data)
    group_list = [list(g) for k, g in itertools.groupby(job_list, key=lambda j: j.order)]
    final_group_list = []
    for group in group_list:
        if max_gpu_num > len(group):
            maximum_allocation(max_gpu_num, group, data)
            final_group_list.append(group)
        else:
            cut_group_list = [group[i:i + max_gpu_num] for i in range(0, len(group), max_gpu_num)]
            for cg in cut_group_list:
                final_group_list.append(cg)

    individual.plan = init_plan(final_group_list, max_gpu_num)

    for i in range(len(individual.plan.plan) - 1):
        current_slice_list = individual.plan.plan[i]
        max_slice = max(current_slice_list, key=lambda s: s.actual_length)
        available_slice_list = []
        for ts in current_slice_list:
            ts.cal_remain_length(max_slice)
            if ts.remain_length > 0:
                available_slice_list.append(ts)
        if len(available_slice_list) > 0:
            next_slice_list = sorted(individual.plan.plan[i + 1], key=lambda s: s.actual_length, reverse=True)
            # TODO: 可以考虑时间片长短和占有的GPU数目之间的关系来分配。
            for cas, ns in zip(available_slice_list, next_slice_list):
                after_job = ns.job_list.pop()
                epoch_time = data[after_job.name][cas.gpu_num].epoch_time
                reduce_epoch_num = int(cas.remain_length // epoch_time)
                reduce_epoch_num = reduce_epoch_num if reduce_epoch_num < after_job.epoch_num else after_job.epoch_num
                after_job.reduce_epoch_num(reduce_epoch_num)
                ns.add_job(after_job)
                new_job = Job(after_job.name, after_job.order, cas.gpu_num, reduce_epoch_num, epoch_time)
                new_job.cal_time()
                cas.add_job(new_job)

    individual.plan.cal_time()


def sequential_execution(job_names: List[str],
                         max_gpu_num: int,
                         data: Dict[str, Dict[int, TrainingData]]):
    job_list = init_job_list(job_names, list(range(1, len(job_names) + 1)), [max_gpu_num] * len(job_names), data)
    final_group_list = []
    for job in job_list:
        final_group_list.append([job])

    plan = init_plan(final_group_list, max_gpu_num)
    plan.print_plan()


def parallel_execution(job_names: List[str],
                       max_gpu_num: int,
                       data: Dict[str, Dict[int, TrainingData]]):
    job_list = init_job_list(job_names, [1] * len(job_names), [1] * len(job_names), data)
    if max_gpu_num > len(job_list):
        maximum_allocation(max_gpu_num, job_list, data)
        plan = init_plan([job_list], max_gpu_num)
    else:
        job_list.sort(key=lambda j: j.completion_time, reverse=True)
        slice_list = []
        for job in job_list[:max_gpu_num]:
            slice_list.append(TimeSlice([job], job.gpu_num, job.completion_time))
        for job in job_list[max_gpu_num:]:
            minimum_slice = min(slice_list, key=lambda s: s.actual_length)
            minimum_slice.add_job(job)
        plan = Plan([slice_list], max_gpu_num)
    plan.print_plan()


def optimus_execution(job_names: List[str],
                      max_gpu_num: int,
                      data: Dict[str, Dict[int, TrainingData]]):
    if max_gpu_num >= len(job_names):
        job_list = init_job_list(job_names, [1] * len(job_names), [1] * len(job_names), data)
        remain_gpu_num = max_gpu_num - len(job_list)
        while remain_gpu_num > 0:
            utility_list = []
            for job in job_list:
                epoch_time = data[job.name][job.gpu_num + 1].epoch_time
                # TODO: 效用公式有待商榷。
                utility_list.append((job.completion_time - job.epoch_num * epoch_time) / job.gpu_num)
            job = job_list[utility_list.index(max(utility_list))]
            job.add_gpu(1, data)
            remain_gpu_num -= 1
        job_list.sort(key=lambda j: j.completion_time)
        plan = init_plan([job_list], max_gpu_num)
        plan.print_plan()


def ga_execution(job_names: List[str],
                 max_gpu_num: int,
                 data: Dict[str, Dict[int, TrainingData]],
                 args):
    job_nums = len(job_names)
    job_orders = list(range(1, job_nums + 1))

    new_group = [init_individual(job_names) for _ in range(args.individual_num)]
    for individual in new_group:
        cal_individual_plan(individual, max_gpu_num, job_names, data)

    for _ in range(args.iteration_times):
        after_group = selection(new_group)
        cross_over(after_group, job_names)
        mutation_process(after_group, job_orders)
        for individual in after_group:
            cal_individual_plan(individual, max_gpu_num, job_names, data)
        new_group = preferential_admission(new_group, after_group)

    new_group[0].plan.print_plan()
