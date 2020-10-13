import itertools
from typing import List, Dict

from entity import Job, TrainingData, TimeSlice, Plan, Individual, Batch
from genetic_algorithm import init_individual, selection, cross_over, mutation_process, preferential_admission


def get_job_list(job_names: List[str],
                 job_orders: List[int],
                 job_gpus: List[int],
                 data: Dict[str, Dict[int, TrainingData]]) -> List[Job]:
    job_list = []
    for job_name, job_order, gpu_num in zip(job_names, job_orders, job_gpus):
        job_epoch_num = data[job_name][gpu_num].epoch_num
        job_epoch_time = data[job_name][gpu_num].epoch_time
        job_list.append(Job(name=job_name,
                            order=job_order,
                            gpu_num=gpu_num,
                            epoch_num=job_epoch_num,
                            epoch_time=job_epoch_time,
                            completion_time=job_epoch_num * job_epoch_time))
    job_list.sort(key=lambda j: (j.order, j.completion_time))
    return job_list


def get_plan(group_list: List[List[Job]],
             max_gpu_num: int) -> Plan:
    batch_list = []
    for job_group in group_list:
        slice_list = []
        for job in job_group:
            slice_list.append(TimeSlice(job_list=[job],
                                        gpu_num=job.gpu_num))
        batch_list.append(Batch(slice_list=slice_list))
    plan = Plan(plan=batch_list,
                max_gpu_num=max_gpu_num)
    plan.arrange_plan()
    return plan


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
                        data: Dict[str, Dict[int, TrainingData]],
                        used_slice: bool):
    job_list = get_job_list(job_names, individual.orders, [1] * len(job_names), data)
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

    individual.plan = get_plan(final_group_list, max_gpu_num)

    if used_slice:
        for i in range(len(individual.plan.plan) - 1):
            current_batch = individual.plan.plan[i]
            available_slice_list = []
            for ts in current_batch.slice_list:
                if ts.remain_length > 0:
                    available_slice_list.append(ts)
            if len(available_slice_list) > 0:
                next_batch = individual.plan.plan[i + 1]
                # TODO: 可以考虑时间片长短和占有的GPU数目之间的关系来分配。
                for cas, nrs in zip(available_slice_list, next_batch.get_reverse_slice_list()):
                    job = nrs.pop_job()
                    epoch_time = data[job.name][cas.gpu_num].epoch_time
                    epoch_num = int(cas.remain_length // epoch_time)
                    epoch_num = epoch_num if epoch_num < job.epoch_num else job.epoch_num
                    job.reduce_epoch_num(epoch_num)
                    nrs.add_job(job)
                    new_job = Job(job.name, job.order, cas.gpu_num, epoch_num, epoch_time, epoch_num * epoch_time)
                    cas.add_job(new_job)
                # TODO: 感觉还是得把剩余时间的计算单独拿出来，不然这些不同的arrange之间会有重复操作。
                next_batch.arrange_batch()
                current_batch.arrange_batch()

    individual.plan.arrange_plan()


def sequential_execution(job_names: List[str],
                         max_gpu_num: int,
                         data: Dict[str, Dict[int, TrainingData]]):
    job_list = get_job_list(job_names, list(range(1, len(job_names) + 1)), [max_gpu_num] * len(job_names), data)
    final_group_list = []
    for job in job_list:
        final_group_list.append([job])

    plan = get_plan(final_group_list, max_gpu_num)
    plan.print_plan()


def parallel_execution(job_names: List[str],
                       max_gpu_num: int,
                       data: Dict[str, Dict[int, TrainingData]]):
    job_list = get_job_list(job_names, [1] * len(job_names), [1] * len(job_names), data)
    if max_gpu_num > len(job_list):
        maximum_allocation(max_gpu_num, job_list, data)
        plan = get_plan([job_list], max_gpu_num)
    else:
        job_list.sort(key=lambda j: j.completion_time, reverse=True)
        slice_list = []
        for job in job_list[:max_gpu_num]:
            slice_list.append(TimeSlice([job], job.gpu_num, job.completion_time))
        for job in job_list[max_gpu_num:]:
            minimum_slice = min(slice_list, key=lambda s: s.actual_length)
            minimum_slice.add_job(job)
        batch = Batch(slice_list)
        plan = Plan([batch], max_gpu_num)
        plan.arrange_plan()
    plan.print_plan()


def optimus_execution(job_names: List[str],
                      max_gpu_num: int,
                      data: Dict[str, Dict[int, TrainingData]]):
    if max_gpu_num >= len(job_names):
        job_list = get_job_list(job_names, [1] * len(job_names), [1] * len(job_names), data)
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
        plan = get_plan([job_list], max_gpu_num)
        plan.print_plan()


def ga_execution(job_names: List[str],
                 max_gpu_num: int,
                 data: Dict[str, Dict[int, TrainingData]],
                 args,
                 used_slice: bool):
    job_nums = len(job_names)
    job_orders = list(range(1, job_nums + 1))

    new_group = [init_individual(job_names) for _ in range(args.individual_num)]
    for individual in new_group:
        cal_individual_plan(individual, max_gpu_num, job_names, data, used_slice)

    for _ in range(args.iteration_times):
        after_group = selection(new_group)
        cross_over(after_group, job_names)
        mutation_process(after_group, job_orders)
        for individual in after_group:
            cal_individual_plan(individual, max_gpu_num, job_names, data, used_slice)
        new_group = preferential_admission(new_group, after_group)

    new_group[0].plan.print_plan()
