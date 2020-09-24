import random
import itertools
import bisect
import copy
import math
from typing import List, Dict

from entity import Individual, Job


def init_individual(job_nums: int, job_orders: List[int], job_gpus: List[int]) -> Individual:
    """
    随机初始化一个个体的调度方案。

    :param job_nums: JOB数量。
    :param job_orders: JOB可能的顺序数组。
    :param job_gpus: JOB可用的GPU数量数组。
    :return: 随机初始化的个体。
    """
    return Individual([random.choice(job_orders) for _ in range(job_nums)],
                      [random.choice(job_gpus) for _ in range(job_nums)])


def calculation_process(job_names: List[str],
                        gpu_nums: int,
                        group: List[Individual],
                        training_data: Dict[str, Dict[int, int]]):
    """
    计算个体当中的“实际”调度方案以及对应的完成时间。

    :param job_names: JOB名称数组。
    :param gpu_nums: 总资源数。
    :param group: 种群。
    :param training_data: 可以根据JOB名称以及GPU数量获得训练时间的数据字典。
    :return: 该函数内部会对个体信息直接进行修改，无返回值。
    """
    for individual in group:
        job_list = []
        # 将JOB名称和调度方案进行整合，[JOB名称, JOB顺序, JOB使用的GPU数量]:
        for z in zip(job_names, individual.orders, individual.gpus):
            job_list.append(Job(z[0], z[1], z[2]))
        # 将JOB对象按照顺序进行排序，然后按照顺序进行分组，即相同顺序的JOB为一组:
        job_list = sorted(job_list, key=lambda job: job.order)
        job_group_list = [tuple(g) for k, g in itertools.groupby(job_list, key=lambda job: job.order)]
        # 将随机出来的GPU数量组合，按比例进行实际分配，如果出现总和不为总资源数的情况，则将剩余的GPU数量随机给到任一JOB:
        for job_group in job_group_list:
            # 计算当前分组的GPU总和:
            current_group_all_gpu = sum([job.gpu_num for job in job_group])
            # 按比例并向下取整得到当前分组中每个JOB的实际分配GPU数量:
            for job in job_group:
                job.gpu_num = math.floor(job.gpu_num / current_group_all_gpu * gpu_nums)
            current_group_all_gpu = sum([job.gpu_num for job in job_group])
            # 如果当前分组的实际GPU总和不等于总资源数，则随机将剩余的GPU数量一一给到当前分组的任一JOB:
            if current_group_all_gpu != gpu_nums:
                remain_gpu_nums = gpu_nums - current_group_all_gpu
                while remain_gpu_nums:
                    index = random.choice(range(len(job_group)))
                    job_group[index].gpu_num += 1
                    remain_gpu_nums -= 1

        individual.solution = job_group_list
        # 计算当前个体调度方案的完成时间，它等于调度方案当中每个分组的最长完成时间之和:
        completion_time = 0
        for job_group in job_group_list:
            group_time = 0
            for job in job_group:
                # 如果出现了不可能的GPU数量，则该分组的完成时间直接为无穷大，相当于舍弃当前调度方案:
                # TODO:可能需要有更好的GPU分配方案。
                job_training_time = training_data[job.name].get(job.gpu_num, float('inf'))
                if job_training_time > group_time:
                    group_time = job_training_time
            completion_time += group_time

        individual.completion_time = completion_time


def selection(group: List[Individual]) -> List[Individual]:
    """
    选择过程。

    :param group: 种群数组。
    :return: 返回一个新的个体种群，这样后续的修改不会影响原始种群。
    """

    def adaptability_func(completion_time: int) -> float:
        """
        计算个体适应度。

        :param completion_time: 个体所表示的调度方案的总完成时间。
        :return: 总完成时间的倒数，即总完成时间越少，个体适应度越大。
        """
        return 1 / completion_time

    adaptability_list = []
    all_adaptability = 0

    for individual in group:
        # 计算个体适应度:
        adaptability = adaptability_func(individual.completion_time)
        # 记录个体适应度:
        adaptability_list.append(adaptability)
        # 累加个体适应度:
        all_adaptability += adaptability

    # 计算个体选择概率:
    isp = [a / all_adaptability for a in adaptability_list]
    # 计算累加概率:
    cp = list(itertools.accumulate(isp))
    rn = [random.random() for _ in range(len(group))]

    # 根据轮盘选择法从原始种群当中选择相同数量的个体组成新的种群。
    # 过程为获得等同种群个数的随机数数组，根据二分查找从累加概率数组当中获取个体索引，最后从种群当中获取个体。
    # 注意深拷贝操作是发生在个体对象上，如果对列表进行深拷贝，会发生意想不到的结果。
    # dp([id1, id1]) => [id2, id2]; [dp(id1), dp(id1)] => [id2, id3]，
    # 列表当中不可避免地会出现被重复选择的个体，如果是对列表进行深拷贝，则这些ID相同个体会转变为ID不同的另外一组ID相同个体，
    # 如果后续对其中一个对象进行修改，会影响到其它ID相同个体:
    return [copy.deepcopy(group[bisect.bisect_left(cp, rn[i])]) for i in range(len(group))]


def cross_over(group: List[Individual]):
    """
    交叉过程。

    :param group: 种群数组。
    :return: 直接对种群个体进行修改，不会返回新数组。
    """

    # 生成一个随机非重复的数组索引数组:
    cross_orders = random.sample(list(range(len(group))), len(group))
    # 两两一组，为每一个个体分配交叉个体:
    cross_couples = [(cross_orders[i], cross_orders[i + 1]) for i in range(0, len(cross_orders), 2)]
    job_num = len(group[0].orders)
    # 按照分组，两两进行调度方案当中的顺序编码和GPU数量编码交叉操作:
    for cross_couple in cross_couples:
        fos, fgs = group[cross_couple[0]].orders, group[cross_couple[0]].gpus
        sos, sgs = group[cross_couple[1]].orders, group[cross_couple[1]].gpus

        cross_point = random.choice(list(range(1, job_num)))
        fos, sos = fos[:cross_point] + sos[cross_point:], sos[:cross_point] + fos[cross_point:]
        cross_point = random.choice(list(range(1, job_num)))
        fgs, sgs = fgs[:cross_point] + sgs[cross_point:], sgs[:cross_point] + fgs[cross_point:]

        group[cross_couple[0]].orders, group[cross_couple[0]].gpus = fos, fgs
        group[cross_couple[1]].orders, group[cross_couple[1]].gpus = sos, sgs


def mutation_process(group: List[Individual], job_orders: List[int], job_gpus: List[int]):
    """
    变异过程，注意基因变异时，不会变异回原来的选项。

    :param group: 种群数组。
    :param job_orders: JOB可能顺序数组。
    :param job_gpus: JOB可用GPU数组。
    :return: 直接对原数组对象进行修改，不返回新数组。
    """

    def mutation(x: int, cl: List[int]) -> int:
        """
        该函数保证基因变异，不会变异回原来的选项，具体操作就是从可用选项当中去除当前选项，然后随机获得另外一个选项。
        注意remove操作会原地修改数组，此处应该传入一个原始数组的拷贝数组。

        :param x: 移除选项。
        :param cl: 可用选项数组的拷贝。
        :return: 随机选项。
        """
        cl.remove(x)
        return random.choice(cl)

    job_num = len(group[0].orders)
    for individual in group:
        ios, igs = individual.orders, individual.gpus

        mutation_point = random.choice(list(range(1, job_num + 1)))
        ios[mutation_point - 1] = mutation(ios[mutation_point - 1], job_orders[:])
        mutation_point = random.choice(list(range(1, job_num + 1)))
        igs[mutation_point - 1] = mutation(igs[mutation_point - 1], job_gpus[:])

        individual.orders, individual.gpus = ios, igs


def preferential_admission(origin_group: List[Individual], change_group: List[Individual]) -> List[Individual]:
    """
    择优过程。将原始种群和经过选择、交叉、变异后的种群合并，按照个体调度方案完成时间排序，选择出前一半的个体组成此次迭代的最后种群。

    :param origin_group: 原始种群。
    :param change_group: 演化种群。
    :return: 择优后的种群。
    """
    return sorted(origin_group + change_group, key=lambda i: i.completion_time)[:len(origin_group)]
