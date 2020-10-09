import bisect
import copy
import itertools
import random
from typing import List, Dict, Any

from entity import Individual, Job
from experiment import utilization_rate


def init_individual(job_nums: int, job_orders: List[int]) -> Individual:
    """
    随机初始化一个个体的调度方案。

    :param job_nums: JOB数量。
    :param job_orders: JOB可能的顺序数组。
    :return: 随机初始化的个体。
    """
    # 根据JOB数量从JOB顺序数组中随机选择出若干个顺序号作为初始调度方案:
    return Individual([random.choice(job_orders) for _ in range(job_nums)])


def calculation_process(job_names: List[str],
                        gpu_nums: int,
                        group: List[Individual],
                        training_data: Dict[str, Dict[int, Dict[str, Any]]],
                        is_allocation: bool):
    """
    计算个体当中的GPU分配方案以及对应的完成时间。
    1.首先为每个JOB分配至少一个GPU，然后将剩余的GPU数量逐一分配给当前完成时间最长的JOB。
    2.然后考虑分组当中每个JOB距离最大完成时间的剩余时间片，将这些时间片依次分配给后续相邻分组当中完成时间从大到小的JOB上。
    为此需要获取这个时间片释放的GPU数量，然后获取后续JOB在这个GPU数量下的epoch时间，然后用时间片除以这个时间得到对应的epoch数量，
    就表示我们利用这一段时间片来提前训练该JOB，这样后续JOB的工作量就需要扣除这一部分epoch数量。

    :param job_names: JOB名称数组。
    :param gpu_nums: 总资源数。
    :param group: 种群。
    :param training_data: 可以根据JOB名称以及GPU数量获得训练时间的数据字典。
    :param is_allocation: 是否利用剩余时间片。
    :return: 该函数内部会对个体信息直接进行修改，无返回值。
    """
    for individual in group:
        job_list = []
        # 每次计算分配方案前，都需要为每个JOB分配至少一个GPU:
        individual.gpus = [1] * len(job_names)
        # 将JOB名称和调度方案进行整合，[JOB名称, JOB顺序, JOB使用的GPU数量, JOB在当前GPU数量下的epoch次数以及时间]:
        for z in zip(job_names, individual.orders, individual.gpus):
            job = Job(z[0],
                      z[1],
                      z[2],
                      training_data[z[0]][z[2]]['epoch_num'],
                      training_data[z[0]][z[2]]['epoch_time'])
            job.completion_time = job.epoch_num * job.epoch_time
            job_list.append(job)
        # 将JOB对象按照顺序进行排序，顺序相同则按照当前epoch时间进行排序（当前epoch时间都是在1个GPU数量下的时间）:
        job_list.sort(key=lambda job: (job.order, job.epoch_time))
        # 将JOB对象按照顺序进行分组，即有相同序号的JOB被分为一组:
        group_list = [list(g) for k, g in itertools.groupby(job_list, key=lambda job: job.order)]
        job_group_list = []
        # 对JOB数量超过资源总数的分组进行拆分，因为此时为每个JOB仅分配了1个GPU，如果还超出资源总数，那么这个分组将无法执行:
        for group in group_list:
            if len(group) <= gpu_nums:
                job_group_list.append(group)
            else:
                # 如果分组中的JOB数大于资源总数，将该分组拆分成多个小的分组:
                cut_group = [group[i:i + gpu_nums] for i in range(0, len(group), gpu_nums)]
                for cg in cut_group:
                    job_group_list.append(cg)
        # 对JOB数量小于资源总数的分组进行剩余资源分配，将剩余的GPU数量逐一分配给当前完成时间最长的JOB:
        for job_group in job_group_list:
            # 获取剩余的GPU数量:
            remain_gpu_nums = gpu_nums - len(job_group)
            # 逐一分配给当前完成时间最长的JOB:
            while remain_gpu_nums > 0:
                job = max(job_group, key=lambda job: job.completion_time)
                job.gpu_num += 1
                # 由于修改了JOB对象的GPU数量，需要重新获取epoch次数和epoch时间:
                job.epoch_num = training_data[job.name][job.gpu_num]['epoch_num']
                job.epoch_time = training_data[job.name][job.gpu_num]['epoch_time']
                job.completion_time = job.epoch_num * job.epoch_time
                remain_gpu_nums -= 1
            # 分配结束后，对分组再次按照完成时间进行排序:
            job_group.sort(key=lambda job: job.completion_time)

        if is_allocation:
            for i in range(len(job_group_list) - 1):
                current_group = job_group_list[i]
                next_group = job_group_list[i + 1]
                current_group_time_piece = []
                max_completion_time = current_group[-1].completion_time
                # 计算可用的剩余时间片:
                for job in current_group:
                    time_piece = max_completion_time - job.completion_time
                    if time_piece > 0:
                        # 如果剩余时间大于零，记录剩余时间以及当前JOB对象:
                        current_group_time_piece.append([time_piece, job])
                if len(current_group_time_piece) > 0:
                    tp_index = list(range(0, len(current_group_time_piece)))
                    ng_index = list(range(0, len(next_group)))
                    ng_index.sort(reverse=True)
                    # 将当前分组的剩余时间片从大到小和下一分组的JOB按完成时间从大到小进行组合:
                    # TODO: 这里如果再优化，可以将释放的GPU数量也作为参考因素。
                    allocation_list = list(zip(tp_index, ng_index))
                    for a in allocation_list:
                        # 当前分组的剩余时间片及对应的JOB:
                        tp, jb = current_group_time_piece[a[0]]
                        # 下一分组的JOB:
                        next_jb = next_group[a[1]]
                        # 下一分组的JOB在当前JOB释放的GPU数量下的epoch时间:
                        et = training_data[next_jb.name][jb.gpu_num]['epoch_time']
                        # 用剩余时间片整除这个epoch时间获得可提前训练的epoch数量:
                        re = int(tp // et) if int(tp // et) < next_jb.epoch_num else next_jb.epoch_num
                        # 下一分组的JOB工作量扣除该部分epoch数量:
                        next_jb.epoch_num -= re
                        # 重新计算完成时间，注意这里依旧使用下一分组的JOB原先分配的GPU数量下的epoch时间:
                        next_jb.completion_time = next_jb.epoch_num * next_jb.epoch_time
                        # 记录:
                        jb.after_job = Job(next_jb.name, next_jb.order, jb.gpu_num, re, et, re * et)
                    # 重新排序下个分组的JOB:
                    next_group.sort(key=lambda job: job.completion_time)

        individual.solution = job_group_list
        # 计算当前个体调度方案的总完成时间，它等于调度方案当中每个分组的最大完成时间之和:
        all_completion_time = 0
        for job_group in job_group_list:
            all_completion_time += job_group[-1].completion_time

        individual.all_completion_time = all_completion_time


def selection(group: List[Individual]) -> List[Individual]:
    """
    选择过程。

    :param group: 种群数组。
    :return: 返回一个新的个体种群，这样后续的修改不会影响原始种群。
    """

    def adaptability_func(all_completion_time: float) -> float:
        """
        计算个体适应度。

        :param all_completion_time: 个体所表示的调度方案的总完成时间。
        :return: 总完成时间的倒数，即总完成时间越少，个体适应度越大。
        """
        return 1 / all_completion_time

    adaptability_list = []
    all_adaptability = 0

    for individual in group:
        # 计算个体适应度:
        adaptability = adaptability_func(individual.all_completion_time)
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
    # 两两一组，即为每一个个体分配交叉个体:
    cross_couples = [(cross_orders[i], cross_orders[i + 1]) for i in range(0, len(cross_orders), 2)]
    job_num = len(group[0].orders)
    # 按照分组，两两进行调度方案当中的顺序编码交叉操作:
    for cross_couple in cross_couples:
        fos = group[cross_couple[0]].orders
        sos = group[cross_couple[1]].orders

        cross_point = random.choice(list(range(1, job_num)))
        fos, sos = fos[:cross_point] + sos[cross_point:], sos[:cross_point] + fos[cross_point:]

        group[cross_couple[0]].orders = fos
        group[cross_couple[1]].orders = sos


def mutation_process(group: List[Individual], job_orders: List[int], job_nums: int):
    """
    变异过程，注意基因变异时，不会变异回原来的选项。

    :param group: 种群数组。
    :param job_orders: JOB可能顺序数组。
    :param job_nums: JOB个数。
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

    for individual in group:
        ios = individual.orders

        mutation_point = random.choice(list(range(1, job_nums + 1)))
        ios[mutation_point - 1] = mutation(ios[mutation_point - 1], job_orders[:])

        individual.orders = ios


def preferential_admission(origin_group: List[Individual], change_group: List[Individual]) -> List[Individual]:
    """
    择优过程。将原始种群和经过选择、交叉、变异后的种群合并，按照个体调度方案完成时间排序，选择出前一半的个体组成此次迭代的最后种群。

    :param origin_group: 原始种群。
    :param change_group: 演化种群。
    :return: 择优后的种群。
    """
    return sorted(origin_group + change_group, key=lambda i: i.all_completion_time)[:len(origin_group)]


def ga_execution(job_names: List[str],
                 gpu_nums: int,
                 td: Dict[str, Dict[int, Dict[str, Any]]],
                 args):
    """
    遗传算法过程。

    :param job_names: JOB名称数组。
    :param gpu_nums: 最大可用GPU数量。
    :param td: 记录了epoch数量和时间的数据字典。
    :param args: 运行参数。
    """

    job_nums = len(job_names)
    job_orders = list(range(1, job_nums + 1))

    new_group = [init_individual(job_nums, job_orders) for _ in range(args.individual_nums)]
    calculation_process(job_names, gpu_nums, new_group, td, args.allocation)

    for _ in range(args.iteration_times):
        # 注意选择过程返回的是一个和原始个体信息相同但ID不同的个体，这样后续对个体的修改不会影响到原始种群个体:
        after_group = selection(new_group)
        cross_over(after_group)
        mutation_process(after_group, job_orders, job_nums)
        # 交叉、变异后个体信息产生变化，需要重新计算个体适应度:
        calculation_process(job_names, gpu_nums, after_group, td, args.allocation)
        new_group = preferential_admission(new_group, after_group)

    optimum_individual = new_group[0]
    optimum_solution = optimum_individual.solution
    optimum_time = optimum_individual.all_completion_time
    print('=' * 100)
    print(f'遗传算法执行时间: {round(optimum_time / 60)}minutes.')
    print(f'遗传算法利用率: {utilization_rate(optimum_solution, optimum_time, gpu_nums)}%.')
    print('遗传算法调度方案:')
    print('-' * 100)
    for jg in optimum_solution:
        for jb in jg:
            jb.completion_time = round(jb.completion_time, 3)
            if jb.after_job:
                jb.after_job.completion_time = round(jb.after_job.completion_time, 3)
            print(jb)
        print('-' * 100)
    print('=' * 100)
