import bisect
import copy
import itertools
import random
from typing import List

from entity import Individual


def init_individual(job_names: List[str]) -> Individual:
    job_orders = list(range(1, len(job_names) + 1))
    return Individual([random.choice(job_orders) for _ in range(len(job_names))])


def selection(group: List[Individual]) -> List[Individual]:
    def adaptability_func(total_time: float) -> float:
        return 1 / total_time

    adaptability_list = []
    all_adaptability = 0

    for individual in group:
        adaptability = adaptability_func(individual.plan.total_time)
        adaptability_list.append(adaptability)
        all_adaptability += adaptability

    isp = [a / all_adaptability for a in adaptability_list]
    cp = list(itertools.accumulate(isp))
    rn = [random.random() for _ in range(len(group))]

    return [copy.deepcopy(group[bisect.bisect_left(cp, rn[i])]) for i in range(len(group))]


def cross_over(group: List[Individual], job_names: List[str]):
    cross_orders = random.sample(list(range(len(group))), len(group))

    cross_couples = [(cross_orders[i], cross_orders[i + 1]) for i in range(0, len(cross_orders), 2)]

    for cross_couple in cross_couples:
        fos = group[cross_couple[0]].orders
        sos = group[cross_couple[1]].orders

        cross_point = random.choice(list(range(1, len(job_names))))
        fos, sos = fos[:cross_point] + sos[cross_point:], sos[:cross_point] + fos[cross_point:]

        group[cross_couple[0]].orders = fos
        group[cross_couple[1]].orders = sos


def mutation_process(group: List[Individual], job_orders: List[int]):
    def mutation(x: int, cl: List[int]) -> int:
        cl.remove(x)
        return random.choice(cl)

    for individual in group:
        ios = individual.orders

        mutation_point = random.choice(list(range(1, len(job_orders) + 1)))
        ios[mutation_point - 1] = mutation(ios[mutation_point - 1], job_orders[:])

        individual.orders = ios


def preferential_admission(origin_group: List[Individual], change_group: List[Individual]) -> List[Individual]:
    return sorted(origin_group + change_group, key=lambda i: i.plan.total_time)[:len(origin_group)]
