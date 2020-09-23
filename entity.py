from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Job:
    """
    JOB对象，包含JOB名称、JOB顺序、GPU数量等信息。
    """
    name: str
    order: int
    gpu_num: int


@dataclass
class Individual:
    """
    种群个体，包含调度方案、完成时间等信息。
    """
    orders: List[int]
    gpus: List[int]
    completion_time: int = 0
    solution: List[Tuple[List[int]]] = None
