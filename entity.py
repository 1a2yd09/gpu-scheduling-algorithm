from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Job:
    """
    JOB对象，包含JOB名称、JOB顺序、GPU数量、epoch数量、epoch时间等信息。
    """
    name: str
    order: int
    gpu_num: int
    epoch_num: int
    epoch_time: float
    completion_time: float = 0
    after_job: Job = None


@dataclass
class Individual:
    """
    种群个体，包含调度方案以及该调度方案的总完成时间等信息。
    """
    orders: List[int]
    gpus: List[int] = None
    all_completion_time: int = 0
    solution: List[List[Job]] = None
    allocation: str = ''
