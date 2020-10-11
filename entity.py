from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Job:
    """
    JOB对象，包含JOB名称、分配的JOB顺序、分配的GPU数量、对应的epoch数量、对应的epoch时间、完成时间、后续JOB等信息。
    """
    name: str
    order: int
    gpu_num: int
    epoch_num: int
    epoch_time: float
    # JOB完成时间等于epoch数量乘以epoch时间:
    completion_time: float = 0
    # 后续JOB对象用于连接多个JOB对象，用于表示一组连续执行的JOB:
    after_job: Job = None


@dataclass
class Individual:
    """
    种群个体，包含调度方案以及该调度方案的总完成时间等信息。
    """
    orders: List[int] = None
    gpus: List[int] = None
    solution: List[List[Job]] = None
    all_completion_time: float = 0
