from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Job:
    name: str
    order: int
    gpu_num: int
    epoch_num: int
    epoch_time: float
    completion_time: float = 0

    def reduce_epoch_num(self, epoch_num: int):
        self.epoch_num -= epoch_num
        self.cal_time()

    def add_gpu(self, gpu_num: int, data: Dict[str, Dict[int, TrainingData]]):
        self.gpu_num += gpu_num
        self.epoch_time = data[self.name][self.gpu_num].epoch_time
        self.cal_time()

    def cal_time(self):
        self.completion_time = self.epoch_num * self.epoch_time


@dataclass
class TimeSlice:
    job_list: List[Job]
    gpu_num: int
    actual_length: float
    remain_length: float = 0

    def cal_actual_length(self):
        self.actual_length = 0
        for job in self.job_list:
            self.actual_length += job.completion_time

    def cal_remain_length(self, max_slice: TimeSlice):
        self.remain_length = max_slice.actual_length - self.actual_length

    def add_job(self, job: Job):
        self.job_list.append(job)
        self.cal_actual_length()


@dataclass
class Plan:
    plan: List[List[TimeSlice]] = None
    total_time: float = 0

    def cal_time(self):
        self.total_time = 0
        for slice_list in self.plan:
            self.total_time += max(slice_list, key=lambda s: s.actual_length).actual_length

    def arrange_plan(self):
        for slice_list in self.plan:
            max_slice = max(slice_list, key=lambda s: s.actual_length)
            for ts in slice_list:
                ts.cal_actual_length()
                ts.cal_remain_length(max_slice)
            slice_list.sort(key=lambda s: s.actual_length)

    def cal_utilization_rate(self, max_gpu_num: int):
        self.arrange_plan()
        self.cal_time()
        unused_resource = 0
        for slice_list in self.plan:
            for ts in slice_list:
                unused_resource += ts.remain_length * ts.gpu_num
        used_resource = self.total_time * max_gpu_num
        return (used_resource - unused_resource) / used_resource * 100


@dataclass
class Individual:
    orders: List[int] = None
    plan: Plan = None


@dataclass
class TrainingData:
    epoch_num: int
    epoch_time: float
