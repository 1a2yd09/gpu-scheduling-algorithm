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
    completion_time: float

    def cal_completion_time(self):
        self.completion_time = self.epoch_num * self.epoch_time

    def add_gpu(self, gpu_num: int, data: Dict[str, Dict[int, TrainingData]]):
        self.gpu_num += gpu_num
        self.epoch_time = data[self.name][self.gpu_num].epoch_time
        self.cal_completion_time()

    def reduce_epoch_num(self, epoch_num: int):
        self.epoch_num -= epoch_num
        self.cal_completion_time()


@dataclass
class TimeSlice:
    job_list: List[Job]
    gpu_num: int
    actual_length: float = 0
    remain_length: float = 0

    def add_job(self, job: Job):
        self.job_list.append(job)
        self.cal_actual_length()

    def pop_job(self) -> Job:
        job = self.job_list.pop()
        self.cal_actual_length()
        return job

    def cal_actual_length(self):
        self.actual_length = 0
        for job in self.job_list:
            self.actual_length += job.completion_time

    def cal_remain_length(self, max_slice_length: float):
        self.remain_length = max_slice_length - self.actual_length


@dataclass
class Batch:
    slice_list: List[TimeSlice]
    max_slice_length: float = 0

    def arrange_batch(self):
        for ts in self.slice_list:
            ts.cal_actual_length()
        self.max_slice_length = max(self.slice_list, key=lambda s: s.actual_length).actual_length
        for ts in self.slice_list:
            ts.cal_remain_length(self.max_slice_length)
        self.slice_list.sort(key=lambda s: s.actual_length)

    def get_reverse_slice_list(self) -> List[TimeSlice]:
        self.slice_list.sort(key=lambda s: s.actual_length, reverse=True)
        return self.slice_list


@dataclass
class Plan:
    plan: List[Batch]
    max_gpu_num: int
    total_time: float = 0
    utilization_rate: float = 0

    def cal_total_time(self):
        self.total_time = 0
        for batch in self.plan:
            self.total_time += batch.max_slice_length

    def arrange_plan(self):
        for batch in self.plan:
            batch.arrange_batch()
        self.cal_total_time()

    def cal_utilization_rate(self):
        unused_resource = 0
        for batch in self.plan:
            for ts in batch.slice_list:
                unused_resource += ts.remain_length * ts.gpu_num
        used_resource = self.total_time * self.max_gpu_num
        self.utilization_rate = (used_resource - unused_resource) / used_resource * 100

    def print_plan(self):
        self.cal_utilization_rate()
        print(f'完成时间: {round(self.total_time / 60)}minutes.')
        print(f'利用率: {round(self.utilization_rate, 3)}%.')
        print(f'执行方案:')
        print('=' * 100)
        for batch in self.plan:
            for ts in batch.slice_list:
                print(ts)
            print('-' * 100)
        print('=' * 100)


@dataclass
class Individual:
    orders: List[int] = None
    plan: Plan = None


@dataclass
class TrainingData:
    epoch_num: int
    epoch_time: float
