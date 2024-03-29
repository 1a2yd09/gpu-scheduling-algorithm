# GPU 集群上的深度学习多作业调度算法

综合多种调度算法给出分布式深度学习多作业在 GPU 集群上的调度次序以及资源分配方案，期望作业总体完成时间尽可能小，资源利用率尽可能高。

## 论文发表

- [Optimizing makespan and resource utilization for multi-DNN training in GPU cluster](https://doi.org/10.1016/j.future.2021.06.021)
- [面向 GPU 集群的动态资源调度方法](http://dx.doi.org/10.7544/issn1000-1239.202220149)

## 调度算法

- ☑️ 顺序调度
- ☑️ 并行调度
- ☑️ Optimus 调度
- ☑️ 遗传算法调度(不利用空闲时间片)
- ☑️ 遗传算法调度(利用空闲时间片)

## 调度过程

1. 收集每个作业在不同 GPU 数量下的单次迭代时间，与迭代次数相乘得到完成时间；
2. 根据每个调度算法各自的实现给出多作业的调度次序和资源分配方案；
3. 计算多作业总体完成时间以及训练过程中的资源利用率。

## 环境配置

1. `python=3.7.11`
2. `mysql-connector-python=8.0.18`

## 启动方式

1. 使用 sql 目录下的脚本文件创建对应数据表，记录模型名称、迭代次数、GPU 个数、迭代时间等信息。
2. 在命令行模式下，使用命令 `python main.py` 启动调度流程。

## 有关分布式训练导致的精度损失问题

- 知乎：[如何理解深度学习分布式训练中的 large batch size 与 learning rate 的关系？](https://www.zhihu.com/question/64134994/answer/216895968)
- 论文：[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
