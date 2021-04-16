# Optimizing makespan and resource utilization for multi-DNN training in GPU cluster

借助遗传算法生成多深度学习作业的 GPU 集群资源调度方案，并实现主流调度算法比较算法之间的性能。

## 运行方式

本地数据库建立如下数据表，修改`database_config.ini`文件中的数据库连接配置。

```mysql
create table training_times
(
    id         bigint auto_increment primary key,
    job_name   varchar(100) default 'job'     not null,
    dataset    varchar(100) default 'dataset' not null,
    batch_size int          default 0         not null,
    epoch_num  int          default 0         not null,
    gpu_num    int          default 0         not null,
    epoch_time float        default 0         not null
);
```

在命令行执行`python main.py`得到各类已实现算法的调度方案以及时间和空间效率。
