create table if not exists training_times
(
    id         bigint auto_increment
        primary key,
    job_name   varchar(100) default 'job'     not null,
    dataset    varchar(100) default 'dataset' not null,
    batch_size int          default 0         not null,
    epoch_num  int          default 0         not null,
    gpu_num    int          default 0         not null,
    epoch_time float        default 0         not null
);


