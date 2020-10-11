import configparser
from typing import List, Dict, Any

import mysql.connector


def obtaining_training_data(job_names: List[str], gpu_nums: int) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    从数据库当中获取训练数据。

    :param job_names: JOB名称数组。
    :param gpu_nums: JOB最大可用GPU数目。
    :return: 返回一个可以根据JOB名称以及GPU数量来获取指定训练数据的集合。
    """

    cp = configparser.ConfigParser()
    cp.read('./database_config.ini', encoding='utf-8')

    conn = mysql.connector.connect(user=cp['debug']['user'],
                                   password=cp['debug']['password'],
                                   database=cp['debug']['database'])
    cursor = conn.cursor()

    job_data = {}
    for job_name in job_names:
        job_data.setdefault(job_name, {})
        for job_gpu in range(1, gpu_nums + 1):
            cursor.execute('SELECT epoch_num, epoch_time FROM training_times WHERE job_name=%s AND gpu_num=%s',
                           [job_name, job_gpu])
            value = cursor.fetchone()
            job_data[job_name][job_gpu] = {'epoch_num': value[0], 'epoch_time': value[1]}

    cursor.close()
    conn.close()

    return job_data
