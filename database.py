from typing import List, Dict

import mysql.connector
import configparser


def obtaining_training_data(job_names: List[str], job_gpus: List[int]) -> Dict[str, Dict[int, Dict]]:
    """
    从数据库当中获取先前记录的JOB训练时间。

    :param job_names: JOB名称。
    :param job_gpus: JOB可用的GPU数量。
    :return: 返回一个可以根据JOB名称以及GPU数量来获取在指定GPU数量下JOB的epoch次数和epoch时间的字典。
    """

    cp = configparser.ConfigParser()
    cp.read('./database_config.ini', encoding='utf-8')

    conn = mysql.connector.connect(user=cp['debug']['user'],
                                   password=cp['debug']['password'],
                                   database=cp['debug']['database'])
    cursor = conn.cursor()

    jobs_training_time = {}
    for job_name in job_names:
        jobs_training_time.setdefault(job_name, {})
        for job_gpu in job_gpus:
            cursor.execute('SELECT epoch_num, epoch_time FROM training_times WHERE job_name=%s AND gpu_num=%s',
                           [job_name, job_gpu])
            value = cursor.fetchone()
            jobs_training_time[job_name][job_gpu] = {'epoch_num': value[0],
                                                     'epoch_time': value[1]}

    cursor.close()
    conn.close()

    return jobs_training_time
