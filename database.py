from typing import List, Dict

import mysql.connector


def obtaining_training_data(job_names: List[str], job_gpus: List[int]) -> Dict[str, Dict[int, int]]:
    """
    从数据库当中获取记录的JOB训练时间。

    :param job_names: JOB名称。
    :param job_gpus: JOB可用的GPU数量。
    :return: 返回一个可以根据JOB名称以及GPU数量来获取在该GPU数量下JOB所需训练时间的字典。
    """
    conn = mysql.connector.connect(user='root', password='admin', database='learn_python')
    cursor = conn.cursor()

    jobs_training_time = {}
    for job_name in job_names:
        jobs_training_time.setdefault(job_name, {})
        for job_gpu in job_gpus:
            cursor.execute('SELECT training_time FROM training_times WHERE job_name=%s AND gpu_num=%s',
                           [job_name, job_gpu])
            jobs_training_time[job_name][job_gpu] = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return jobs_training_time
