import configparser
from typing import List, Dict

import mysql.connector

from entity import TrainingData


def get_training_data(job_names: List[str],
                      max_gpu_num: int) -> Dict[str, Dict[int, TrainingData]]:
    cp = configparser.ConfigParser()
    cp.read('./database_config.ini', encoding='utf-8')

    conn = mysql.connector.connect(user=cp['debug']['user'],
                                   password=cp['debug']['password'],
                                   database=cp['debug']['database'])
    cursor = conn.cursor()

    data = {}
    for job_name in job_names:
        data.setdefault(job_name, {})
        for gpu_num in range(1, max_gpu_num + 1):
            cursor.execute('SELECT epoch_num, epoch_time FROM training_times WHERE job_name=%s AND gpu_num=%s',
                           [job_name, gpu_num])
            result = cursor.fetchone()
            data[job_name][gpu_num] = TrainingData(result[0], result[1])

    cursor.close()
    conn.close()

    return data
