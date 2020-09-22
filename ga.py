import math
import random
import itertools
import mysql.connector
import bisect
import copy

conn = mysql.connector.connect(user='root', password='admin', database='learn_python')
cursor = conn.cursor()

job_nums = 5
gpu_nums = 8
job_orders = list(range(1, job_nums + 1))
job_gpus = list(range(1, gpu_nums + 1))
job_names = ['job', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

individual_nums = 50


def init_individual_info():
    # 这里我就保证了每个任务的GPU数量至少为1，但是后续按比例向下取整的时候还是会出现为零的情况，
    # 我只是保证了在初始化的种群中不会出现GPU数量零，但是后续演化过程中还是会出现零。
    # TODO:后续演化过程中也应该排除数量为零的情况
    return {'orders': [random.choice(job_orders) for _ in range(job_nums)],
            'gpus': [random.choice(job_gpus) for _ in range(job_nums)]}


# 交叉:
def cross_over(group):
    cg = copy.deepcopy(group)
    cross_orders = random.sample(list(range(individual_nums)), 50)
    # print(cross_orders)
    cross_couples = [[cross_orders[i], cross_orders[i + 1]] for i in range(0, len(cross_orders), 2)]
    # print(cross_couples)
    for cross_couple in cross_couples:
        # print(group[cross_couple[0]])
        # print(group[cross_couple[1]])
        fo = cg[cross_couple[0]]['orders']
        fg = cg[cross_couple[0]]['gpus']

        so = cg[cross_couple[1]]['orders']
        sg = cg[cross_couple[1]]['gpus']

        cross_point = random.choice(list(range(1, job_nums)))
        # print(cross_point)
        fo, so = fo[:cross_point] + so[cross_point:], so[:cross_point] + fo[cross_point:]
        cross_point = random.choice(list(range(1, job_nums)))
        # print(cross_point)
        fg, sg = fg[:cross_point] + sg[cross_point:], sg[:cross_point] + fg[cross_point:]

        cg[cross_couple[0]]['orders'], cg[cross_couple[0]]['gpus'] = fo, fg
        cg[cross_couple[1]]['orders'], cg[cross_couple[1]]['gpus'] = so, sg
        # print(group[cross_couple[0]])
        # print(group[cross_couple[1]])
        # print('=' * 100)
    return cg


def mutation(x, cl):
    cl.remove(x)
    return random.choice(cl)


# 变异:
def mutation_process(group):
    mg = copy.deepcopy(group)
    for i in mg:
        # print(i)
        io = i['orders']
        ig = i['gpus']

        mutation_point = random.choice(list(range(1, job_nums + 1)))
        # print(mutation_point)
        io[mutation_point - 1] = mutation(io[mutation_point - 1], job_orders[:])
        mutation_point = random.choice(list(range(1, job_nums + 1)))
        # print(mutation_point)
        ig[mutation_point - 1] = mutation(ig[mutation_point - 1], job_gpus[:])
        i['orders'] = io
        i['gpus'] = ig
        # print(i)
    return mg


def find_min_gpu_num_index(t):
    min_index = 0
    min_value = t[0][2]
    for i in range(len(t)):
        if t[i][2] < min_value:
            min_value = t[i][2]
            min_index = i
    return min_index


def computing_fitness(group):
    # 按执行顺序分组后，按比例求出每个JOB所需的GPU数量，并计算总完成时间即个体适应度:
    for individual in group:
        z = zip(job_orders, individual['orders'], individual['gpus'])
        z = [list(x) for x in z]
        z = sorted(z, key=lambda t: t[1])
        z = [tuple(g) for k, g in itertools.groupby(z, key=lambda t: t[1])]
        for i in range(len(z)):
            s = sum([x[2] for x in z[i]])
            # z[i]是分组，z[i][j]是JOB，先求分组的和，再按每一个JOB求比例，
            # 比例应该乘以总数减去JOB数，这样是为了保证每一个JOB都能有一个GPU。
            for j in range(len(z[i])):
                z[i][j][2] = math.floor(z[i][j][2] / s * gpu_nums)
            s = sum([x[2] for x in z[i]])
            if s != gpu_nums:
                # 当按照向下取整的规则分配GPU数量时，会出现总和达不到总资源数的情况，
                # 解决方法是将剩余的GPU数量每次拿出1个分配给分组当中GPU最少的JOB，
                # 该方法主要是为了防止出现GPU数量为零的JOB，实际上在解决这个问题后，可以考虑将多余的GPU分配给时间较长的JOB。
                # TODO:将GPU数量为零的问题解决后，考虑把多余的GPU分配给时间较长的JOB，或者仍随机分配。
                remain_gpu_nums = gpu_nums - s
                while remain_gpu_nums:
                    remain_gpu_nums -= 1
                    index = find_min_gpu_num_index(z[i])
                    z[i][index][2] += 1
            s = sum([x[2] for x in z[i]])
            if s != gpu_nums:
                print(z[i])
        # print(z)
        # print('=' * 100)
        # TODO:根据分组，按顺序遍历，取得JOB编号以及所需的GPU数量到数据库中查询总完成时间，取最长相加，得到个体适应度
        # 循环z列表，获取到每一个分组，循环分组，取到每一个JOB及其对应的GPU数量，到数据库中获取完成时间，
        # 那么这个分组的完成时间就取决于完成时间最长的那个JOB，把这个时间累加到总完成时间，循环分组结束后得到每一个个体对应调度方案的总完成时间。
        # 将个体表示以及总完成时间重新记录为一个新的列表。
        training_time = 0
        for g in z:
            group_time = 0
            for j in g:
                job_num = j[0]
                job_gpu = j[2]
                # print(g)
                # TODO:数据库读取应该是一次读取多次使用，因为没有牵涉到数据库修改语句，应该一次性读取到内存
                cursor.execute('SELECT training_time FROM training_times WHERE job_name=%s AND gpu_num=%s',
                               [job_names[job_num], job_gpu])
                values = cursor.fetchone()
                # print(values)
                if values[0] > group_time:
                    group_time = values[0]
            training_time += group_time
        individual['fitness'] = training_time


# TODO:利用轮盘选择法进行选择-复制
# 完成时间越少，则个体越优质，因此需要对每一个个体完成时间求倒数，完成时间越少，倒数就越大，
# 累加这些倒数，用每个个体的倒数除以倒数的和得到每一个个体的选择概率，所有个体选择概率的和应该为1。
# 计算累加概率到一个列表当中，从0到1之间随机出和个体个数相等的随机数，利用二分查找来确定被复制的个体。
def selection(group):
    # print(group)
    reversal_fitness_list = []
    all_reversal_fitness = 0
    for individual in group:
        reversal_fitness = 1 / individual['fitness']
        reversal_fitness_list.append(reversal_fitness)
        all_reversal_fitness += reversal_fitness
    # print(reversal_fitness_list)
    # print(all_reversal_fitness)
    individual_choice = [i / all_reversal_fitness for i in reversal_fitness_list]
    # print(individual_choice)
    accumulation_choice = list(itertools.accumulate(individual_choice))
    # print(accumulation_choice)
    random_number = [random.random() for _ in range(individual_nums)]
    # print(random_number)
    # individual_select = [bisect.bisect_left(accumulation_choice, random_number[i]) for i in range(individual_nums)]
    # print(individual_select)
    return [group[bisect.bisect_left(accumulation_choice, random_number[i])] for i in range(individual_nums)]


# TODO:从原始种群和变化后的种群中选择出完成时间最少的前individual_nums个个体组成新种群，使用这个新种群继续进行下一轮遗传演变
# 将两个种群组合成一个新列表，按照适应度进行升序排序，取前前individual_nums个个体组成新种群。
def preferential_admission(origin_group, change_group):
    # print('-' * 100)
    # print(sorted(origin_group, key=lambda d: d['fitness']))
    # print(sorted(change_group, key=lambda d: d['fitness']))
    # print('-' * 100)
    return sorted(origin_group + change_group, key=lambda d: d['fitness'])[:individual_nums]


new_group = [init_individual_info() for _ in range(individual_nums)]
computing_fitness(new_group)

for _ in range(1000):
    # 选择-复制是在原来种群上进行选择，不会改变个体信息:
    print(f'起始种群: {new_group}')
    after_group = selection(new_group)
    print(f'选择种群: {after_group}')
    after_group = cross_over(after_group)
    print(f'交叉种群: {after_group}')
    after_group = mutation_process(after_group)
    print(f'变异种群: {after_group}')
    # 交叉和变异会修改个体信息，因此需要重新计算适应度:
    computing_fitness(after_group)
    print(f'起始种群: {new_group}')
    print(f'变化种群: {after_group}')
    new_group = preferential_admission(new_group, after_group)
    print('=' * 100)

# 6912/23471
print(new_group)

cursor.close()
conn.close()
