from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Microsoft YaHei']

image_data_dict = {
    'SeqEx': [537, 301, 178],
    'NoReuse': [520, 266, 145],
    'Themis': [480, 298, 172],
    'Optimus': [520, 382, 172],
    'OMRU': [462, 254, 135]
}
action_data_dict = {
    'SeqEx': [1197, 654, 381],
    'NoReuse': [1171, 631, 349],
    'Themis': [1314, 1314, 380],
    'Optimus': [1671, 1356, 694],
    'OMRU': [1097, 566, 303]
}

image_ratio_dict = {
    'SeqEx': [537 / 462, 301 / 254, 178 / 135],
    'NoReuse': [520 / 462, 266 / 254, 145 / 135],
    'Themis': [480 / 462, 298 / 254, 172 / 135],
    'Optimus': [520 / 462, 382 / 254, 172 / 135]
}
action_ratio_dict = {
    'SeqEx': [1197 / 1097, 654 / 566, 381 / 303],
    'NoReuse': [1171 / 1097, 631 / 566, 349 / 303],
    'Themis': [1314 / 1097, 1314 / 566, 380 / 303],
    'Optimus': [1671 / 1097, 1356 / 566, 694 / 303]
}

image_resource_dict = {
    'SeqEx': [100, 100, 100],
    'NoReuse': [88.758, 98.219, 90.67],
    'Themis': [99.839, 77.562, 72.708],
    'Optimus': [88.758, 60.464, 72.708],
    'OMRU': [99.839, 99.936, 97.996]
}
action_resource_dict = {
    'SeqEx': [100, 100, 100],
    'NoReuse': [97.127, 98.265, 98.613],
    'Themis': [80.744, 40.372, 73.959],
    'Optimus': [63.502, 39.113, 40.954],
    'OMRU': [99.897, 99.964, 98.754]
}


def draw_experience_data(file_name: str,
                         data_dict: Dict[str, List[float]],
                         y_ticks: List[float],
                         ratio: int):
    x_data = [2, 4, 8]
    x_index = [1, 2, 3]

    colors = ['#0000ff', '#ff0000', '#000000', '#007f00', '#00bfbf']
    shapes = ['d', 'p', '^', 's', 'D', 'o']

    index = 0
    for model_name, data in data_dict.items():
        y_data = [x / ratio for x in data]
        plt.plot(x_index,
                 y_data,
                 color=colors[index],
                 marker=shapes[index],
                 linestyle='--',
                 markersize=10,
                 markeredgewidth=2,
                 linewidth=2,
                 fillstyle='none',
                 label=model_name)
        index += 1

    fontsize = 16
    plt.xticks(x_index, x_data, fontsize=fontsize)
    plt.yticks(y_ticks, fontsize=fontsize)
    plt.xlabel('节点数量', fontsize=fontsize)
    plt.ylabel('完工时间/h', fontsize=fontsize)
    plt.legend(framealpha=0.4, labelspacing=0.2, fontsize=fontsize)
    plt.tight_layout()

    plt.savefig(fname=f'./fig/ex-{file_name}.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    draw_experience_data('1', image_data_dict, [2, 4, 6, 8], 60)
    draw_experience_data('2', action_data_dict, [5, 10, 15, 20, 25], 60)

    draw_experience_data('11', image_ratio_dict, [1.1, 1.2, 1.3, 1.4, 1.5], 1)
    draw_experience_data('22', action_ratio_dict, [1.00, 1.25, 1.50, 1.75, 2.00, 2.25], 1)

    draw_experience_data('111', image_resource_dict, [60, 70, 80, 90, 100], 1)
    draw_experience_data('222', action_resource_dict, [40, 60, 80, 100], 1)
