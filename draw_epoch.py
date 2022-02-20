from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Microsoft YaHei']

alexnet = [25.245, 16.03, 14.637, 9.074, 8.987, 7.541, 6.419, 5.63]
resnet50 = [33.595, 18.84, 16.389, 13.064, 11.136, 9.81, 8.786, 8.068]
resnext50 = [71.954, 40.441, 28.105, 21.81, 18.178, 15.688, 13.771, 12.428]
seresnet101 = [89.27, 51.516, 34.69, 26.048, 21.045, 17.608, 14.956, 15.097]
googlenet = [41.505, 24.089, 17.129, 13.582, 11.537, 10.135, 9.056, 8.801]
vgg16 = [15.391, 10.07, 8.72, 6.592, 5.347, 4.486, 3.817, 3.248]
tsn = [31.725, 20.131, 14.546, 11.622, 10.042, 8.912, 8.005, 7.325]
tsm = [85.36, 48.126, 33.661, 26.119, 22.054, 19.147, 16.82, 15.074]
slowfast = [307.975, 162.67, 111.205, 84.415, 69.983, 59.671, 51.418, 45.227]
r2plus1d = [104.832, 60.892, 42.641, 33.096, 27.945, 24.258, 21.304, 19.086]
i3d = [228.03, 122.116, 83.978, 64.123, 53.426, 45.783, 39.666, 35.078]

image_data_dict = {
    'VGG16': [60, 40, 25, 20, 15, 10],
    'AlexNet': [90, 55, 40, 30, 20, 15],
    'ResNet50': [130, 70, 50, 40, 30, 20],
    'GoogLeNet': [160, 95, 60, 40, 30, 20],
    'ResNeXt50': [280, 150, 90, 50, 35, 25],
    'SE-ResNet101': [350, 180, 100, 60, 40, 30]
}

action_data_dict = {
    'TSN': [125, 60, 35, 20, 15, 10],
    'TSM': [350, 180, 95, 50, 30, 20],
    'R(2+1)D': [400, 200, 110, 60, 30, 15],
    'I3D': [900, 450, 225, 100, 50, 25],
    'SlowFast': [1240, 600, 310, 150, 75, 40]
}


def draw_epoch_data(file_name: str,
                    data_dict: Dict[str, List[int]],
                    y_ticks: List[int]) -> None:
    x_data = [1, 2, 4, 8, 16, 32]
    x_index = [1, 2, 3, 4, 5, 6]

    colors = ['#bf00bf', '#00bfbf', '#007f00', '#000000', '#ff0000', '#0000ff']
    shapes = ['d', 'p', '^', 's', 'D', 'o']

    index = len(data_dict) - 1
    for model_name, data in data_dict.items():
        plt.plot(x_index,
                 data,
                 color=colors[index],
                 marker=shapes[index],
                 linestyle='--',
                 markersize=10,
                 markeredgewidth=2,
                 linewidth=2,
                 fillstyle='none',
                 label=model_name)
        index -= 1

    fontsize = 16
    plt.xticks(x_index, x_data, fontsize=fontsize)
    plt.yticks(y_ticks, fontsize=fontsize)
    plt.xlabel('GPU数量', fontsize=fontsize)
    plt.ylabel('迭代回合时间/s', fontsize=fontsize)
    plt.legend(framealpha=0.4, labelspacing=0.2, fontsize=fontsize)
    plt.tight_layout()

    plt.savefig(fname=f'./fig/{file_name}.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    draw_epoch_data('1', image_data_dict, [0, 100, 200, 300])
    draw_epoch_data('2', action_data_dict, [0, 250, 500, 750, 1000, 1250])
