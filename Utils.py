import numpy as np
import matplotlib
import math
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# % matplotlib inline
# % config InlineBackend.figure_format = 'svg'

img_dim = 48

def plot(gen_data):
    pad_dim = 1
    paded = pad_dim + img_dim
    gen_data = gen_data.reshape(gen_data.shape[0], img_dim, img_dim)
    # math.ceil()函数返回数字的上入整数（取靠近0的整数）
    n = int(math.ceil(math.sqrt(gen_data.shape[0])))
    # np.pad()方法：扩充矩阵，其中‘constant’指的是连续填充相同的值
    # np.transpose()函数用于对换数组的维度
    gen_data = (np.pad(
        gen_data, [[0, n * n - gen_data.shape[0]], [pad_dim, 0], [pad_dim, 0]],
        'constant').reshape((n, n, paded, paded)).transpose((0, 2, 1, 3))
                .reshape((n * paded, n * paded)))
    fig = plt.figure(figsize=(8, 8))
    # plt.axis('off')中'off'表示关闭轴线和标签
    plt.axis('off')
    plt.imshow(gen_data, cmap='Greys_r', vmin=-1, vmax=1)
    return fig