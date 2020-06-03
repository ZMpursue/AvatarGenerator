import sys
import os
import matplotlib
import PIL
import six
import numpy as np
import math
import time
import paddle
import paddle.fluid as fluid

use_cudnn = True
use_gpu = True
n = 0
def bn(x, name=None, act=None,momentum=0.5):
    return fluid.layers.batch_norm(
        x,
        param_attr=name + '1',
        # 指定权重参数属性的对象
        bias_attr=name + '2',
        # 指定偏置的属性的对象
        moving_mean_name=name + '3',
        # moving_mean的名称
        moving_variance_name=name + '4',
        # moving_variance的名称
        name=name,
        act=act,
        momentum=momentum,
    )


###卷积池化组
def conv(x, num_filters,name=None, act=None):
    return fluid.nets.simple_img_conv_pool(
        input=x,
        filter_size=5,
        num_filters=num_filters,
        pool_size=2,
        # 池化窗口大小
        pool_stride=2,
        # 池化滑动步长
        param_attr=name + 'w',
        bias_attr=name + 'b',
        use_cudnn=use_cudnn,
        act=act
    )


###全连接层
def fc(x, num_filters, name=None, act=None):
    return fluid.layers.fc(
        input=x,
        size=num_filters,
        act=act,
        param_attr=name + 'w',
        bias_attr=name + 'b'
    )


###转置卷积层
def deconv(x, num_filters, name=None, filter_size=5, stride=2, dilation=1, padding=2, output_size=None, act=None):
    return fluid.layers.conv2d_transpose(
        input=x,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        num_filters=num_filters,
        # 滤波器数量
        output_size=output_size,
        # 输出图片大小
        filter_size=filter_size,
        # 滤波器大小
        stride=stride,
        # 步长
        dilation=dilation,
        # 膨胀比例大小
        padding=padding,
        use_cudnn=use_cudnn,
        # 是否使用cudnn内核
        act=act
        # 激活函数
    )

def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act=None,
                  groups=64,
                  name=None):
    tmp = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=name + '_conv_b',
        param_attr=name + '_conv_w',
    )
    return fluid.layers.batch_norm(
        input=tmp,
        act=act,
        param_attr=name + '_bn_1',
        # 指定权重参数属性的对象
        bias_attr=name + '_bn_2',
        # 指定偏置的属性的对象
        moving_mean_name=name + '_bn_3',
        # moving_mean的名称
        moving_variance_name=name + '_bn_4',
        # moving_variance的名称
        name=name + '_bn_',
        momentum=0.5,
    )

def shortcut(input, ch_in, ch_out, stride,name):
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None,name=name)
    else:
        return input


def basicblock(input, ch_in, ch_out, stride,name,act):
    tmp = conv_bn_layer(input, ch_out, 3, stride, 1, name=name + '_1_',act=act)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, name=name + '_2_')
    short = shortcut(input, ch_in, ch_out, stride,name=name)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')


def layer_warp(block_func, input, ch_in, ch_out, count, stride,name,act='relu'):
    tmp = block_func(input, ch_in, ch_out, stride,name=name + '1',act=act)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1,name=name + str(i + 1),act=act)
    return tmp

###判别器
def D(x):
    # (96 + 2 * 1 - 4) / 2 + 1 = 48
    x = conv_bn_layer(x, 64, 4, 2, 1, act=None, name='conv_bn_1')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='leaky_relu_1')
    x = fluid.layers.dropout(x,0.4,name='dropout1')
    # (48 + 2 * 1 - 4) / 2 + 1 = 24
    x = conv_bn_layer(x, 128, 4, 2, 1, act=None, name='conv_bn_2')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='leaky_relu_2')
    x = fluid.layers.dropout(x,0.4,name='dropout2')
    # (24 + 2 * 1 - 4) / 2 + 1 = 12
    x = conv_bn_layer(x, 256, 4, 2, 1, act=None, name='conv_bn_3')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='leaky_relu_3')
    x = fluid.layers.dropout(x,0.4,name='dropout3')
    # (12 + 2 * 1 - 4) / 2 + 1 = 6
    x = conv_bn_layer(x, 512, 4, 2, 1, act=None, name='conv_bn_4')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='leaky_relu_4')
    x = fluid.layers.dropout(x,0.4,name='dropout4')
    x = fluid.layers.reshape(x,shape=[-1, 512 * 6 * 6])
    x = fc(x, 2, name='fc1')
    return x


###生成器
def G(x):
    # x = fc(x,6 * 6 * 2,name='g_fc1',act='relu')
    # x = bn(x, name='g_bn_1', act='relu',momentum=0.5)
    x = fluid.layers.reshape(x, shape=[-1, 2, 6, 6])
    x = layer_warp(basicblock, x, 2, 256, 1, 1, name='g_res1', act='relu')

    # 2 * (6 - 1) - 2 * 1  + 4 = 12
    x = deconv(x, num_filters=256, filter_size=4, stride=2, padding=1, name='g_deconv_1')
    x = bn(x, name='g_bn_2', act='relu', momentum=0.5)

    # 2 * (12 - 1) - 2 * 1  + 4 = 24
    x = deconv(x, num_filters=128, filter_size=4, stride=2, padding=1, name='g_deconv_2')
    x = bn(x, name='g_bn_3', act='relu', momentum=0.5)

    # 2 * (24 - 1) - 2 * 1  + 4 = 48
    x = deconv(x, num_filters=64, filter_size=4, stride=2, padding=1, name='g_deconv_3')
    x = bn(x, name='g_bn_4', act='relu', momentum=0.5)

    # 2 * (48 - 1) - 2 * 1  + 4 = 96
    x = deconv(x, num_filters=3, filter_size=4, stride=2, padding=1, name='g_deconv_4', act='relu')

    return x

###损失函数
def loss(x, label):
    return fluid.layers.mean(fluid.layers.softmax_with_cross_entropy(logits=x, label=label))

