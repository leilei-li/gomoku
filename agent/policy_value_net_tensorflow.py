"""
采用tensorflow实现 policy value net
"""

import numpy as np
import tensorflow as tf


class PolicyValueNet:
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height

        # 定义输入层
        self.input_states = tf.placeholder(tf.float32, shape=[None, 4, self.board_width, self.board_height])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])  # 对矩阵进行转置

        # 定义神经网络的层，池化层不用？
        ## 第一个卷积层
        self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32,  # 卷积核的数目，即输出的维度
                                      kernel_size=[3, 3],
                                      padding='same',
                                      activation=tf.nn.relu)  # 输出是self.conv1
        ## 第二个卷积层
        self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                      filters=64,
                                      kernel_size=[3, 3],
                                      padding='same',
                                      activation=tf.nn.relu)
        ## 第三个卷积层
        self.conv3 = tf.layers.conv2d(inputs=self.conv2,
                                      filters=128,
                                      kernel_size=[3, 3],
                                      padding='same',
                                      activation=tf.nn.relu)

        # 动作卷积网络
        self.action_conv = tf.layers.conv2d(inputs=self.conv3,
                                            filters=4,
                                            kernel_size=[1, 1],
                                            padding='same',
                                            activation=tf.nn.relu)
        # 将Tensor转化为一维的
        self.action_conv_flat = tf.reshape(self.action_conv, [-1, 4 * self.board_width * self.board_height])

        # 全联接层(fully connected layer)，输出动作move的概率的对数形式
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=self.board_height * self.board_width,
                                         activation=tf.nn.log_softmax)
        # 评估网络
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3,
                                                filters=2,
                                                kernel_size=[1, 1],
                                                padding='same',
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(self.evaluation_conv, [-1, 2 * self.board_width * self.board_height])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=self.board_height * self.board_width,
                                              activation=tf.nn.relu)
        # 输出当前的状态的评估分，即全连接层只连接一个神经元,激活函数采用tanh是为了能够输出[-1.0,1.0]中的负值
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1,
                                              activation=tf.nn.tanh)

        # 定义损失函数 Loss Function
