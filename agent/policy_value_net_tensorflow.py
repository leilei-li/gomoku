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
        self.input_state=
