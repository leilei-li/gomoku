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
        # 第一个卷积层
        self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32,  # 卷积核的数目，即输出的维度
                                      kernel_size=[3, 3],
                                      padding='same',
                                      activation=tf.nn.relu)  # 输出是self.conv1
        # 第二个卷积层
        self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                      filters=64,
                                      kernel_size=[3, 3],
                                      padding='same',
                                      activation=tf.nn.relu)
        # 第三个卷积层
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
        # 输出当前的状态的评估分，评估分指的是黑棋和白棋谁能够赢得比赛，越接近1那么黑棋越容易赢得比赛，
        # 即全连接层只连接一个神经元,激活函数采用tanh是为了能够输出[-1.0,1.0]中的负值
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1,
                                              activation=tf.nn.tanh)

        # 定义损失函数 Loss Function
        # label即每局比赛是哪个玩家赢得比赛，取值只有-1.0和1.0
        # prediction是就是当前状态的评估,即evaluation_fc2是[-1.0,1.0]之间的取值，
        # 我们可以用label和prediction来进行均方误差来定义损失函数了
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])

        # value损失函数（value loss function），label和evaluation_fc2进行均方误差
        self.value_loss = tf.losses.mean_squared_error(self.labels, self.evaluation_fc2)

        # policy损失函数(policy value function), 取mcts_prob和action_fc的交叉熵
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None, self.board_width * self.board_height])
        self.policy_loss = tf.losses.softmax_cross_entropy(self.mcts_probs, self.action_fc)

        # L2正则化惩罚项,L2 regularization penalty
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        self.l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])

        # 那么总的loss可以定义出来了
        self.loss = self.value_loss + self.policy_loss + self.l2_penalty

        # 接下来定义优化器，采用Adam最小化loss
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # 初始化Tensorflow
        self.session = tf.Session()

        # calc policy entropy, for monitoring only???
        self.entropy = tf.losses.softmax_cross_entropy(tf.exp(self.action_fc), self.action_fc)

        # 初始化session
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
            print('restore model_file:', model_file)

    def policy_value(self, state_batch):
        """
         计算动作概率和当前的value
         输入当前的状态
         输出动作的softmax概率和value
        :return:
        """
        log_act_probs, value = self.session.run([self.action_fc, self.evaluation_fc2],
                                                feed_dict={self.input_states: state_batch})
        act_probs = np.exp(log_act_probs)  # 前面算全连接用的是tf.nn.log_softmax激活函数
        return act_probs, value

    def policy_value_fn(self, board):
        """
        输入当前的board
        输出 action-probability 对及当前board状态的value
        :param board:
        :return:
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, learning_rate):
        winner_batch = np.reshape(winner_batch, (-1, 1))  # 转化成列向量
        loss, entropy, _ = self.session.run([self.loss, self.entropy, self.optimizer],
                                            feed_dict={
                                                self.input_states: state_batch,
                                                self.mcts_probs: mcts_probs,
                                                self.labels: winner_batch,
                                                self.learning_rate: learning_rate
                                            })
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
