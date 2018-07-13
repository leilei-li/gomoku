import random
import numpy as np
from collections import defaultdict, deque
from env.board import Board
from env.game import Game
from agent.mcts_pure import MCTSPlayer as MCTS_Pure
from player.MCTS_player import MCTSPlayer
from agent.policy_value_net_tensorflow import PolicyValueNet


class TrainPipeline:
    """
    通过策略价值网络训练学习最优解
    """

    def __init__(self, board_width=8, board_height=8, n_in_row=5, init_modle=None):
        # 初始化棋盘和游戏服务器
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.loss_to_show = -1
        self.entropy_to_show = -1

        # 初始化训练所用的参数
        self.learning_rate = 2e-3
        self.lr_multiplier = 1.0  # 根据KL散度自动调整学习速率
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # 每次取batch_size进行梯度下降
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50  # 一个间隔，取模为0时存储模型到硬盘
        self.game_batch_num = 1500  # 最多进行1500局游戏
        self.best_win_ratio = 0.0

        # agent的对手，纯粹的mcts算法产生的棋手
        self.pure_mcts_playout_num = 1000

        # 是否加载原先已经存在的训练数据
        if init_modle:
            self.policy_value_net = PolicyValueNet(board_width=self.board_width, board_height=self.board_height,
                                                   model_file=init_modle)
        else:
            self.policy_value_net = PolicyValueNet(board_width=self.board_width, board_height=self.board_height)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):
        """
            由于棋盘是上下左右对称的，所以我们可以通过翻转和旋转来获得更多的数据集
            play_data: [(state, mcts_prob, winner_z), ..., ...]
        :param play_data:
        :return:
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """
            收集selfplay的数据用来训练
        :param n_games:
        :return:
        """
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            # 拓展数据集
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)  # 存入双向队列

    def policy_update(self):
        """
            更新策略函数
        :return:
        """
        try:
            mini_batch = random.sample(self.data_buffer, self.batch_size)
        except:
            mini_batch = random.sample(list(self.data_buffer), self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch=state_batch, mcts_probs=mcts_probs_batch,
                                                             winner_batch=winner_batch,
                                                             learning_rate=self.learning_rate * self.lr_multiplier)
            self.loss_to_show = loss
            self.entropy_to_show = entropy
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # explained_var_old = (1 -
        #                      np.var(np.array(winner_batch) - old_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        # explained_var_new = (1 -
        #                      np.var(np.array(winner_batch) - new_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        # print(("kl:{:.5f},"
        #        "lr_multiplier:{:.3f},"
        #        "loss:{},"
        #        "entropy:{},"
        #        "explained_var_old:{:.3f},"
        #        "explained_var_new:{:.3f}"
        #        ).format(kl,
        #                 self.lr_multiplier,
        #                 loss,
        #                 entropy,
        #                 explained_var_old,
        #                 explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
            与单纯的MCTS_Pure进行对抗训练，来监控当前策略的好坏
        :param n_games:
        :return:
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0)
            win_cnt[winner] = win_cnt[winner] + 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """
            开始训练
        :return:
        """
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(("batch i:{},\t"
                       "episode_len:{},\t"
                       "loss:{:.8f},\t"
                       "entropy:{:.8f},"
                       ).format(i + 1,
                                self.episode_len,
                                self.loss_to_show,
                                self.entropy_to_show))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('model/current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('model/best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline(8, 8, 5)
    training_pipeline.run()
