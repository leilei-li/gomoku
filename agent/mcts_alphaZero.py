"""
MCTS算法，采用策略价值网络(policy-value network)进行树搜索和评估叶子结点
"""

import numpy as np
import copy


def softmax(x):
    """
    根据公式'S_i=\frac{e_i}{\sum_je_j}'
    计算相应的softmax值
    :param x:
    :return:
    """
    probs = np.exp(x - np.max(x))  # 计算分子,利用softmax(x)=softmax(x+C)来保证不溢出
    probs = probs / np.sum(probs)
    return probs


class TreeNode:
    """
    蒙特卡洛树的树结点的类,
    每个结点用自己的Q值，
    先验概率P，
    访问计数值(visit-count-adjusted prior score):U
    U值的计算公式：U(s,a)=c_{puct}P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}
    详情可以参考Alphazero的论文
    """

    def __init__(self, parent, prior_p):
        self._parent = parent  # 父结点
        self._children = {}  # 子结点，代表动作，即落子动作
        self._n_visits = 0  # 结点访问次数
        self._Q = 0  # Q值
        self._u = 0  # U值
        self._P = prior_p  # 先验概率

    def expand(self, action_priors):
        """
        通过生成新的子结点来扩展树
        :param action_priors: 通过policy func获得的tuple（actions,prior_p）
        :return:
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)  # 父结点是self，先验概率prob

    def select(self, c_puct):
        """
        选择子结点，能够得到最大的 Q+u(P)
        :param c_puct:
        :return: (action,next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        更新当前叶子结点的值
        :param leaf_value:
        :return:
        """
        self._n_visits = self._n_visits + 1
        self._Q = self._Q + 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        更新叶子结点后，我们还需要递归更新当前叶结点的所有祖先
        :param leaf_value:
        :return:
        """
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        计算当前结点的value
        :param
            c_puct: a number in (0, inf) controlling the relative impact of
                value Q, and prior probability P, on this node's score.
        :return:
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._u + self._Q

    def is_leaf(self):
        """
        判断当前结点是否是叶子结点
        :return:
        """
        return self._children == {}

    def is_root(self):
        """
        判断当前结点是否是根结点
        :return:
        """
        return self._parent is None


class MCTS:
    """
    Monte Carlo Tree Search Algorithm
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """

        :param policy_value_fn:
            当前采用的策略函数
            输入当前棋盘的状态，
            输出(action,prob)元组，和score[-1,1]
        :param c_puct:
            控制MCTS exploration and exploitation 的关系，值越大表示越依赖之前的先验概率
        :param n_playout:
            MCTS算法的执行次数，次数越大效果越好，但是耗费的时间也会越多.

        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        根据当前的状态进行play
        :param state:
        :return:
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # 贪心算法选择下一个move
            action, node = node.select(self._c_puct)
            state.do_move(action)
        # 重新评估叶子结点
        action_probs, leaf_value = self._policy(state)
        # 判断游戏是否结束，如果游戏没有结束，那么扩展该叶子结点
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 游戏结束。如果平局，叶子结点的值为0，如果是叶子结点当前的玩家胜利，值为1，否则为-1
            if winner == -1:
                leaf_value = 0.0
            if winner == state.get_current_player():
                leaf_value = 1.0
            if winner != state.get_current_player():
                leaf_value = -1.0
        # 递归更新叶子结点的值和所有的祖先
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
            从当前状态开始获得所有可行动作及它们的概率，为了保证数据不出错，必须要采用深拷贝
        :param state:
            当前的游戏动作
        :param temp:
            类似epsilon值
        :return:
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        # 通过蒙特卡洛树计算所有动作的概率
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        执行一个动作move后，更新蒙特卡洛树的子树
        :return:
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
