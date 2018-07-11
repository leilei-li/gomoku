import numpy as np


class Board(object):
    """
    初始化棋盘类
    """

    def __init__(self, **kwargs):
        """
        参数都是采用dict来存储
        :param kwargs:
        """
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {}  # {key: move， value:player}
        self.n_in_row = int(kwargs.get('n_in_row', 5))  # 多少颗棋子连成一排能赢，五指棋默认是5颗
        self.players = [1, 2]  # 新建两个玩家

    def init_borad(self, start_player=0):
        """
        根据类的初始化值，初始化board
        :param start_player:
        :return:
        """
        if self.width < self.n_in_row or self.height < self.n_in_row:
            print('棋盘高度或者宽度不足，无法进行游戏！')
            return
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))  # 将所有位置初始化后用list存成一排
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        这里借用参考项目的描述:
        将一维的move转化成二维的location
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        :param move:
        :return:
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """
        与上个函数刚好相反，将二维的location转化成一维的move
        :param location:
        :return:
        """
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        返回当前棋盘的状态
        :return: state shape: 4*width*height，不太搞得懂为什么要4维，黑白或者没下？加上禁手？
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        """
        接收一个move，改变states、availables、current_player
        :param move:
        :return:
        """
        self.states[move] = self.current_player  # 记录当前move是哪个玩家进行的
        self.availables.remove(move)  # 棋盘上该点已经被占用，所以不再available
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )  # 下棋后切换选手
        self.last_move = move

    def has_a_winner(self):
        """
        判断棋盘上的棋子分布是否分出胜负
        :return: bool: 是否分出胜负, player: 胜利的玩家,-1代表平局？
        """
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """
        返回本局游戏结束信息
        :return: bool: 是否分出胜负, player: 胜利的玩家,-1代表平局？
        """
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player
