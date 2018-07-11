import pickle
from env.board import Board
from env.game import Game

from agent.mcts_alphaZero import MCTSPlayer
from agent.policy_value_net_tensorflow import PolicyValueNet


class Human:
    """
    通过控制台输入当前坐标与agent进行游戏
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move (y,x): ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)  # 将location转化为一维的move
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            # 判断输入的location是否合理
            print("invalid move, input (y,x) again")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run(model_file):
    n = 5
    width, height = 8, 8
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNet(width, height)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance
        mcts_player2 = MCTSPlayer(best_policy.policy_value_fn,
                                  c_puct=5,
                                  n_playout=400)

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(mcts_player2, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
