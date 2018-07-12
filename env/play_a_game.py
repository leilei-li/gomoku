import pickle
from env.board import Board
from env.game import Game
from player.human_player import HumanPlayer
from player.MCTS_player import MCTSPlayer
from agent.policy_value_net_tensorflow import PolicyValueNet
from agent.policy_value_net_numpy import PolicyValueNetNumpy


def run(model_file, width=8, height=8, n=5):
    n = n
    width = width
    height = height
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
            try:
                policy_param = pickle.load(open(model_file, 'rb'),
                                           encoding='bytes')  # To support python3
            except:
                pass
        best_policy = PolicyValueNet(width, height, model_file=model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=2000)  # set larger n_playout for better performance
        mcts_player2 = MCTSPlayer(best_policy.policy_value_fn,
                                  c_puct=5,
                                  n_playout=400)

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = HumanPlayer()

        # set start_player=0 for human first
        game.start_play(human, mcts_player2, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
