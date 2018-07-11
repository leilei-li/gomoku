from agent.mcts_alphaZero import *


class MCTSPlayer:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(acts,
                                        p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                                        )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print('棋盘已经满了')

    def __str__(self):
        return "MCTS {}".format(self.player)
