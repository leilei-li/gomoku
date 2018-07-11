class HumanPlayer:
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
