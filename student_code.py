from dataclasses import dataclass
import numpy as np
import typing as t

from gym_connect_four.connect_four_env import ConnectFourEnv

# -- Constants -- #
STUDENT = 1
SERVER  = -1
EMPTY   = 0
MINIMIZE = -1
MAXIMIZE = 1
MAX_DEPTH = 4
MAX = 1000000
MIN = -1*MAX

# -- Classes -- #
class HeuristicWeights:
    TWOS   = 1
    THREES = 5
    FOURS  = 100

@dataclass
class Player:
    """ Data struct to hold values in the heuristic function. """
    horizontal_twos: int
    horizontal_threes: int
    vertical_twos: int
    vertical_threes: int
    diagonal_twos: int
    diagonal_threes: int
    horizontal_fours: int
    vertical_fours: int
    diagonal_fours: int


class Game:
    """ Wrapper class with some helper-methods. """

    def __init__(self, env: ConnectFourEnv) -> None:
        self.env = ConnectFourEnv(env=env)

    def actions(self, state: np.ndarray):
        env_board = self.env.board
        self.env._board = state
        actions = self.env.available_moves()
        self.env._board = env_board

        return actions

    def result(self, state: np.ndarray, action: int, maximize: bool) -> np.ndarray:
        env_player = self.env.current_player
        env_board = self.env.board

        self.env.current_player = MAXIMIZE if maximize else MINIMIZE
        self.env._board = state.copy()
        state, _, done, _ = self.env.step(action)

        self.env._board = env_board
        self.env.current_player = env_player
        return state, done


# -- Functions -- #

def eval_player(board: np.ndarray, row: int, col: int, player_turn: int, player: Player) -> None:
    """
        Evaluation/Heuristic function.
        params:
            board: Input state.
            row: Given row to search from.
            col: Given column to search from.
            player_turn: Integer representing maximize or minimize player turn.
            player: Player object that holds "scores" for marks in a row.
    """

    # This is not the prettiest code, but hopefully not too hard to understand what's going on :)

    # Check vertical
    twos = (row-1 >= 0) and (board[row-1, col] == player_turn) and (board[row, col] == player_turn)
    threes = twos and (row-2 >= 0) and (board[row-2, col] == player_turn)
    fours = threes and (row-3 >= 0) and (board[row-3, col] == player_turn)

    if fours:
        player.vertical_fours += 1
        player.vertical_threes -= 1
    elif threes:
        player.vertical_threes += 1
        player.vertical_twos -= 1
    elif twos:
        player.vertical_twos += 1

    # Check horizontal
    twos = (col-1 >= 0) and (board[row, col-1] == player_turn) and (board[row, col] == player_turn)
    threes = twos and (col-2 >= 0) and (board[row, col-2] == player_turn)
    fours = threes and (col-3 >= 0) and (board[row, col-3] == player_turn)
    if fours:
        player.horizontal_fours += 1
        player.horizontal_threes -= 1
    elif threes:
        player.horizontal_threes += 1
        player.horizontal_twos -= 1
    elif twos:
        player.horizontal_twos += 1

    # Check diagonal
    twos = (row-1 >= 0) and (col-1 >= 0) and (board[row-1, col-1] == player_turn) and (board[row, col] == player_turn)
    threes = twos and (row-2 >= 0) and (col-2 >= 0) and (board[row-2, col-2] == player_turn)
    fours = threes and (row-3 >= 0) and (row-3 >= 0) and (board[row-3, col-3] == player_turn)
    if fours:
        player.diagonal_fours += 1
        player.diagonal_threes -= 1
    elif threes:
        player.diagonal_threes += 1
        player.diagonal_twos -= 1
    elif twos:
        player.diagonal_twos += 1


def get_score(player: Player) -> int:
    """
        Calculates the scores for the player using the heuristic weights. 
        params:
            player: Player object that holds "scores" for marks in a row.
        returns:
            Score of the state.
    """
    return player.horizontal_twos * HeuristicWeights.TWOS + \
           player.vertical_twos * HeuristicWeights.TWOS + \
           player.diagonal_twos * HeuristicWeights.TWOS + \
           player.horizontal_threes * HeuristicWeights.THREES + \
           player.vertical_threes * HeuristicWeights.THREES + \
           player.diagonal_threes * HeuristicWeights.THREES + \
           player.horizontal_fours * HeuristicWeights.FOURS + \
           player.vertical_fours * HeuristicWeights.FOURS + \
           player.diagonal_fours * HeuristicWeights.FOURS


def heuristic_function(state: np.ndarray) -> int:
    """
        Performs the heuristic evaluation of the given state.
        params:
            state: Input state.
        returns:
            Heuristic value of the state
    """
    student = Player(0, 0, 0, 0, 0, 0, 0, 0, 0)
    server = Player(0, 0, 0, 0, 0, 0, 0, 0, 0)
    rows, cols = state.shape

    for row in range(rows):
        for col in range(cols):
            eval_player(state, row, col, STUDENT, student)
            eval_player(state, row, col, SERVER, server)

    return get_score(student) - get_score(server)



def alpha_beta(depth: int, game: Game, state: np.ndarray, is_done: bool,
               maximize: bool, alpha: int, beta: int) -> t.Tuple[int, int]:
    """
        Performs the alpha-beta pruning search algorithm.
        params:
            depth: Maximum depth of search traversal.
            game: A game object, which wraps a game environment.
            state: State of the game.
            is_done: Boolean indicating whether the game is over or not.
            maximize: Boolean indicating whether the algorithm should maximize or minimize.
            alpha: Value of alpha parameter.
            beta: Value of beta parameter.
        returns:
            A tuple consisting of (heuristic value of state, best action)
    """
    # Check if we've reached depth or end of game.
    if depth == 0 or is_done:
        return heuristic_function(state), None

    # Get available actions for this state. If there's no actions available, return heuristic value of state.
    actions = game.actions(state)
    if not actions:
        return heuristic_function(state), None

    best_action = None

    # Perform minmax search, updating the best-value and best-action as well as
    # alpha and beta if we've found new better values.
    if maximize:
        # Maximizing: Set default value to minimum
        best_value = MIN

        for action in actions:
            new_value, _ = alpha_beta(depth-1, game, *game.result(state, action, maximize=True), False, alpha, beta)

            if new_value > best_value:
                best_value, best_action = new_value, action
                alpha = max(alpha, best_value)

            # If our best value beats beta, we prune the rest of the children.
            if best_value >= beta:
                return best_value, best_action

    else:
        # Minimizing: Set default value to maximum
        best_value = MAX

        for action in actions:
            new_value, _ = alpha_beta(depth-1, game, *game.result(state, action, maximize=False), True, alpha, beta)
            if new_value < best_value:
                best_value, best_action = new_value, action
                beta = min(beta, best_value)

            # If our best value beats alpha, we prune the rest of the children.
            if best_value <= alpha:
                return best_value, best_action


    return best_value, best_action


def student_move(env: ConnectFourEnv, state: np.ndarray = None, depth: int = MAX_DEPTH) -> int:
    """
        Does a move for the student.
        params:
            env: Environment of the game.
            state: Current state.
            depth: Maximum depth to search
    """
    # Wrap the environment as a game object, for easier use of step predictions.
    env._board = state
    game = Game(env)
    start_state = env.board if state is None else state

    # Perform search algorithm.
    _, action = alpha_beta(depth, game, start_state, False, True, MIN, MAX)

    return action


if __name__ == '__main__':
    env = ConnectFourEnv()
    state = env.reset()
    env.step(0)
    state, *_ = env.step(0)
    action = student_move(env, state)
    print(f'Action: {action}')
    print(state)
    print(env.available_moves())
