import numpy as np

from agents.agent import Agent
from helpers import random_move, get_valid_moves, execute_move
from store import register_agent
import copy
import random


# Important: you should register your agent with a name
@register_agent("steve_agent")
class SteveAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(SteveAgent, self).__init__()
        self.name = "SteveAgent"
        self.autoplay = True

    def step(self, board, color, opponent):
        """
        Choose a move based on a simple Ataxx heuristic.
        """
        legal_moves = get_valid_moves(board, color)
        
        if not legal_moves:
            return None  # No valid moves available, pass turn
        
        best_move = None
        best_score = float("-inf")
        depth = getattr(self, "search_depth", 2)

        alpha = float("-inf")
        beta = float("inf")
        
        # Evaluate each legal move
        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            # Search opponent's moves after our move
            move_score = self.minimax(simulated_board, max(0, depth - 1), opponent, color, opponent)
            
            if move_score > best_score:
                best_score = move_score
                best_move = move
        
        return best_move or random.choice(legal_moves)

    # TODO:
    # 1. Alpha-beta pruning
    # 2. Monte Carlo Tree Search

    # Minimax implementation (Need alpha-beta pruning)
    def minimax(self, board_state, depth_remaining, current_player, agent_color, opponent_color, alpha=float("-inf"), beta=float("inf")):
        """
        Simple minimax without alpha-beta pruning.
        """
        legal = get_valid_moves(board_state, current_player)
        
        # Base case: terminal state or depth limit reached
        if depth_remaining <= 0 or not legal:
            return self.evaluate_board(board_state, agent_color, opponent_color)
        
        # Maximize for agent's turn, minimize for opponent's turn
        if current_player == agent_color:
            best_val = float("-inf")
            for move in legal:
                next_board = copy.deepcopy(board_state)
                execute_move(next_board, move, current_player)
                val = self.minimax(next_board, depth_remaining - 1, opponent_color, agent_color, opponent_color, alpha, beta)
                if val >= beta:
                    return val  # Beta cut-off -- value not considered by minimizer
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
            return best_val
        else:
            best_val = float("inf")
            for move in legal:
                next_board = copy.deepcopy(board_state)
                execute_move(next_board, move, current_player)
                val = self.minimax(next_board, depth_remaining - 1, agent_color, agent_color, opponent_color, alpha, beta)
                if val <= alpha:
                    return val  # Alpha cut-off -- value not considered by maximizer
                best_val = min(best_val, val)
                beta = min(beta, best_val)
            return best_val
        
    
    # Evaluation function -- can be tweaked
    def evaluate_board(self, board, color, opponent):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
        - player_score: Score of the current player.
        - opponent_score: Score of the opponent.

        Returns:
        - int: The evaluated score of the board.
        """
        # piece difference
        player_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count
        # corner control bonus
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_bonus = (
            sum(1 for (i, j) in corners if board[i, j] == color) * 5
        )  # can tweak that to lower value
        # penalize opponent mobility
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_penalty = -opp_moves
        return score_diff + corner_bonus + mobility_penalty
