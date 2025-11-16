# Student agent: Add your own agent here
# python simulator.py --player_1 greedy_corners_agent --player_2 student_agent --display

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, get_directions, count_disc_count_change

@register_agent("alphaBeta_agent")
class AlphaBetaAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(AlphaBetaAgent, self).__init__()
    self.name = "AlphaBeta"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_limit = 1.95

    # statistics
    avg_calc_time = 0
    n = 0

    
    valid_moves = get_valid_moves(chess_board, player)
    bestMove = None
    bestValue = -10000

    breakOut = False
    # MINIMAX ALGORITHM
    for d in range(1,10):
      for move in valid_moves:
        sim_board = deepcopy(chess_board)
        execute_move(sim_board, move, player)

        value = self.minimax(sim_board, d, -10000, 10000, False, player, opponent, start_time, time_limit)
        

        if (value > bestValue):
          bestValue = value
          bestMove = move
      print(" ---> d: ",d)
      if time.time() - start_time > time_limit:
        break

    
      
    if (bestMove != None):
      print("\t\tfrom: (", bestMove.row_src ,", ", bestMove.col_src,") --> to: (", bestMove.row_dest ,", ", bestMove.col_dest,")")


    time_taken = time.time() - start_time
    print("\t\tMy AI's turn took ", time_taken, "seconds.")

    # Returning a random move if no valid move
    return bestMove or random_move
  
  
  def jump_move(self, move):
    return ((move.col_src-move.col_dest)*(move.col_src-move.col_dest)+(move.row_src-move.row_dest)*(move.row_src-move.row_dest)) >= 4
  
  def get_num_vulnerable_squares_from_jump(self, board, move, plyr):
    count = 0
    for dir in get_directions():
      adj_tile = (move.row_src+dir[0], move.col_src+dir[1])
      if (adj_tile[0] > 6 or adj_tile[1] > 6 or adj_tile[0] < 0 or adj_tile[1] < 0):
        continue

      if (board[adj_tile[0], adj_tile[1]] == plyr): 
        count += 1

    return count
  
  def calculate_heuristic(self, board, player, opponent):
   
    # greedy corners agent heuristic

    # piece difference
    player_count = np.count_nonzero(board == player)
    opp_count = np.count_nonzero(board == opponent)
    score_diff = player_count - opp_count
    # corner control bonus
    n = board.shape[0]
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    corner_bonus = sum(1 for (i, j) in corners if board[i, j] == player) * 5
    # penalize opponent mobility
    opp_moves = len(get_valid_moves(board, opponent))
    mobility_penalty = -opp_moves
    return score_diff + corner_bonus + mobility_penalty
  
    
  def minimax(self, board_state, depth, alpha, beta, MAX, plyr, opp, start_time, time_limit):
    if depth == 0 or check_endgame(board_state) == True or time.time() - start_time > time_limit:
      return self.calculate_heuristic(board_state, plyr, opp) 

    if MAX:
      value = -10000
      legal_moves = get_valid_moves(board_state, plyr)
      for move in legal_moves:
        sim_board = deepcopy(board_state)
        execute_move(sim_board, move, plyr)

        child_value = self.minimax(sim_board, depth-1, alpha, beta, False, plyr, opp, start_time, time_limit)

        value = max(value, child_value)
        alpha = max(value, alpha)
        if (alpha >= beta):
          # print("\tMax: pruned")
          break
    else:
      value = 10000
      legal_moves = get_valid_moves(board_state, opp)
      for move in legal_moves:
        sim_board = deepcopy(board_state)
        execute_move(sim_board, move, opp)
        child_value = self.minimax(sim_board, depth-1, alpha, beta, True, plyr, opp, start_time, time_limit)

        value = min(value, child_value)
        beta = min(value, beta)
        if (alpha >= beta):
          # print("\tMin: pruned")
          break

    return value


