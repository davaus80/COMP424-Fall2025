# Student agent: Add your own agent here
# python simulator.py --player_1 greedy_corners_agent --player_2 student_agent --display

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

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

    # statistics
    avg_calc_time = 0
    n = 0

    
    valid_moves = get_valid_moves(chess_board, player)
    bestMove = None
    bestValue = -10000

    # MINIMAX ALGORITHM 
    for move in valid_moves:
      sim_board = deepcopy(chess_board)
      execute_move(sim_board, move, player)

      value = self.minimax(sim_board, 2, False, player, opponent)

      if (value > bestValue):
        bestValue = value
        bestMove = move

    
      
    if (bestMove != None):
      print("\t\tfrom: (", bestMove.row_src ,", ", bestMove.col_src,") --> to: (", bestMove.row_dest ,", ", bestMove.col_dest,")")


    time_taken = time.time() - start_time
    print("\t\tMy AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return bestMove or random_move
  
  def calculate_heuristic(self, board, player, opponent):
    plyr_count = np.count_nonzero(board == player)
    opp_count = np.count_nonzero(board == opponent)

    # print("\t\t h: ", plyr_count - opp_count, " - ", plyr_count," - ",  opp_count)
    return plyr_count - opp_count
  
  def minimax(self, board_state, depth, MAX, plyr, opp):
    if depth == 0 or check_endgame(board_state) == True:
      return self.calculate_heuristic(board_state, plyr, opp) 

    if MAX:
      value = -10000
      legal_moves = get_valid_moves(board_state, plyr)
      for move in legal_moves:
        sim_board = deepcopy(board_state)
        execute_move(sim_board, move, plyr)
        value = max(value, self.minimax(sim_board, depth-1, False, plyr, opp))
        # print(" ---+++---max: ",value)
    else:
      value = 10000
      legal_moves = get_valid_moves(board_state, opp)
      for move in legal_moves:
        sim_board = deepcopy(board_state)
        execute_move(sim_board, move, opp)
        value = min(value, self.minimax(sim_board, depth-1, True, plyr, opp))
        # print(" ---+++---min: ",value)

    return value


