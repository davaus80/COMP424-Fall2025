# Student agent: Add your own agent here
# python simulator.py --player_1 greedy_corners_agent --player_2 student_agent --display

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, get_directions, MoveCoordinates, count_disc_count_change

@register_agent("bootyCall_agent")
class BootyCallAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(BootyCallAgent, self).__init__()
    self.name = "BootyCall"

  def step(self, chess_board, player, opponent):

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

    
    # MINIMAX ALGORITHM
    d=1
    init_player = np.count_nonzero(chess_board == player)
    init_opp = np.count_nonzero(chess_board == opponent)

    while time.time() - start_time < time_limit:
      new_order = []
      i = 0
      for move in valid_moves:
        if (isinstance(move, MoveCoordinates) == False):
          move = move[0]
        sim_board = deepcopy(chess_board)
        execute_move(sim_board, move, player)

        # update player count
        gain_player = count_disc_count_change(chess_board, move, player)
        initial_count_player = init_player + gain_player

        # update opp count
        if self.jump_move(move):
          initial_count_opp = init_opp - gain_player
        else:
          initial_count_opp = init_opp - gain_player + 1
        

        value = self.minimax(sim_board, d, -10000, 10000, False, player, opponent, start_time, time_limit, initial_count_player, initial_count_opp)
        
        new_order.append((move, value))

        if (value > bestValue):
          print("\t\t --- ", i,"/",len(valid_moves))
          bestValue = value
          bestMove = move
        i +=1

      new_order.sort(key=lambda sorted_move: sorted_move[1],reverse=True)
      valid_moves = new_order

      print(" ---> d: ",d)
      d += 1
      

    
      
    if (bestMove != None):
      print("\t\tfrom: (", bestMove.row_src ,", ", bestMove.col_src,") --> to: (", bestMove.row_dest ,", ", bestMove.col_dest,")")


    time_taken = time.time() - start_time
    print("\t\tMy AI's turn took ", time_taken, "seconds.")

    # Returning a random move if no valid move
    return bestMove or random_move
  
  
 
  
  def calculate_heuristic(self, board, player, opponent, player_c, opp_c):
   
    # greedy corners agent heuristic

    # piece difference
    # player_count = np.count_nonzero(board == player)
    # opp_count = np.count_nonzero(board == opponent)
    # score_diff = player_count - opp_count
 

    score_diff = player_c - opp_c

    # corner control bonus
    # n = board.shape[0]
    # corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    # corner_bonus = sum(1 for (i, j) in corners if board[i, j] == player)
    # penalize opponent mobility
    # opp_moves = len(get_valid_moves(board, opponent))
    # mobility_penalty = -opp_moves
    return score_diff #+ corner_bonus #+ mobility_penalty
  
    
  def minimax(self, board_state, depth, alpha, beta, MAX, plyr, opp, start_time, time_limit, player_count, opp_count):
    if depth == 0 or check_endgame(board_state) == True or time.time() - start_time > time_limit:
      return self.calculate_heuristic(board_state, plyr, opp, player_count, opp_count) 
    

    if MAX:
      value = -10000
      legal_moves = get_valid_moves(board_state, plyr)
      for move in legal_moves:
        sim_board = deepcopy(board_state)
        execute_move(sim_board, move, plyr)

        penalty = 0

        # update player count
        gain_player = count_disc_count_change(board_state, move, plyr)
        new_count_player = player_count + gain_player

        # update opp count
        if self.jump_move(move):
          new_count_opp = opp_count - gain_player
          penalty = -3
        else:
          new_count_opp = opp_count - gain_player + 1

        child_value = self.minimax(sim_board, depth-1, alpha, beta, False, plyr, opp, start_time, time_limit, new_count_player, new_count_opp)
        
        child_value += penalty

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

        penalty = 0

        gain_opp = count_disc_count_change(board_state, move, opp)
        
        # update player count
        if self.jump_move(move):
          new_count_player = player_count - gain_opp
          penalty = 3
        else:
          new_count_player = player_count - gain_opp + 1

        # update opp count
        new_count_opp = opp_count + gain_opp

        child_value = self.minimax(sim_board, depth-1, alpha, beta, True, plyr, opp, start_time, time_limit, new_count_player, new_count_opp)

        child_value += penalty


        value = min(value, child_value)
        beta = min(value, beta)
        if (alpha >= beta):
          # print("\tMin: pruned")
          break

    return value


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