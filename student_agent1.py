# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves


EVAL_WEIGHTS = {
    # board[1,1] == 3
    "big_x": {
        "score_diff":   1,   # material still matters
        "gap_offender": 20,   # small reward for blobs
        "gap_defender": 30,   # mild penalty for bad jumps leaving holes
        "move_bonus":   30,   # jumps slightly punished (your move_bonus is 0 or -5)
        #"board_bonus":  0.0,   # no special positional mask
        "mobility": 50,   # open so mobility important
    },

    # board[2,3] == 3 and board[1,3] != 3
    "plus1": {
        "score_diff":   1.0,
        "gap_offender": 10,   # blobs around cross arms are nice
        "gap_defender": 20,   # avoid leaving big 1-holes near the plus
        "move_bonus":   30,   # jumps are somewhat risky
        #"board_bonus":  0.8,   # if you later add a plus1 bonus grid
        "mobility": 40,   # open so mobility important
    },

    # board[1,3] == 3
    "plus2": {
        "score_diff":   1.0,
        "gap_offender": 10,   # like plus1 but a bit stronger clustering
        "gap_defender": 30,
        "move_bonus":   20,
        #"board_bonus":  1.0,
        "mobility": 20,   # open so mobility important
    },

    # board[2,2] == 3
    "point4": {
        "score_diff":   1,
        "gap_offender": 10,   # islands-ish: safe blobs good
        "gap_defender": 30,   # really punish dangerous gaps
        "move_bonus":   20,
        #"board_bonus":  1.2,   # central-ish control good
        "mobility": 40,   # open so mobility important
    },

    # board[1,2] == 3
    "circle": {
        "score_diff":   1,
        "gap_offender": 10,   # ring control
        "gap_defender": 20,
        "move_bonus":   30,
        #"board_bonus":  1.0,   # if you have a ring bonus mask later
        "mobility": 20,   # open so mobility important
    },

    # board[0,3] == 3
    "wall": {
        "score_diff":   1,
        "gap_offender": 30,   # big safe blobs behind wall are great
        "gap_defender": 10,   # strongly avoid 1-hole disasters
        "move_bonus":   20,   # jumps more dangerous here
        #"board_bonus":  2.0,   # chokepoints near the wall are super valuable
        "mobility": 0,   # not open so mobility does not matter
    },

    # board[0,2] == 3
    "watch_sides": {
        "score_diff":   1,
        "gap_offender": 10,   # side clusters good
        "gap_defender": 20,
        "move_bonus":   10,
        #"board_bonus":  1.5,   # strongly prefer good side squares
        "mobility": 50,   # open so mobility important
    },

    # else: empty / no obstacles
    "empty": {
        "score_diff":   1,
        "gap_offender": 10,   # moderate value for clustering
        "gap_defender": 20,   # avoid terrible gaps but not overkill
        "move_bonus":   10,   # your original: jumps punished by full move_bonus
        #"board_bonus":  0.5,   # mild center-ish preference if you use it
        "mobility": 50,   # open so mobility important
    },
}

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
  boards = []
  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, you, opp):
    """
      You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    start_time = time.time()
    
    legal_moves = get_valid_moves(chess_board, you)
    
    if not legal_moves:
      return None  # No valid moves available, pass turn
    
    move_set = self.split_moves(legal_moves)

    a = float('-inf')
    b =  float('inf')
    best_score = float('-inf')
    best_move = None
    i = 3
    n = chess_board.shape[0] 
    at_row, at_col = 0, 0
    dest_row, dest_col = 0, 0
    self.boards = []
    for moves in move_set:
      for move in moves:
        
        pieces = np.count_nonzero(chess_board)
        enemy_pieces = np.count_nonzero(chess_board == opp)
        your_pieces = np.count_nonzero(chess_board == you)
        at_row, at_col = move.get_src()
        dest_row, dest_col = move.get_dest()
      
       
          
        sim_board = deepcopy(chess_board)

        execute_move(sim_board, move, you)
        
        
        val = self.minVal(sim_board, move, you, opp, a, b, i, start_time, move)

        if val > best_score:
          best_score = val
          best_move = move  
    print(best_score)
    # for move in legal_moves:
      
    #   pieces = np.count_nonzero(chess_board)
    #   enemy_pieces = np.count_nonzero(chess_board == opp)
    #   your_pieces = np.count_nonzero(chess_board == you)
    #   at_row, at_col = move.get_src()
    #   dest_row, dest_col = move.get_dest()
      
    #   if not(dest_row - at_row == 2 or dest_col - at_col == 2):
        
    #     sim_board = deepcopy(chess_board)

    #     execute_move(sim_board, move, you)
        
        
    #     val = self.minVal(sim_board, move, you, opp, a, b, i, start_time)

    #     if val > best_score:
    #       best_score = val
    #       best_move = move  
  
    # for move in legal_moves:
      
    #   pieces = np.count_nonzero(chess_board)
    #   enemy_pieces = np.count_nonzero(chess_board == opp)
    #   your_pieces = np.count_nonzero(chess_board == you)
    #   at_row, at_col = move.get_src()
    #   dest_row, dest_col = move.get_dest()
      
    #   if (dest_row - at_row == 2 or dest_col - at_col == 2):
        
    #     sim_board = deepcopy(chess_board)

    #     execute_move(sim_board, move, you)
        
        
    #     val2 = self.minVal(sim_board, move, you, opp, a, b, i, start_time)

    #     if val2 > best_score:
    #       best_score = val2
    #       best_move = move  
    
    
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")
    
    return best_move or random_move(chess_board, you)
    
    
  
  

  def maxVal(self, board, move, you, opp, a, b, i, start_time,og_move):
    time_taken = time.time() - start_time

    if i == 0 or time_taken > 1.8:
      # sim_board = deepcopy(board)
      # execute_move(sim_board, move, opp)
      return self.evalMove(board, move, you, opp, i, og_move)
    legal_moves = get_valid_moves(board, you)
    move_set = self.split_moves(legal_moves)

    for moves in move_set:
      for move in moves:
        
        sim_board = deepcopy(board)
      
        execute_move(sim_board, move, you)
        if self.board_checker(sim_board):
          continue
        else:
          self.boards.append(sim_board)
        a = max(a, self.minVal(sim_board, move, you, opp, a, b, i - 1, start_time,og_move))
        
        if a >= b:
          return b
       
    return a
    
 
  def minVal(self, board, move, you, opp, a, b, i, start_time,og_move):
    time_taken = time.time() - start_time

    if i == 0 or time_taken > 1.8:
      # sim_board = deepcopy(board)
      # execute_move(sim_board, move, you)
      return self.evalMove(board, move, you, opp, i, og_move)
    
    legal_moves = get_valid_moves(board, opp)
    move_set = self.split_moves(legal_moves)

    for moves in move_set:
      for move in moves:
        
        sim_board = deepcopy(board)
        execute_move(sim_board, move, opp)
        
        if self.board_checker(sim_board):
          continue
        else:
          self.boards.append(sim_board)
        b = min(b, self.maxVal(sim_board, move, you, opp, a, b, i - 1, start_time,og_move))
        
        if a >= b:
          return a
        
    return b
  
  def evalMove(self, board, move, you, opp, i, og_move):
    finishing_move = self.last_move_win(board, you, opp)
    player_count = np.count_nonzero(board == you)
    opp_count = np.count_nonzero(board == opp)
    score_diff = player_count - opp_count
    corner_bonus = 0
    n = board.shape[0]
    # board_bonus = self.find_bonus_board(board, move, you, opp)
    
    mobility_penalty = self.mobility(board, you, opp)
    gap_defender = self.gap_defender(board, og_move, you, opp) 
    gap_offender = self.gap_offender(board, og_move, you, opp)
    # gap_penalty = 0
    edge_bonus =  20 * 0 #* self.edge_bonus(board, you) 
    center_bonus =  2 * 0 #* self.center_bonus(board, you) 
    move_bonus =  self.move_bonus(move, you) 
    #safe_territory_bonus = 10 * self.far_from_opp(board, move, you, opp)

    btype = self.get_board_type(board)
    w = EVAL_WEIGHTS[btype]

    total_score = (w["score_diff"] * score_diff + 
                   w["gap_offender"] * gap_offender + 
                   w["gap_defender"] * gap_defender + 
                   0 * corner_bonus + 
                   w["mobility"] * mobility_penalty +
                   edge_bonus +  
                   finishing_move + 
                   center_bonus + 
                   w["move_bonus"] * move_bonus 
                   # 0 * board_bonus 
                   #safe_territory_bonus 
                   )
    
    return total_score
  
  def evalMove1(self, board, move, you, opp, i, og_move):
    finishing_move = self.last_move_win(board, you, opp)
    player_count = np.count_nonzero(board == you)
    opp_count = np.count_nonzero(board == opp)
    score_diff = player_count - opp_count
    return score_diff + finishing_move

  def last_move_win(self, board, you, opp):
    if check_endgame(board)[0]:
      you_pts = np.count_nonzero(board == you)
      opp_pts = np.count_nonzero(board == opp)
      return 33550336 * (you_pts - opp_pts)
    return 0

  def mobility(self, board, you, opp):
    opp_moves = len(get_valid_moves(board, opp))
    you_moves = len(get_valid_moves(board, you))
    return -opp_moves
  
  def gap_defender(self, board, move, you, opp):
    n = board.shape[0] 
    gap_penalty = 0
   
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()
    dir_coords = [
        (-1,-1), (-1, 0), (-1, 1),
        ( 0,-1),          ( 0, 1),
        ( 1,-1), ( 1, 0), ( 1, 1)
    ]
 
    if (dest_col - at_col) == 2 or (dest_row - at_row) == 2:
      if board[at_row][at_col] == 0:
            opp_close_by = 0

            for row_diff, col_diff in dir_coords:
              opp_pos = at_row + row_diff ,  at_col + col_diff
            
              if 0 <= opp_pos[0] < n and 0 <= opp_pos[1] < n:
                if board[opp_pos[0], opp_pos[1]] == you:
                  opp_close_by += 1
            if opp_close_by >= 5:
              gap_penalty -= opp_close_by
      # for row in range(n):
      #   for col in range(n):
      #     if board[row][col] == 0:
      #       opp_close_by = 0

      #       for row_diff, col_diff in dir_coords:
      #         opp_pos = row + row_diff ,  col + col_diff
            
      #         if 0 <= opp_pos[0] < n and 0 <= opp_pos[1] < n:
      #           if board[opp_pos[0], opp_pos[1]] == you:
      #             opp_close_by += 1
      #       if opp_close_by >= 5:
      #         gap_penalty -= opp_close_by
            

    return gap_penalty

  def gap_offender(self, board, move, you, opp):
    n = board.shape[0] 
    gap_penalty = 0
   
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()
    dir_coords = [
        (-1,-1), (-1, 0), (-1, 1),
        ( 0,-1),          ( 0, 1),
        ( 1,-1), ( 1, 0), ( 1, 1)
    ]
        
    for row in range(n):
      for col in range(n):
        if board[row][col] == you:
          opp_close_by = 0

          for row_diff, col_diff in dir_coords:
            opp_pos = row + row_diff ,  col + col_diff
           
            if 0 <= opp_pos[0] < n and 0 <= opp_pos[1] < n:
              if board[opp_pos[0], opp_pos[1]] == you:
                opp_close_by += 1
          
          gap_penalty += opp_close_by 

    return gap_penalty
    # if board[dest_row][dest_col] == 0:
    #   if n-2 > dest_row >= 1 and n-2 > dest_col >= 1 :
    #       if board[dest_row - 1][dest_col] == opp:
    #         count += 1
    #       if board[dest_row + 1][dest_col] == opp:
    #         count += 1
    #       if board[dest_row][dest_col - 1] == opp:
    #         count += 1
    #       if board[dest_row][dest_col - 1] == opp:
    #         count += 1

    #       if board[dest_row - 1][dest_col - 1] == opp:
    #         count += 1
    #       if board[dest_row + 1][dest_col - 1] == opp:
    #         count += 1
    #       if board[dest_row - 1][dest_col + 1] == opp:
    #         count += 1
    #       if board[dest_row + 1][dest_col + 1] == opp:
    #         count += 1
      
    return count
  
  def edge_bonus(self, board, you):
    n = board.shape[0] 
    edges = []
    edges.append((0,0))
    edges.append((0,1))
    edges.append((1,0))
    edges.append((1,1))
    edges.append((n-1,n-1))
    edges.append((n-2,n-1))
    edges.append((n-1,n-2))
    edges.append((n-2,n-2))
    edges.append((0,n-1))
    edges.append((0,n-2))
    edges.append((1,n-1))
    edges.append((1,n-2))
    edges.append((n-1,0))
    edges.append((n-2,0))
    edges.append((n-1,1))
    edges.append((n-2,1))

    edge_bonus = sum(1 for (i, j) in edges if board[i, j] == you)
    return edge_bonus

  def center_bonus(self, board, you):
    return 0
    n = board.shape[0] 
    center = [
      (n//2 - 1, n//2 -1), (n//2 - 1, n//2), (n//2 - 1, n//2 + 1), 
      (n//2, n//2 - 1),        (n//2, n//2), (n//2,n//2 + 1),
      (n//2 + 1, n//2 -1), (n//2 + 1, n//2), (n//2 + 1, n//2 + 1)
    ]
    center_bonus = sum(1 for (i, j) in center if board[i, j] == you)
    return center_bonus
  
  def move_bonus(self, move, you):
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()

    if (dest_col - at_col) == 2 or (dest_row - at_row) == 2:
      return -5
    else:
      return 0

  def split_moves(self, moves):
    
    duplicate_moves = []
    jump_moves = []

    for move in moves:
      at_row, at_col = move.get_src()
      dest_row, dest_col = move.get_dest()

      if dest_row - at_row == 2 or dest_col - at_col == 2:
        jump_moves.append(move)
      else:
        duplicate_moves.append(move)

    move_set = [duplicate_moves, jump_moves]
    return move_set



  def board_checker(self, board):
    n = board.shape[0]
    
    for old_board in self.boards:
      return False
      if np.array_equal(old_board, board):
        return True
   
        

      # count = 0
      # flag = False
      # for row in range(n):
      #   for col in range(n):
      #     if not old_board[row][col] == board[row][col]:
      #       flag = True
      #     else:
      #       count += 1

      #     if flag:
      #       break
      #   if flag:
      #     break
      # if count == n**2:
      #   print("DUPLICATE MOVE FOUND")
      #   return True
      
    return False
  
  def get_board_type(self, board):
    if (board[1, 1] == 3):
      # Board is big x 
      return "big_x"
    if (board[2,3] == 3 and board[1, 3] != 3):
      # Board is plus1
      return "plus1"
    if (board[1, 3] == 3):
      # board is plus2
      return "plus2"
    if (board[2, 2] == 3):
      # board is point4
      return "point4"
    if (board[1, 2] == 3):
      # Board is the circle
      return "circle"
    if (board[0, 3] == 3):
      # the wall, we want to take over spots that are harder for the opponent to attack
      return "wall"

    if (board[0, 2] == 3):
      # Board is watch the sides 
      return "watch_sides"
    else:
      # Board is empty
      return "empty"    
    

    #bonus_board = [
    #    [ 0,  0,  10,  0,  10,  0,  0],
    #    [ 0,  0,  10,  0,  10,  0,  0],
    #    [ 10,  10, 10,  0, 10,  10,  10],
    #    [ 0,  0,  0,  10,  0,  0,  0],
    #    [ 10,  10, 10,  0, 10,  10,  10],
    #    [ 0,  0,  10,  0,  10,  0,  0],
    #    [ 0,  0,  10,  0,  10,  0,  0]
    #  ]
        

  # give higher priority to moves that can't be attacked by opponent
  def far_from_opp(self, board, move, you, opp):
    n = board.shape[0] 
    for row in range(n):
      safe_count = 0
    total_pieces = 0

    for r in range(n):
        for c in range(n):
            if board[r][c] != you:
                continue

            total_pieces += 1
            threatened = False

            # Look for any opponent piece within Chebyshev distance <= 3
            for dr in range(-3, 4):
                if threatened:
                    break
                for dc in range(-3, 4):
                    if dr == 0 and dc == 0:
                        continue
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        if board[nr][nc] == opp:
                            threatened = True
                            break

            if not threatened:
                safe_count += 1

    if total_pieces == 0:
        return 0.0

    # Fraction of your pieces that are "safe"
    return (int) (safe_count / total_pieces)
    

  def evalMove2(self, board, move, you, opp, i):
    if check_endgame(board):
      return 5000
    player_count = np.count_nonzero(board == you)
    opp_count = np.count_nonzero(board == opp)
    score_diff = player_count - opp_count
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()
    jmp_move = 0
    n = board.shape[0] 
    zeros = n^2 - np.count_nonzero(board)
    
    

    # mvs = get_valid_moves(board, you)
    # for mv in mvs :
    #   at_row2, at_col2 = mv.get_src()
    #   dest_row2, dest_col2 = mv.get_dest()
    #   if abs(dest_row2 - at_row2) == 2 or abs(dest_col2 - at_col2) == 2:
    #     jmp_moves += 1
   

    # corner control bonus
   
   

    if abs(dest_row - at_row) == 2 or abs(dest_col - at_col) == 2:
      zeros = n^2 - np.count_nonzero(board)
      print(player_count + opp_count == np.count_nonzero(board))
      if zeros >= (n**2) / 4:
        jmp_move = - 50
    # # for x in range(n):
    # #   for y in range(n):
    # #     if board[x][y] == 0:
    # #       i = 0
    # #       if  x - 1 >= 0:


    # #       adj = 
    # #   return 0
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    
    corner_bonus = sum(1 for (i, j) in corners if board[i, j] == you) * 5
    # edges = []
    # for x in range(n):
    #   edges.append((0,x))
    #   edges.append((x,0))
    #   edges.append((n-1,x))
    #   edges.append((x,n-1))
    # edge_bonus = sum(1 for (i, j) in edges if board[i, j] == you)
    # penalize opponent mobility
    opp_moves = len(get_valid_moves(board, opp))
    mobility_penalty = -opp_moves

    return score_diff + jmp_move + mobility_penalty

        