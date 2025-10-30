# Student agent: Add your own agent here
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
    
    self.random_pool = np.random.randint(0, 48, size=10_000)

    
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
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return random_move(chess_board,player)

class MCTSNode:
  def __init__(self, state, player=0, parent=None, action=None):
    self.state = state
    self.parent = parent
    self.action = action
    self.children = []

    self.player = player

    self.visits = 0
    self.wins  = 0

    #ratio of friendly vs hostile tiles may prove usefull
    self.ratio = 0.5
    self.untried_action = get_valid_moves(self.state, player)

    def is_terminal(self):
      is_endgame = False

      if np.sum(self.state == 0) == 0:
          is_endgame = True
      
      return is_endgame
    
    def is_fully_expanded(self):
      return len(self.untried_action) == 0  


    def expand(self):
      #consider switching to the queue library to make this efficient
      action = self.untried_actions.pop()
      new_state = execute_move(self.state, action, self.player)
      
      #flip player
      new_player = self.player ^ 1

      child = MCTSNode(new_state, player= new_player, parent=self, action=action)
      return child

    def best_child(self):
      pass
        

    def UCB1(self, c=1.4):
      pass






  











