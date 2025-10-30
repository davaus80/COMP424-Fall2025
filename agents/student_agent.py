# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

# random_pool = np.random.randint(0, 48, size=10_000)


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

  def mcts_search(self, root_state, player, iter=500):
    root = MCTSNode(root_state, player, minmax=0)

    for _ in range(iter):
      node = root

      while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

      if not node.is_terminal():
        node = node.expand()

          # Simulation
      result = node.rollout()

      # Backpropagation
      node.backpropagate(result)

    return root.best_child(c=0).action
      
    
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
  def __init__(self, state, player=1, minmax=0, parent=None, action=None, rng=np.random.default_rng()):
    self.state = state
    self.parent = parent
    self.action = action
    self.children = []

    self.player = player
    self.minmax = minmax #0 for max 1 for min players

    self.visits = 0
    self.wins  = 0

    self.rng = rng

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
    """
    Expand new child node
    """
    #consider switching to the queue library to make this efficient
    action = self.untried_action.pop()
    new_state = execute_move(self.state, action, self.player)
    
    #flip player
    new_player = 3 - self.player 
    minmax = 1 - self.minmax
    child = MCTSNode(new_state, player=new_player, minmax=minmax, parent=self, action=action, rng=self.rng) 
    return child

  def best_child(self, c=1.4):
    """
    Return best child using upper confidence tree comparison.  
    """
    if self.minmax == 0: #max player
      return max(self.children, key=lambda child: self.UCB1(child, c))
    else:
      return min(self.children, key=lambda child: self.UCB1(child, -c))

        
  def UCB1(self, child, c):
    return (child.wins/child.visits) + c * np.sqrt(np.log(self.visits)/child.visits)
  
  def rollout(self):
    """
    Simulate game until completion 
    """
    curr_state = np.copy(self.state)
    curr_player = self.player
    while True:
      allowed_moves = get_valid_moves(curr_state, curr_player)
      number_allowed_moves = len(allowed_moves)

      #grab a random move
      move = allowed_moves[self.rng.integers(0, number_allowed_moves)]

      curr_state = execute_move(curr_state, move, curr_player)
      is_endgame, p0, p1 = check_endgame(curr_state)
      if is_endgame:
        if p0 > p1: return 1
        if p0 < p1: return -1
        else: return 0
      curr_player = 3 - curr_player

  def backpropagate(self, result):
    """
    Update tree
    """
    self.visits += 1
    self.wins += result
    
    if self.parent:
      self.parent.backpropagate(result)












  











