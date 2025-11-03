# Student agent: Add your own agent here
import copy
from random import betavariate

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates, get_directions, get_two_tile_directions, count_disc_count_change

# Lightweight timing profiler for method-level benchmarking
import time
import functools

class SimpleProfiler:
  def __init__(self):
    self.data = {}  # label -> {'time': float, 'count': int}

  def _record(self, label, elapsed):
    d = self.data.setdefault(label, {'time': 0.0, 'count': 0})
    d['time'] += elapsed
    d['count'] += 1

  def profile(self, label):
    def decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
          return fn(*args, **kwargs)
        finally:
          t1 = time.perf_counter()
          self._record(label, t1 - t0)
      return wrapper
    return decorator

  def report(self, top=10):
    items = sorted(self.data.items(), key=lambda kv: kv[1]['time'], reverse=True)[:top]
    out = ["Profiler report (label, total_time_s, calls, avg_s):"]
    for label, v in items:
      avg = v['time'] / v['count'] if v['count'] else 0.0
      out.append(f"{label:30} {v['time']:.6f}s  {v['count']:6d}  avg={avg:.6f}s")
    return "\n".join(out)

PROFILER = SimpleProfiler()

def _get_valid_moves(chess_board,player:int) -> list[MoveCoordinates]:
    """
    Vectorized get_valid_moves using numpy broadcasting.
    Returns a list[MoveCoordinates].
    """
    board_h, board_w = chess_board.shape
    # locate source pieces
    src_rows, src_cols = np.nonzero(chess_board == player)

    if src_rows.size == 0:
        return []

    # all offsets to consider
    offsets = np.array(get_directions() + get_two_tile_directions(), dtype=int)  # (M,2)
    M = offsets.shape[0]

    # create (N,1,2) src array and broadcast with (1,M,2) offsets -> (N,M,2) dests
    src = np.stack((src_rows, src_cols), axis=1)[:, None, :]  # (N,1,2)
    dests = src + offsets[None, :, :]                         # (N,M,2)

    # flatten dest coordinates and corresponding src repeats
    dest_rows = dests[..., 0].ravel()
    dest_cols = dests[..., 1].ravel()
    src_rows_rep = np.repeat(src_rows, M)
    src_cols_rep = np.repeat(src_cols, M)

    # mask: dests inside board
    in_bounds = (dest_rows >= 0) & (dest_rows < board_h) & (dest_cols >= 0) & (dest_cols < board_w)
    if not np.any(in_bounds):
        return []

    dest_rows_ib = dest_rows[in_bounds].astype(int)
    dest_cols_ib = dest_cols[in_bounds].astype(int)
    src_rows_ib = src_rows_rep[in_bounds].astype(int)
    src_cols_ib = src_cols_rep[in_bounds].astype(int)

    # mask: dest empty
    empty_mask = (chess_board[dest_rows_ib, dest_cols_ib] == 0)
    if not np.any(empty_mask):
        return []

    final_src_rows = src_rows_ib[empty_mask]
    final_src_cols = src_cols_ib[empty_mask]
    final_dest_rows = dest_rows_ib[empty_mask]
    final_dest_cols = dest_cols_ib[empty_mask]

    # build MoveCoordinates list
    moves = [MoveCoordinates((int(sr), int(sc)), (int(dr), int(dc)))
             for sr, sc, dr, dc in zip(final_src_rows, final_src_cols, final_dest_rows, final_dest_cols)]

    return moves

def print_tree(node, prefix: str = "", is_tail: bool = True):
  """Print tree sideways with branches going upward."""
  if node.parent:
    UCT = node.parent.UCB1(node, 1.4)
  else:
    UCT = 0.0

  if node.children:
    for i, child in enumerate(node.children[:-1]):
      print_tree(child, prefix + ("│   " if not is_tail else "    "), False)
    print_tree(node.children[-1], prefix + ("│   " if not is_tail else "    "), True)
  # print(prefix + ("└── " if is_tail else "┌── ") + f"[{node.minmax}] {node.wins}/{node.visits} UCT:{UCT: .3} Rat:{node.ratio: .3} ")
  print(prefix + ("└── " if is_tail else "┌── ") + f"[{node.minmax}] {node.wins}/{node.visits} UCT:{UCT: .3} ")


class MinimaxNode:
  def __init__(self, chess_board, max_player: int, min_player: int, is_max):
    self.board = chess_board
    self.is_max = is_max
    self.player = max_player if is_max else min_player
    self.opponent = min_player if is_max else max_player
    self.max_player = max_player
    self.min_player = min_player

  def is_max_node(self):
    return self.is_max

  def is_terminal(self):
    is_endgame, _, _ = check_endgame(self.board)
    return is_endgame

  def get_successors(self, valid_moves=None) -> list["MinimaxNode"]:
    if valid_moves is None:
      valid_moves = _get_valid_moves(self.board, self.player)
    succ = []

    for move in valid_moves:
      board_ = copy.deepcopy(self.board)
      execute_move(board_, move, self.player)
      succ.append(MinimaxNode(board_, self.max_player, self.min_player, not self.is_max))

    return succ

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.MAX_DEPTH = 4

    self.random_pool = np.random.randint(0, 48, size=10_000)


  def utility(self, state: MinimaxNode) -> float:
    f1 = np.sum(state.board == state.max_player)  # literally just the nb of discs of StudentAgent
    return f1

  def _minimax(self, node, depth) -> float:
    if depth == self.MAX_DEPTH or node.is_terminal():
      return self.utility(node)

    succ = node.get_successors()
    depth_ = depth + 1

    if node.is_max_node():
      return max([self._minimax(x, depth_) for x in succ])
    return min([self._minimax(x, depth_) for x in succ])


  @PROFILER.profile("StudentAgent.ab_pruning")
  def ab_pruning(self, chess_board, player, opponent) -> MoveCoordinates|None:
    valid_moves = _get_valid_moves(chess_board, player)

    if len(valid_moves) == 0:
      return None

    node = MinimaxNode(chess_board, player, opponent, True)
    succ = node.get_successors(valid_moves)

    l = list(zip(succ, valid_moves))
    l.sort(
      reverse=True,
      key=lambda x: self._minimax(x[0], 2)
    )
    return l[0][1]


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

    # next_move = get_valid_moves(chess_board, player)[0] #this litterally wins against a random agent 82% of the time
    next_move = self.ab_pruning(chess_board, player, opponent)

    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Print profiler summary for this step
    print(PROFILER.report(top=10))

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    # return random_move(chess_board,player)
    return next_move























