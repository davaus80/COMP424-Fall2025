class Agent:
    def __init__(self):
        """
        Initialize the agent, add a name which is used to register the agent
        """
        self.name = "DummyAgent"
        # Flag to indicate whether the agent can be used to autoplay
        self.autoplay = True

    def __str__(self) -> str:
        return self.name

    def step(self, chess_board, player, opponent):
        """
        Main decision logic of the agent, which is called by the simulator.
        Extend this method to implement your own agent to play the game.

        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black (Player 1),
            and 2 for white (Player 2).
        player : int
            The current player (1 for black, 2 for white).
        opponent : int
            The opponent player (1 for black, 2 for white).

        Returns
        -------
        move_pos : tuple of int
            The position (x, y) where the player places the disc.
        """
       
   
        # Get all legal moves for the current player
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        # Apply heuristic: maximize piece difference, corner control, and minimize opponent mobility
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            # evaluate by piece difference, corner bonus, and opponent mobility
            move_score = self.evaluate_board(simulated_board, color, opponent)

            if move_score > best_score:
                best_score = move_score
                best_move = move

        # Return the best move found (or random fallback)
        return best_move or random.choice(legal_moves)

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
        corner_bonus = sum(1 for (i, j) in corners if board[i, j] == color) * 5
        # penalize opponent mobility
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_penalty = -opp_moves
        return score_diff + corner_bonus + mobility_penalty
