import chess
from typing import Callable, Dict

def king_danger_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    my_king_square = board.king(player_to_move)
    opponent = not player_to_move

    squares_around_king = set(chess.SquareSet(chess.BB_KING_ATTACKS[my_king_square]))
    attacked_squares = 0
    for square in squares_around_king:
        if board.is_attacked_by(opponent, square):
            attacked_squares += 1

    return 1 if attacked_squares >= 5 else 0

def king_danger_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    opponent_king_square = board.king(opponent)

    squares_around_king = set(chess.SquareSet(chess.BB_KING_ATTACKS[opponent_king_square]))
    attacked_squares = 0
    for square in squares_around_king:
        if board.is_attacked_by(player_to_move, square):
            attacked_squares += 1

    return 1 if attacked_squares >= 5 else 0

def is_square_under_defensive_threat(fen: str, square_index: int) -> int:
    board = chess.Board(fen)
    piece = board.piece_at(square_index)

    if piece is None or piece.color != chess.WHITE:
        return 0

    attackers = board.attackers(chess.BLACK, square_index)
    defenders = board.attackers(chess.WHITE, square_index)

    return 1 if len(attackers) > len(defenders) else 0

def is_square_under_offensive_threat(fen: str, square_index: int) -> int:
    board = chess.Board(fen)
    piece = board.piece_at(square_index)

    if piece is None or piece.color != chess.BLACK:
        return 0

    attackers = board.attackers(chess.WHITE, square_index)
    defenders = board.attackers(chess.BLACK, square_index)

    return 1 if len(attackers) > len(defenders) else 0

def pawn_fork_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    pawn_squares = board.pieces(chess.PAWN, player_to_move)

    for pawn_square in pawn_squares:
        if board.is_pinned(player_to_move, pawn_square):
            continue

        attacks = board.attacks(pawn_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)

            if attacked_piece and attacked_piece.color != player_to_move:
                if attacked_piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def pawn_fork_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    pawn_squares = board.pieces(chess.PAWN, opponent)

    for pawn_square in pawn_squares:
        if board.is_pinned(opponent, pawn_square):
            continue

        attacks = board.attacks(pawn_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)

            if attacked_piece and attacked_piece.color != opponent:
                if attacked_piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def knight_fork_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    knight_squares = board.pieces(chess.KNIGHT, player_to_move)

    for knight_square in knight_squares:
        if board.is_pinned(player_to_move, knight_square):
            continue

        attacks = board.attacks(knight_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)
            if attacked_piece and attacked_piece.color != player_to_move:
                if attacked_piece.piece_type in [chess.ROOK, chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def knight_fork_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    knight_squares = board.pieces(chess.KNIGHT, opponent)

    for knight_square in knight_squares:
        if board.is_pinned(opponent, knight_square):
            continue

        attacks = board.attacks(knight_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)

            if attacked_piece and attacked_piece.color != opponent:
                if attacked_piece.piece_type in [chess.ROOK, chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def bishop_fork_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    for bishop_square in bishop_squares:
        if board.is_pinned(player_to_move, bishop_square):
            continue

        attacks = board.attacks(bishop_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)
            if attacked_piece and attacked_piece.color != player_to_move:
                if attacked_piece.piece_type in [chess.ROOK, chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def bishop_fork_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    bishop_squares = board.pieces(chess.BISHOP, opponent)

    for bishop_square in bishop_squares:
        if board.is_pinned(opponent, bishop_square):
            continue

        attacks = board.attacks(bishop_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)

            if attacked_piece and attacked_piece.color != opponent:
                if attacked_piece.piece_type in [chess.ROOK, chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def rook_fork_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    rook_squares = board.pieces(chess.ROOK, player_to_move)

    for rook_square in rook_squares:
        if board.is_pinned(player_to_move, rook_square):
            continue

        attacks = board.attacks(rook_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)
            if attacked_piece and attacked_piece.color != player_to_move:
                if attacked_piece.piece_type in [chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def rook_fork_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    rook_squares = board.pieces(chess.ROOK, opponent)

    for rook_square in rook_squares:
        if board.is_pinned(opponent, rook_square):
            continue

        attacks = board.attacks(rook_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)

            if attacked_piece and attacked_piece.color != opponent:
                if attacked_piece.piece_type in [chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def has_pinned_pawn_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.PAWN, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_pawn_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.PAWN, opponent)

    for square in squares:
        if board.is_pinned(opponent, square):
            return 1

    return 0

def has_pinned_knight_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.KNIGHT, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_knight_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.KNIGHT, opponent)

    for square in squares:
        if board.is_pinned(opponent, square):
            return 1

    return 0

def has_pinned_bishop_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.BISHOP, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_bishop_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.BISHOP, opponent)

    for square in squares:
        if board.is_pinned(opponent, square):
            return 1

    return 0

def has_pinned_rook_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.ROOK, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_rook_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.ROOK, opponent)

    for square in squares:
        if board.is_pinned(opponent, square):
            return 1

    return 0

def has_pinned_queen_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.QUEEN, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_queen_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.QUEEN, opponent)

    for square in squares:
        if board.is_pinned(opponent, square):
            return 1

    return 0

def has_connected_rooks_mine(pos: str) -> int:
    board = chess.Board(pos)

    player_to_move = board.turn
    rook_squares = board.pieces(chess.ROOK, player_to_move)
    if len(rook_squares) < 2:
        return 0

    for square in rook_squares:
        for other_square in rook_squares:
            if square == other_square:
                continue

            if chess.square_rank(square) == chess.square_rank(other_square):
                if all(board.piece_at(chess.square(chess.square_file(sq), chess.square_rank(square))) is None for sq in range(min(chess.square_file(square), chess.square_file(other_square))+1, max(chess.square_file(square), chess.square_file(other_square)))):
                    return 1
            elif chess.square_file(square) == chess.square_file(other_square):
                if all(board.piece_at(chess.square(chess.square_file(square), chess.square_rank(sq))) is None for sq in range(min(chess.square_rank(square), chess.square_rank(other_square))+1, max(chess.square_rank(square), chess.square_rank(other_square)))):
                    return 1

    return 0

def has_connected_rooks_opponent(pos: str) -> int:
    board = chess.Board(pos)

    player_to_move = board.turn
    opponent = not player_to_move
    rook_squares = board.pieces(chess.ROOK, opponent)
    if len(rook_squares) < 2:
        return 0

    for square in rook_squares:
        for other_square in rook_squares:
            if square == other_square:
                continue

            if chess.square_rank(square) == chess.square_rank(other_square):
                if all(board.piece_at(chess.square(chess.square_file(sq), chess.square_rank(square))) is None for sq in range(min(chess.square_file(square), chess.square_file(other_square))+1, max(chess.square_file(square), chess.square_file(other_square)))):
                    return 1
            elif chess.square_file(square) == chess.square_file(other_square):
                if all(board.piece_at(chess.square(chess.square_file(square), chess.square_rank(sq))) is None for sq in range(min(chess.square_rank(square), chess.square_rank(other_square))+1, max(chess.square_rank(square), chess.square_rank(other_square)))):
                    return 1

    return 0

def has_bishop_pair_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn

    if len(board.pieces(chess.BISHOP, player_to_move)) == 2:
        return 1

    return 0

def has_bishop_pair_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move

    if len(board.pieces(chess.BISHOP, opponent)) == 2:
        return 1

    return 0

def has_control_of_open_file_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move

    for file in range(8):
        if not any(board.piece_at(chess.square(file, rank)) and board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN
                   for rank in range(8)):
            control_flag = False
            for rank in range(8):
                piece = board.piece_at(chess.square(file, rank))

                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == opponent:
                    control_flag = False
                    break
                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == player_to_move:
                    control_flag = True

            if control_flag:
                return 1

    return 0

def has_control_of_open_file_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move

    for file in range(8):
        if not any(board.piece_at(chess.square(file, rank)) and board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN
                   for rank in range(8)):
            control_flag = False
            for rank in range(8):
                piece = board.piece_at(chess.square(file, rank))

                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == player_to_move:
                    control_flag = False
                    break
                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == opponent:
                    control_flag = True

            if control_flag:
                return 1

    return 0

def has_contested_open_file(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move

    for file in range(8):
        if not any(board.piece_at(chess.square(file, rank)) and board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN
                   for rank in range(8)):
            control_flag_opp = False
            control_flag_mine = False
            for rank in range(8):
                piece = board.piece_at(chess.square(file, rank))

                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == opponent:
                    control_flag_opp = True
                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == player_to_move:
                    control_flag_mine = True

                if control_flag_opp and control_flag_mine:
                    return 1

    return 0

def can_capture_queen_mine(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    queen_square = board.pieces(chess.QUEEN, opponent)

    for move in board.legal_moves:
        if move.to_square in queen_square:
            return 1

    return 0

def can_capture_queen_opponent(pos: str) -> int:
    board = chess.Board(pos)
    player_to_move = board.turn
    queen_square = board.pieces(chess.QUEEN, player_to_move)
    board.turn = not board.turn

    for move in board.legal_moves:
        if move.to_square in queen_square:
            return 1

    return 0

def create_capture_possible_function(target_square_name: str, is_mine: bool) -> Callable[[str], int]:
    def capture_possible(pos: str) -> int:
        board = chess.Board(pos)
        if not is_mine:
            board.turn = not board.turn

        player_to_move = board.turn
        target_square = getattr(chess, target_square_name.upper()) if player_to_move == chess.WHITE else getattr(chess, target_square_name.upper()[0] + str(9 - int(target_square_name[1])))

        for move in board.legal_moves:
            if move.to_square == target_square and board.is_capture(move):
                return 1
        return 0
    return capture_possible

def get_all_concept_functions() -> Dict[str, Callable[[str], int]]:
    functions = {}

    functions['king_danger_mine'] = king_danger_mine
    functions['king_danger_opponent'] = king_danger_opponent

    for square in range(64):
        square_name = chess.square_name(square)
        functions[f'defensive_threat_{square_name}'] = lambda pos, sq=square: is_square_under_defensive_threat(pos, sq)
        functions[f'offensive_threat_{square_name}'] = lambda pos, sq=square: is_square_under_offensive_threat(pos, sq)

    functions['pawn_fork_mine'] = pawn_fork_mine
    functions['pawn_fork_opponent'] = pawn_fork_opponent
    functions['knight_fork_mine'] = knight_fork_mine
    functions['knight_fork_opponent'] = knight_fork_opponent
    functions['bishop_fork_mine'] = bishop_fork_mine
    functions['bishop_fork_opponent'] = bishop_fork_opponent
    functions['rook_fork_mine'] = rook_fork_mine
    functions['rook_fork_opponent'] = rook_fork_opponent

    functions['has_pinned_pawn_mine'] = has_pinned_pawn_mine
    functions['has_pinned_pawn_opponent'] = has_pinned_pawn_opponent
    functions['has_pinned_knight_mine'] = has_pinned_knight_mine
    functions['has_pinned_knight_opponent'] = has_pinned_knight_opponent
    functions['has_pinned_bishop_mine'] = has_pinned_bishop_mine
    functions['has_pinned_bishop_opponent'] = has_pinned_bishop_opponent
    functions['has_pinned_rook_mine'] = has_pinned_rook_mine
    functions['has_pinned_rook_opponent'] = has_pinned_rook_opponent
    functions['has_pinned_queen_mine'] = has_pinned_queen_mine
    functions['has_pinned_queen_opponent'] = has_pinned_queen_opponent

    functions['has_connected_rooks_mine'] = has_connected_rooks_mine
    functions['has_connected_rooks_opponent'] = has_connected_rooks_opponent
    functions['has_bishop_pair_mine'] = has_bishop_pair_mine
    functions['has_bishop_pair_opponent'] = has_bishop_pair_opponent
    functions['has_control_of_open_file_mine'] = has_control_of_open_file_mine
    functions['has_control_of_open_file_opponent'] = has_control_of_open_file_opponent
    functions['has_contested_open_file'] = has_contested_open_file

    functions['can_capture_queen_mine'] = can_capture_queen_mine
    functions['can_capture_queen_opponent'] = can_capture_queen_opponent

    key_squares = ['d1', 'd2', 'd3', 'e1', 'e2', 'e3', 'g5', 'b5']
    for square_name in key_squares:
        functions[f'capture_possible_{square_name}_mine'] = create_capture_possible_function(square_name, True)
        functions[f'capture_possible_{square_name}_opponent'] = create_capture_possible_function(square_name, False)

    return functions

CONCEPT_CATEGORIES = {
    'king_queen_safety': ['king_danger_mine', 'king_danger_opponent', 'can_capture_queen_mine', 'can_capture_queen_opponent'],
    'square_threats': [f'defensive_threat_{chess.square_name(i)}' for i in range(64)] + [f'offensive_threat_{chess.square_name(i)}' for i in range(64)],
    'tactical_patterns_forks': ['pawn_fork_mine', 'pawn_fork_opponent', 'knight_fork_mine', 'knight_fork_opponent',
                                'bishop_fork_mine', 'bishop_fork_opponent', 'rook_fork_opponent'],
    'tactical_patterns_pins': ['has_pinned_pawn_mine', 'has_pinned_pawn_opponent', 'has_pinned_knight_mine', 'has_pinned_knight_opponent',
                               'has_pinned_bishop_mine', 'has_pinned_bishop_opponent', 'has_pinned_rook_mine', 'has_pinned_rook_opponent',
                               'has_pinned_queen_mine', 'has_pinned_queen_opponent'],
    'positional_structure': ['has_connected_rooks_mine', 'has_connected_rooks_opponent', 'has_bishop_pair_mine', 'has_bishop_pair_opponent',
                             'has_control_of_open_file_mine', 'has_control_of_open_file_opponent', 'has_contested_open_file'] +
                            [f'capture_possible_{sq}_mine' for sq in ['d1', 'd2', 'd3', 'e1', 'e2', 'e3', 'g5', 'b5']] +
                            [f'capture_possible_{sq}_opponent' for sq in ['d1', 'd2', 'd3', 'e1', 'e2', 'e3', 'g5', 'b5']],
}

TOTAL_CONCEPTS = sum(len(v) for v in CONCEPT_CATEGORIES.values())
