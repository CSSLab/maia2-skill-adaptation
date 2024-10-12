import subprocess
import csv
import chess
#import bz2
#import maia_lib
import tqdm
import json
import numpy as np
import argparse
import pandas as pd
import inspect

def calculate_features(fen):
    command = f'node -e "console.log(JSON.stringify(require(\'./index2.js\').lf(\'{fen}\')))"'
    try:
        output = subprocess.check_output(command, shell=True).decode("utf-8").strip()
        print(fen)
        return json.loads(output)
    except Exception as e:
        print(f"Error occurred while calculating features for FEN: {fen}")
        print(f"Command: {command}")
        raise e

def flip_board(fen_str):
    board = chess.Board(fen=fen_str)
    if board.turn == chess.BLACK:
        board = board.mirror()
    return board.fen()


def square_name(square_idx: int) -> str:
    if square_idx < 0 or square_idx > 63:
        raise ValueError("Square index must be between 0 and 63")

    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']

    file_idx = square_idx % 8  
    rank_idx = square_idx // 8  

    return files[file_idx] + ranks[rank_idx]

# Self-implemented custom concepts; Python-chess version

def is_piece_on_square(piece: str, square: int, pos: str) -> int:
    # Ensure the square is in the valid range
    if square < 0 or square > 63:
        raise ValueError("Square must be an integer between 0 and 63.")

    # Convert the square integer into corresponding rank and file
    rank_idx = square // 8  # rank is determined by dividing by 8
    file_idx = square % 8  # file is determined by the remainder when dividing by 8

    # Split the FEN into board configuration (the first part)
    board_fen = pos.split()[0]

    # Convert the FEN into a 2D board array
    board = []
    for row in board_fen.split('/'):
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['.'] * int(char))  # empty squares as '.'
            else:
                board_row.append(char)
        board.append(board_row)

    return int(board[rank_idx][file_idx] == piece)


def in_check(pos):
    board = chess.Board(pos)
    
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    if white_king_square is not None and black_king_square is not None and board.is_check():
        return 1

    return 0

def has_mate_threat(pos):
    board = chess.Board(pos)
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return 1
        board.pop()
    return 0

def has_connected_rooks_mine(pos):
    board = chess.Board(pos)
    
    player_to_move = board.turn
    rook_squares = board.pieces(chess.ROOK, player_to_move)
    if len(rook_squares) < 2:
        return 0

    for square in rook_squares:
        for other_square in rook_squares:
            if square == other_square:
                continue
            
            # Check if two rooks are on the same rank or file and no other pieces lying between them.
            if chess.square_rank(square) == chess.square_rank(other_square):
                if all(board.piece_at(chess.square(chess.square_file(sq), chess.square_rank(square))) is None for sq in range(min(chess.square_file(square), chess.square_file(other_square))+1, max(chess.square_file(square), chess.square_file(other_square)))):
                    return 1
            elif chess.square_file(square) == chess.square_file(other_square):
                if all(board.piece_at(chess.square(chess.square_file(square), chess.square_rank(sq))) is None for sq in range(min(chess.square_rank(square), chess.square_rank(other_square))+1, max(chess.square_rank(square), chess.square_rank(other_square)))):
                    return 1

    return 0

def has_connected_rooks_opponent(pos):
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
            
            # Check if two rooks are on the same rank or file and no other pieces lying between them.
            if chess.square_rank(square) == chess.square_rank(other_square):
                if all(board.piece_at(chess.square(chess.square_file(sq), chess.square_rank(square))) is None for sq in range(min(chess.square_file(square), chess.square_file(other_square))+1, max(chess.square_file(square), chess.square_file(other_square)))):
                    return 1
            elif chess.square_file(square) == chess.square_file(other_square):
                if all(board.piece_at(chess.square(chess.square_file(square), chess.square_rank(sq))) is None for sq in range(min(chess.square_rank(square), chess.square_rank(other_square))+1, max(chess.square_rank(square), chess.square_rank(other_square)))):
                    return 1

    return 0

def pawn_fork_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    pawn_squares = board.pieces(chess.PAWN, player_to_move)

    for pawn_square in pawn_squares:

        # If the current pawn is pinned, continue;
        if board.is_pinned(player_to_move, pawn_square):
            continue

        attacks = board.attacks(pawn_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)
            
            if attacked_piece and attacked_piece.color != player_to_move:
                if attacked_piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        # If a single pawn is attacking at least two pieces of higher value, return True
        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def pawn_fork_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    pawn_squares = board.pieces(chess.PAWN, opponent)

    for pawn_square in pawn_squares:

        # If the current pawn is pinned, continue;
        if board.is_pinned(opponent, pawn_square):
            continue

        attacks = board.attacks(pawn_square)
        attacked_higher_value_pieces = 0

        for attack_square in attacks:
            attacked_piece = board.piece_at(attack_square)
            
            if attacked_piece and attacked_piece.color != opponent:
                if attacked_piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                    attacked_higher_value_pieces += 1

        # If a single pawn is attacking at least two pieces of higher value, return True
        if attacked_higher_value_pieces >= 2:
            return 1

    return 0

def knight_fork_mine(pos):
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

def knight_fork_opponent(pos):
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

def bishop_fork_mine(pos):
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

def bishop_fork_opponent(pos):
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

def rook_fork_mine(pos):
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

def rook_fork_opponent(pos):
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

def has_pinned_pawn_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.PAWN, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_pawn_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.PAWN, opponent)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_knight_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.KNIGHT, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_knight_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.KNIGHT, opponent)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_bishop_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.BISHOP, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_bishop_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.BISHOP, opponent)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_rook_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.ROOK, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_rook_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.ROOK, opponent)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_queen_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    squares = board.pieces(chess.QUEEN, player_to_move)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def has_pinned_queen_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    squares = board.pieces(chess.QUEEN, opponent)

    for square in squares:
        if board.is_pinned(player_to_move, square):
            return 1

    return 0

def material_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn

    material_score = (len(board.pieces(chess.PAWN, player_to_move))) + \
                     (3 * len(board.pieces(chess.KNIGHT, player_to_move))) + \
                     (3 * len(board.pieces(chess.BISHOP, player_to_move))) + \
                     (5 * len(board.pieces(chess.ROOK, player_to_move))) + \
                     (9 * len(board.pieces(chess.QUEEN, player_to_move)))
                      
    return material_score

def material_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move

    material_score = (len(board.pieces(chess.PAWN, opponent))) + \
                     (3 * len(board.pieces(chess.KNIGHT, opponent))) + \
                     (3 * len(board.pieces(chess.BISHOP, opponent))) + \
                     (5 * len(board.pieces(chess.ROOK, opponent))) + \
                     (9 * len(board.pieces(chess.QUEEN, opponent)))
                      
    return material_score

def material_diff(pos):
    return material_mine(pos) - material_opponent(pos)

def num_pieces_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn

    num_score = (len(board.pieces(chess.PAWN, player_to_move))) + \
                     (len(board.pieces(chess.KNIGHT, player_to_move))) + \
                     (len(board.pieces(chess.BISHOP, player_to_move))) + \
                     (len(board.pieces(chess.ROOK, player_to_move))) + \
                     (len(board.pieces(chess.QUEEN, player_to_move)))
                      
    return num_score

def num_pieces_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move

    num_score = (len(board.pieces(chess.PAWN, opponent))) + \
                     (len(board.pieces(chess.KNIGHT, opponent))) + \
                     (len(board.pieces(chess.BISHOP, opponent))) + \
                     (len(board.pieces(chess.ROOK, opponent))) + \
                     (len(board.pieces(chess.QUEEN, opponent)))
                      
    return num_score

def num_pieces_diff(pos):
    return num_pieces_mine(pos) - num_pieces_opponent(pos)

def has_bishop_pair_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn

    if len(board.pieces(chess.BISHOP, player_to_move)) == 2:
        return 1
    
    return 0

def has_bishop_pair_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move

    if len(board.pieces(chess.BISHOP, opponent)) == 2:
        return 1
    
    return 0

def has_control_of_open_file_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move

    for file in range(8):
        if not any(board.piece_at(chess.square(file, rank)) and board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN
                   for rank in range(8)):
            control_flag = False
            for rank in range(8):
                piece = board.piece_at(chess.square(file, rank))

                # Clarification required here. If there are r/q of your opponent, return false no matter how. 
                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == opponent:
                    control_flag = False
                    break
                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == player_to_move:
                    control_flag = True

            if control_flag:
                return 1

    return 0

# Clarification required here. If there are pieces of your opponent, return false no matter how. 
def has_control_of_open_file_opponent(pos):
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

def can_capture_queen_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    queen_square = board.pieces(chess.QUEEN, opponent)

    for move in board.legal_moves:
        if move.to_square in queen_square:
            return 1

    return 0

def can_capture_queen_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    queen_square = board.pieces(chess.QUEEN, player_to_move)
    board.turn = not board.turn

    for move in board.legal_moves:
        if move.to_square in queen_square:
            return 1

    return 0

def num_king_attacked_squares_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    opponent_king_square = board.king(opponent)
    squares_around_king = set(chess.SquareSet(chess.BB_KING_ATTACKS[opponent_king_square]))

    attacked_squares = 0
    for square in squares_around_king:
        if board.is_attacked_by(player_to_move, square):
            attacked_squares += 1

    return attacked_squares

def num_king_attacked_squares_opponent(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    opponent = not player_to_move
    my_king_square = board.king(player_to_move)
    squares_around_king = set(chess.SquareSet(chess.BB_KING_ATTACKS[my_king_square]))

    attacked_squares = 0
    for square in squares_around_king:
        if board.is_attacked_by(player_to_move, square):
            attacked_squares += 1

    return attacked_squares 

def num_king_attacked_squares_diff(pos):
    return num_king_attacked_squares_mine(pos) - num_king_attacked_squares_opponent(pos)

def has_contested_open_file(pos):
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

                # Clarification required here. If there are r/q of your opponent, return false no matter how. 
                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == opponent:
                    control_flag_opp = True
                if piece and piece.piece_type in [chess.ROOK, chess.QUEEN] and piece.color == player_to_move:
                    control_flag_mine = True

                if control_flag_opp and control_flag_mine:
                    return 1

    return 0

# Clarification Required: 
# Context: The squares are named as if the side were playing White.
# Does it mean White capturing D1 or Black capturing D8?
def capture_happens_next_move_on_d1(pos, move):
    board = chess.Board(pos)
    move = chess.Move.from_uci(move)
    target_square = chess.D1 if board.turn == chess.WHITE else chess.D8

    if move.to_square == target_square and board.is_capture(move):
        return 1
    else:
        return 0

def capture_happens_next_move_on_d2(pos, move):
    board = chess.Board(pos)
    move = chess.Move.from_uci(move)
    target_square = chess.D2 if board.turn == chess.WHITE else chess.D7

    if move.to_square == target_square and board.is_capture(move):
        return 1
    else:
        return 0

def capture_happens_next_move_on_d3(pos, move):
    board = chess.Board(pos)
    move = chess.Move.from_uci(move)
    target_square = chess.D3 if board.turn == chess.WHITE else chess.D6

    if move.to_square == target_square and board.is_capture(move):
        return 1
    else:
        return 0

def capture_happens_next_move_on_e1(pos, move):
    board = chess.Board(pos)
    move = chess.Move.from_uci(move)
    target_square = chess.E1 if board.turn == chess.WHITE else chess.E8

    if move.to_square == target_square and board.is_capture(move):
        return 1
    else:
        return 0
    
def capture_happens_next_move_on_e2(pos, move):
    board = chess.Board(pos)
    move = chess.Move.from_uci(move) 
    target_square = chess.E2 if board.turn == chess.WHITE else chess.E7

    if move.to_square == target_square and board.is_capture(move):
        return 1
    else:
        return 0

def capture_happens_next_move_on_e3(pos, move):
    board = chess.Board(pos)
    move = chess.Move.from_uci(move)
    target_square = chess.E3 if board.turn == chess.WHITE else chess.E6

    if move.to_square == target_square and board.is_capture(move):
        return 1
    else:
        return 0

def capture_happens_next_move_on_g5(pos, move):
    board = chess.Board(pos)
    move = chess.Move.from_uci(move)
    target_square = chess.G5 if board.turn == chess.WHITE else chess.G4

    if move.to_square == target_square and board.is_capture(move):
        return 1
    else:
        return 0

def capture_happens_next_move_on_b5(pos, move):
    board = chess.Board(pos)
    move = chess.Move.from_uci(move)
    target_square = chess.B5 if board.turn == chess.WHITE else chess.B4

    if move.to_square == target_square and board.is_capture(move):
        return 1
    else:
        return 0

# helper function
def is_passed_pawn(board, square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    color = board.piece_at(square).color
    oppo_color = not color

    # Check all squares on the same or higher ranks on the same, previous and next file
    for r in range(rank, 7 if color == chess.WHITE else 0, 1 if color == chess.WHITE else -1):
        for f in range(max(0, file - 1), min(7, file + 1) + 1):
            if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, oppo_color):
                return False

    return True

def has_right_bc_ha_promotion_mine(pos):
    board = chess.Board(pos)
    color = board.turn
    a_file_pawn_square = h_file_pawn_square = None

    for rank in range(8):
        if board.piece_at(chess.square(0, rank)) == chess.Piece(chess.PAWN, color):
            a_file_pawn_square = chess.square(0, rank)
        if board.piece_at(chess.square(7, rank)) == chess.Piece(chess.PAWN, color):
            h_file_pawn_square = chess.square(7, rank)

    if a_file_pawn_square is None and h_file_pawn_square is None:
        return 0

    for pawn_square in [a_file_pawn_square, h_file_pawn_square]:
        if pawn_square is not None:
            if is_passed_pawn(board, pawn_square):
                promotion_square = chess.square(chess.square_file(pawn_square), 7 if color == chess.WHITE else 0)
                bishop_squares = board.pieces(chess.BISHOP, color)
                for bishop_square in bishop_squares:
                    if (chess.square_rank(promotion_square) + chess.square_file(promotion_square)) % 2 == (chess.square_rank(bishop_square) + chess.square_file(bishop_square)) % 2:
                        return 1
    return 0

def has_right_bc_ha_promotion_opponent(pos):
    board = chess.Board(pos)
    color = not board.turn
    a_file_pawn_square = h_file_pawn_square = None

    for rank in range(8):
        if board.piece_at(chess.square(0, rank)) == chess.Piece(chess.PAWN, color):
            a_file_pawn_square = chess.square(0, rank)
        if board.piece_at(chess.square(7, rank)) == chess.Piece(chess.PAWN, color):
            h_file_pawn_square = chess.square(7, rank)

    if a_file_pawn_square is None and h_file_pawn_square is None:
        return 0

    for pawn_square in [a_file_pawn_square, h_file_pawn_square]:
        if pawn_square is not None:
            if is_passed_pawn(board, pawn_square):
                promotion_square = chess.square(chess.square_file(pawn_square), 7 if color == chess.WHITE else 0)
                bishop_squares = board.pieces(chess.BISHOP, color)
                for bishop_square in bishop_squares:
                    if (chess.square_rank(promotion_square) + chess.square_file(promotion_square)) % 2 == (chess.square_rank(bishop_square) + chess.square_file(bishop_square)) % 2:
                        return 1
    return 0

def capture_possible_on_d1_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    target_square = chess.D1 if player_to_move == chess.WHITE else chess.D8
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_d2_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    target_square = chess.D2 if player_to_move == chess.WHITE else chess.D7
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_d3_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    target_square = chess.D3 if player_to_move == chess.WHITE else chess.D6
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_e1_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    target_square = chess.E1 if player_to_move == chess.WHITE else chess.E8
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_e2_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    target_square = chess.E2 if player_to_move == chess.WHITE else chess.E7
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_e3_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    target_square = chess.E3 if player_to_move == chess.WHITE else chess.E6
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_g5_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    target_square = chess.G5 if player_to_move == chess.WHITE else chess.G4
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_b5_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    target_square = chess.B5 if player_to_move == chess.WHITE else chess.B4
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_d1_opponent(pos):
    board = chess.Board(pos)
    board.turn = not board.turn
    player_to_move = board.turn
    target_square = chess.D1 if player_to_move == chess.WHITE else chess.D8
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_d2_opponent(pos):
    board = chess.Board(pos)
    board.turn = not board.turn
    player_to_move = board.turn
    target_square = chess.D2 if player_to_move == chess.WHITE else chess.D7
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_d3_opponent(pos):
    board = chess.Board(pos)
    board.turn = not board.turn
    player_to_move = board.turn
    target_square = chess.D3 if player_to_move == chess.WHITE else chess.D6
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_e1_opponent(pos):
    board = chess.Board(pos)
    board.turn = not board.turn
    player_to_move = board.turn
    target_square = chess.E1 if player_to_move == chess.WHITE else chess.E8
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_e2_opponent(pos):
    board = chess.Board(pos)
    board.turn = not board.turn
    player_to_move = board.turn
    target_square = chess.E2 if player_to_move == chess.WHITE else chess.E7
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_e3_opponent(pos):
    board = chess.Board(pos)
    board.turn = not board.turn
    player_to_move = board.turn
    target_square = chess.E3 if player_to_move == chess.WHITE else chess.E6
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_g5_opponent(pos):
    board = chess.Board(pos)
    board.turn = not board.turn
    player_to_move = board.turn
    target_square = chess.G5 if player_to_move == chess.WHITE else chess.G4
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def capture_possible_on_b5_opponent(pos):
    board = chess.Board(pos)
    board.turn = not board.turn
    player_to_move = board.turn
    target_square = chess.B5 if player_to_move == chess.WHITE else chess.B4
    for move in board.legal_moves:
        if move.to_square == target_square and board.is_capture(move):
            return 1
    return 0

def num_scb_pawns_same_side_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    if len(bishop_squares) != 1:
        return 0
    
    bishop_square = list(bishop_squares)[0]
    bishop_color = ((chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2)
    pawn_squares = board.pieces(chess.PAWN, player_to_move)
    
    count = 0
    for pawn_square in pawn_squares:
        if (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 == bishop_color:
            count += 1
            
    return count

def num_scb_pawns_same_side_opponent(pos):
    board = chess.Board(pos)
    player_to_move = not board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    if len(bishop_squares) != 1:
        return 0
    
    bishop_square = list(bishop_squares)[0]
    bishop_color = (chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2 == 0
    pawn_squares = board.pieces(chess.PAWN, player_to_move)
    
    count = 0
    for pawn_square in pawn_squares:
        if (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 == bishop_color:
            count += 1
            
    return count

def num_scb_pawns_same_side_diff(pos):
    return num_scb_pawns_same_side_mine(pos) - num_scb_pawns_same_side_opponent(pos)

def num_ocb_pawns_same_side_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    if len(bishop_squares) != 1:
        return 0
    
    bishop_square = list(bishop_squares)[0]
    bishop_color = ((chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2)
    pawn_squares = board.pieces(chess.PAWN, player_to_move)
    
    count = 0
    for pawn_square in pawn_squares:
        if (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 != bishop_color:
            count += 1
            
    return count

def num_ocb_pawns_same_side_opponent(pos):
    board = chess.Board(pos)
    player_to_move = not board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    if len(bishop_squares) != 1:
        return 0
    
    bishop_square = list(bishop_squares)[0]
    bishop_color = (chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2 == 0
    pawn_squares = board.pieces(chess.PAWN, player_to_move)
    
    count = 0
    for pawn_square in pawn_squares:
        if (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 != bishop_color:
            count += 1
            
    return count

def num_ocb_pawns_same_side_diff(pos):
    return num_ocb_pawns_same_side_mine(pos) - num_ocb_pawns_same_side_opponent(pos)

def num_scb_pawns_other_side_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    if len(bishop_squares) != 1:
        return 0
    
    bishop_square = list(bishop_squares)[0]
    bishop_color = ((chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2)
    pawn_squares = board.pieces(chess.PAWN, not player_to_move)
    
    count = 0
    for pawn_square in pawn_squares:
        if (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 == bishop_color:
            count += 1
            
    return count

def num_scb_pawns_other_side_opponent(pos):
    board = chess.Board(pos)
    player_to_move = not board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    if len(bishop_squares) != 1:
        return 0
    
    bishop_square = list(bishop_squares)[0]
    bishop_color = (chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2 == 0
    pawn_squares = board.pieces(chess.PAWN, not player_to_move)
    
    count = 0
    for pawn_square in pawn_squares:
        if (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 == bishop_color:
            count += 1
            
    return count

def num_scb_pawns_other_side_diff(pos):
    return num_scb_pawns_other_side_mine(pos) - num_scb_pawns_other_side_opponent(pos)

def num_ocb_pawns_other_side_mine(pos):
    board = chess.Board(pos)
    player_to_move = board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    if len(bishop_squares) != 1:
        return 0
    
    bishop_square = list(bishop_squares)[0]
    bishop_color = ((chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2)
    pawn_squares = board.pieces(chess.PAWN, not player_to_move)
    
    count = 0
    for pawn_square in pawn_squares:
        if (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 != bishop_color:
            count += 1
            
    return count

def num_ocb_pawns_other_side_opponent(pos):
    board = chess.Board(pos)
    player_to_move = not board.turn
    bishop_squares = board.pieces(chess.BISHOP, player_to_move)

    if len(bishop_squares) != 1:
        return 0
    
    bishop_square = list(bishop_squares)[0]
    bishop_color = (chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2 == 0
    pawn_squares = board.pieces(chess.PAWN, not player_to_move)
    
    count = 0
    for pawn_square in pawn_squares:
        if (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 != bishop_color:
            count += 1
            
    return count

def num_ocb_pawns_other_side_diff(pos):
    return num_ocb_pawns_other_side_mine(pos) - num_ocb_pawns_other_side_opponent(pos)

custom_header = [ #"pawn_fork_mine", "pawn_fork_opponent", "knight_fork_mine", "knight_fork_opponent", "bishop_fork_mine", "bishop_fork_opponent", "rook_fork_mine", "rook_fork_opponent",
                  #"has_pinned_pawn_mine", "has_pinned_pawn_opponent", "has_pinned_knight_mine", "has_pinned_knight_opponent", "has_pinned_bishop_mine", 
                  #"has_pinned_bishop_opponent", "has_pinned_rook_mine", "has_pinned_rook_opponent", "has_pinned_queen_mine", "has_pinned_queen_opponent",
                  #"material_mine", "material_opponent", "material_diff", "num_pieces_mine", "num_pieces_opponent", "num_pieces_diff",
                 "in_check", "has_mate_threat", "has_connected_rooks_mine", "has_connected_rooks_opponent",
                 "has_bishop_pair_mine", "has_bishop_pair_opponent", "has_control_of_open_file_mine", "has_control_of_open_file_opponent",
                 "can_capture_queen_mine", "can_capture_queen_opponent", # "num_king_attacked_squares_mine", "num_king_attacked_squares_opponent", "num_king_attacked_squares_diff",
                 "has_contested_open_file", "has_right_bc_ha_promotion_mine", "has_right_bc_ha_promotion_mine", "has_right_bc_ha_promotion_opponent", 
                #  "num_scb_pawns_same_side_mine", "num_scb_pawns_same_side_opponent", "num_scb_pawns_same_side_diff",
                #  "num_ocb_pawns_same_side_mine", "num_ocb_pawns_same_side_opponent", "num_ocb_pawns_same_side_diff",
                #  "num_scb_pawns_other_side_mine", "num_scb_pawns_other_side_opponent", "num_scb_pawns_other_side_diff",
                #  "num_ocb_pawns_other_side_mine", "num_ocb_pawns_other_side_opponent", "num_ocb_pawns_other_side_diff",
                 "capture_possible_on_d1_mine", "capture_possible_on_d2_mine", "capture_possible_on_d3_mine", "capture_possible_on_e1_mine",
                 "capture_possible_on_e2_mine", "capture_possible_on_e3_mine", "capture_possible_on_g5_mine", "capture_possible_on_b5_mine",
                 "capture_possible_on_d1_opponent", "capture_possible_on_d2_opponent", "capture_possible_on_d3_opponent", "capture_possible_on_e1_opponent",
                 "capture_possible_on_e2_opponent", "capture_possible_on_e3_opponent", "capture_possible_on_g5_opponent", "capture_possible_on_b5_opponent",
                #  "capture_happens_next_move_on_d1", "capture_happens_next_move_on_d2", "capture_happens_next_move_on_d3", "capture_happens_next_move_on_e1",
                #  "capture_happens_next_move_on_e2", "capture_happens_next_move_on_e3", "capture_happens_next_move_on_g5", "capture_happens_next_move_on_b5",       
                 # "move_ply", "id", "fen",
                 ]

function_map = {
    "pawn_fork_mine": pawn_fork_mine,
    "pawn_fork_opponent": pawn_fork_opponent,
    "knight_fork_mine": knight_fork_mine,
    "knight_fork_opponent": knight_fork_opponent,
    "bishop_fork_mine": bishop_fork_mine,
    "bishop_fork_opponent": bishop_fork_opponent,
    "rook_fork_mine": rook_fork_mine,
    "rook_fork_opponent": rook_fork_opponent,
    "has_pinned_pawn_mine": has_pinned_pawn_mine, 
    "has_pinned_pawn_opponent": has_pinned_pawn_opponent,
    "has_pinned_knight_mine": has_pinned_knight_mine, 
    "has_pinned_knight_opponent": has_pinned_knight_opponent,
    "has_pinned_bishop_mine": has_pinned_bishop_mine, 
    "has_pinned_bishop_opponent": has_pinned_bishop_opponent,
    "has_pinned_rook_mine": has_pinned_rook_mine, 
    "has_pinned_rook_opponent": has_pinned_rook_opponent,
    "has_pinned_queen_mine": has_pinned_queen_mine, 
    "has_pinned_queen_opponent": has_pinned_queen_opponent,
    "material_mine": material_mine,
    "material_opponent": material_opponent,
    "material_diff": material_diff,
    "num_pieces_mine": num_pieces_mine,
    "num_pieces_opponent": num_pieces_opponent,
    "num_pieces_diff": num_pieces_diff,
    "in_check": in_check,
    "has_mate_threat": has_mate_threat,
    "has_connected_rooks_mine": has_connected_rooks_mine,
    "has_connected_rooks_opponent": has_connected_rooks_opponent,
    "has_bishop_pair_mine": has_bishop_pair_mine,
    "has_bishop_pair_opponent": has_bishop_pair_opponent,
    "has_control_of_open_file_mine": has_control_of_open_file_mine,
    "has_control_of_open_file_opponent": has_control_of_open_file_opponent,
    "can_capture_queen_mine": can_capture_queen_mine,
    "can_capture_queen_opponent": can_capture_queen_opponent,
    "num_king_attacked_squares_mine": num_king_attacked_squares_mine,
    "num_king_attacked_squares_opponent": num_king_attacked_squares_opponent,
    "num_king_attacked_squares_diff": num_king_attacked_squares_diff,
    "has_contested_open_file": has_contested_open_file,
    "capture_happens_next_move_on_d1": capture_happens_next_move_on_d1,
    "capture_happens_next_move_on_d2": capture_happens_next_move_on_d2,
    "capture_happens_next_move_on_d3": capture_happens_next_move_on_d3,
    "capture_happens_next_move_on_e1": capture_happens_next_move_on_e1,
    "capture_happens_next_move_on_e2": capture_happens_next_move_on_e2,
    "capture_happens_next_move_on_e3": capture_happens_next_move_on_e3,
    "capture_happens_next_move_on_b5": capture_happens_next_move_on_b5,
    "capture_happens_next_move_on_g5": capture_happens_next_move_on_g5,
    "has_right_bc_ha_promotion_mine": has_right_bc_ha_promotion_mine,
    "has_right_bc_ha_promotion_opponent": has_right_bc_ha_promotion_opponent,
    "capture_possible_on_d1_mine": capture_possible_on_d1_mine,
    "capture_possible_on_d2_mine": capture_possible_on_d2_mine,
    "capture_possible_on_d3_mine": capture_possible_on_d3_mine,
    "capture_possible_on_e1_mine": capture_possible_on_e1_mine,
    "capture_possible_on_e2_mine": capture_possible_on_e2_mine,
    "capture_possible_on_e3_mine": capture_possible_on_e3_mine,
    "capture_possible_on_b5_mine": capture_possible_on_b5_mine,
    "capture_possible_on_g5_mine": capture_possible_on_g5_mine,
    "capture_possible_on_d1_opponent": capture_possible_on_d1_opponent,
    "capture_possible_on_d2_opponent": capture_possible_on_d2_opponent,
    "capture_possible_on_d3_opponent": capture_possible_on_d3_opponent,
    "capture_possible_on_e1_opponent": capture_possible_on_e1_opponent,
    "capture_possible_on_e2_opponent": capture_possible_on_e2_opponent,
    "capture_possible_on_e3_opponent": capture_possible_on_e3_opponent,
    "capture_possible_on_b5_opponent": capture_possible_on_b5_opponent,
    "capture_possible_on_g5_opponent": capture_possible_on_g5_opponent,
    "num_scb_pawns_same_side_opponent": num_scb_pawns_same_side_opponent,
    "num_scb_pawns_same_side_mine": num_scb_pawns_same_side_mine,
    "num_scb_pawns_same_side_diff": num_scb_pawns_same_side_diff,
    "num_ocb_pawns_same_side_opponent": num_ocb_pawns_same_side_opponent,
    "num_ocb_pawns_same_side_mine": num_ocb_pawns_same_side_mine,
    "num_ocb_pawns_same_side_diff": num_ocb_pawns_same_side_diff,
    "num_scb_pawns_other_side_opponent": num_scb_pawns_other_side_opponent,
    "num_scb_pawns_other_side_mine": num_scb_pawns_other_side_mine,
    "num_scb_pawns_other_side_diff": num_scb_pawns_other_side_diff,
    "num_ocb_pawns_other_side_opponent": num_ocb_pawns_other_side_opponent,
    "num_ocb_pawns_other_side_mine": num_ocb_pawns_other_side_mine,
    "num_ocb_pawns_other_side_diff": num_ocb_pawns_other_side_diff,
}

def main() -> None:
    # fen = "3qk2r/p2ppppp/5b2/8/7P/1PQ1PP2/2P1P1P1/RN1B1KNR w KQkq - 1 1"
    # print(num_scb_pawns_other_side_mine(fen))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )    
    parser.add_argument("--dataset_elo", type=int, help="Elo rate for matches")
    parser.add_argument(
        "--skip_rows",
        type=int,
        default=None,
        help="Number of rows to skip",
    )
    parser.add_argument(
        "--max_positions",
        type=int,
        default=None,
        help="Maximum number of positions to use",
    )
    args = parser.parse_args()
    dataset = args.dataset_elo

    dataset_path = f"flipped_fen.csv"
    output_file_path = f"stockfish_concepts/custom_concepts.csv"

    index = 0
    with open(output_file_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, custom_header)
        writer.writeheader()
        #games = maia_lib.GamesFile(dataset_path)
        
        fen_df = pd.read_csv(dataset_path)
        for _, row in tqdm.tqdm(fen_df.iterrows(), total=fen_df.shape[0]):  
            # features = calculate_features(b.fen())
            
            fen = row['board']
            rowDict = {}
            #move = dat.move
            for header in custom_header:
                if header in function_map:
                    function = function_map[header]
                    num_params = len(inspect.signature(function).parameters)

                    # For evaluating concepts that require only 1 fen
                    if num_params == 1:
                        rowDict[header] = function(fen)
                    # For evaluating concepts that require 1 fen and 1 move
                    if num_params == 2:
                        rowDict[header] = function(fen, move)
                    
            #rowDict['id'] = dat.game_id
            #rowDict['move_ply'] = dat.move_ply
            rowDict['fen'] = fen
            writer.writerow(rowDict)
        
    #df = pd.read_csv(output_file_path)
    #df.to_csv(output_file_path, index=True)

if __name__ == "__main__":
    main()