import pandas as pd
import chess
from ..maia2.utils import *
from ..maia2.main import *
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
import random
import threading
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
import glob
import pdb
from typing import Dict, Union

# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

_thread_local = threading.local()

def extract_json_data(json_line):
    data = json.loads(json_line)
    fen = data['fen']
    best_move = data['evals'][0]['pvs'][0]['line'].split()[0]
    best_cp = data['evals'][0]['pvs'][0].get('cp', 100000)

    best_moves = []

    pvs = []
    for eval in data['evals']:
        for pv in eval['pvs']:
            pvs.append(pv)

    for pv in pvs:
        if abs(pv.get('cp', 0) - best_cp) < 30:
            best_moves.append(pv['line'].split()[0])

    elo_dict = {
        '<1100': 0, '1100-1199': 1, '1200-1299': 2, '1300-1399': 3, '1400-1499': 4,
        '1500-1599': 5, '1600-1699': 6, '1700-1799': 7, '1800-1899': 8, '1900-1999': 9,
        '>=2000': 10
    }

    maia_data = {}
    for elo_range, category in elo_dict.items():
        if elo_range in data:
            move_probs = data[elo_range]['move_probs']
            top_maia_move = max(move_probs, key=move_probs.get)
            best_move_prob = move_probs.get(best_move, 0)

            maia_data[category] = {
                'top_maia_move': top_maia_move,
                'best_move_prob': best_move_prob
            }

    return {
        'fen': fen,
        'best_move': best_move,
        'maia_data': maia_data,
        'best_moves': list(set(best_moves))
    }


def is_monotonic(maia_data: Dict[int, Dict[str, Any]]) -> bool:
    probs = [data['best_move_prob'] for data in maia_data.values()]
    return all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1))


def is_transitional(maia_data: Dict[int, Dict[str, Any]], best_moves: List[str]) -> Tuple[bool, int]:
    moves = [data['top_maia_move'] for data in maia_data.values()]
    elo_categories = sorted(maia_data.keys())

    # Check if all moves are correct (not truly transitional)
    if all(move in best_moves for move in moves):
        return False, -1

    for x in range(1, len(elo_categories)):
        if all(moves[i] not in best_moves for i in range(x)) and \
                all(moves[i] in best_moves for i in range(x, len(moves))):
            return True, elo_categories[x]
    return False, -1


def analyze_positions(puzzle_data: Dict[str, Any]) -> Dict[str, Any]:
    fen = puzzle_data['fen']
    best_move = puzzle_data['best_move']
    maia_data = puzzle_data['maia_data']
    best_moves = puzzle_data['best_moves']

    monotonic = is_monotonic(maia_data)
    transitional, transition_point = is_transitional(maia_data, best_moves)
    all_correct = all(data['top_maia_move'] == best_move for data in maia_data.values())

    return {
        'fen': fen,
        'best_move': best_move,
        'maia_moves': {elo: data['top_maia_move'] for elo, data in maia_data.items()},
        'maia_probs': {elo: data['best_move_prob'] for elo, data in maia_data.items()},
        'is_monotonic': monotonic,
        'is_transitional': transitional,
        'transition_point': transition_point,
        'is_both': monotonic and transitional,
        'all_correct': all_correct,
    }

def process_positions(file_path, special_fens, best_transitional_moves, all_best_transitional_moves,
                    transitional_maia_moves, counters, transition_points) -> tuple:
    with open(file_path, 'r') as file:
        for line in file:
            puzzle_data = extract_json_data(line)
            analyzed_data = analyze_positions(puzzle_data)
            # results.append(analyzed_data)

            counters['total'] += 1
            if analyzed_data['is_monotonic']:
                counters['monotonic'] += 1
                special_fens['monotonic'].append(analyzed_data['fen'])
            if analyzed_data['is_transitional']:
                counters['transitional'] += 1
                special_fens['transitional'].append(analyzed_data['fen'])
                transition_points.append(analyzed_data['transition_point'])
                best_transitional_moves.append(puzzle_data['best_move'])
                all_best_transitional_moves.append(puzzle_data['best_moves'])
                transitional_maia_moves.append(analyzed_data['maia_moves'])
            if analyzed_data['is_both']:
                counters['both'] += 1
                special_fens['both'].append(analyzed_data['fen'])
            if analyzed_data['all_correct']:
                counters['all_correct'] += 1

    return special_fens, counters, transition_points, best_transitional_moves, transitional_maia_moves

def is_piece_no_longer_under_attack(fen: str, move: str, square_index) -> bool:
    board = chess.Board(fen)
    piece_before_move = board.piece_at(square_index)

    def is_under_attack(board, square_index):
        attackers = board.attackers(chess.BLACK, square_index) 
        defenders = board.attackers(chess.WHITE, square_index)  
        return len(attackers) > len(defenders)

    was_under_attack_before = is_under_attack(board, square_index) and piece_before_move is not None and piece_before_move.color is not chess.BLACK
    move_obj = chess.Move.from_uci(move)
    if not board.is_legal(move_obj):
        raise ValueError(f"Illegal move: {move}")

    board.push(move_obj)
    piece_after_move = board.piece_at(square_index)
    is_under_attack_after = is_under_attack(board, square_index) and piece_after_move is not None

    return was_under_attack_before and not is_under_attack_after

def square_index(square_name: str) -> int:
    if not isinstance(square_name, str) or len(square_name) != 2:
        raise ValueError("Square name must be a string of length 2")

    file = square_name[0].lower()
    rank = square_name[1]

    if file not in 'abcdefgh' or rank not in '12345678':
        raise ValueError("Invalid square name")

    file_index = 'abcdefgh'.index(file)
    rank_index = '12345678'.index(rank)

    return rank_index * 8 + file_index

def is_blunder(args):
    fen, uci_move, stockfish_path, threshold = args
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves:
                return False
            info_before = engine.analyse(board, chess.engine.Limit(depth=18))
            eval_before = info_before['score'].relative.score()
            if eval_before is None:
                return False
            board.push(move)
            info_after = engine.analyse(board, chess.engine.Limit(depth=18))
            eval_after = info_after['score'].relative.score()
            if eval_after is None:
                return False
            centipawn_loss = abs(eval_before - eval_after)
            return centipawn_loss > threshold
    except Exception as e:
        logging.error(f"Error processing FEN {fen} with move {uci_move}: {str(e)}")
        return None

# helper functions for intervention

# hooks for clear run at residual streams
def _enable_activation_hook(model, cfg):
    def get_activation(name):
        def hook(model, input, output):
            if not hasattr(_thread_local, 'residual_streams'):
                _thread_local.residual_streams = {}
            _thread_local.residual_streams[name] = output.detach()
        return hook
        
    for i in range(cfg.num_blocks_vit):
        feedforward_module = model.module.transformer.elo_layers[i][1]
        feedforward_module.register_forward_hook(get_activation(f'transformer block {i} hidden states'))

def apply_sae_to_activations(sae, activations, target_key_list):
    sae_activations = {}
    for key in target_key_list:
        if key in activations and key in sae:
            # act = activations[key].view(-1, activations[key].size(-1))
            act = torch.mean(activations[key], dim=1)
            # print(act.shape)
            encoded = nn.functional.linear(act, sae[key]['encoder_DF.weight'], sae[key]['encoder_DF.bias'])
            encoded = nn.functional.relu(encoded)
            
            sae_activations[key] = encoded
    
    return sae_activations

def apply_sae_to_reconstruction(sae, activations, target_key_list):
    sae_activations = {}
    for key in target_key_list:
        if key in activations and key in sae:
            act = torch.mean(activations[key], dim=1)
            encoded = nn.functional.linear(act, sae[key]['encoder_DF.weight'], sae[key]['encoder_DF.bias'])
            encoded = nn.functional.relu(encoded)
            decoded = nn.functional.linear(encoded, sae[key]['decoder_FD.weight'], sae[key]['decoder_FD.bias'])
            
            sae_activations[key] = decoded
    
    return sae_activations

def get_legal_moves_idx(board, all_moves_dict):
    legal_moves = torch.zeros(len(all_moves_dict))
    legal_moves_idx = []
    for move in board.legal_moves:
        move_uci = move.uci()
        if move_uci in all_moves_dict:
            legal_moves_idx.append(all_moves_dict[move_uci])
    legal_moves_idx = torch.tensor(legal_moves_idx)
    legal_moves[legal_moves_idx] = 1
    return legal_moves

# hooks for patched run at residual streams

def _enable_intervention_hook(model, cfg):
    def get_intervention_hook(name):
        def hook(module, input, output):
            if not hasattr(_thread_local, 'residual_streams'):
                _thread_local.residual_streams = {}
            _thread_local.residual_streams[name] = output.detach()
            if hasattr(_thread_local, 'modified_values') and name in _thread_local.modified_values:
                return _thread_local.modified_values[name]
            return None
        return hook
    
    for i in range(cfg.num_blocks_vit):
        feedforward_module = model.module.transformer.elo_layers[i][1]
        feedforward_module.register_forward_hook(
            get_intervention_hook(f'transformer block {i} hidden states')
        )

def set_modified_values(modified_dict):
    _thread_local.modified_values = modified_dict

def clear_modified_values():
    if hasattr(_thread_local, 'modified_values'):
        _thread_local.modified_values.clear()

# vanilla intervention pipeline

def mediated_intervention(cfg, layer_one_features, elo_range, candidate_strength, intervention_site, filtered_all_ground_truths, filtered_special_fens, filtered_all_best_transitional_moves, filtered_transition_points, board_inputs, sae, model, move_dict, all_moves_dict):
    target_key_list = ['transformer block 1 hidden states']
    elo_dict = create_elo_dict()
    original_results_dict = {
        k: {
            strength: {elo: [0, 0] for elo in elo_range}
            for strength in candidate_strength
        }
        for k in [key for key, value in layer_one_features.items()]
    }
    intervened_results_dict = {
        k: {
            strength: {elo: [0, 0] for elo in elo_range}
            for strength in candidate_strength
        }
        for k in [key for key, value in layer_one_features.items()]
    }
    transition_points_deviation = {k:{strength: {} for strength in candidate_strength} for k in [key for key, value in layer_one_features.items()]}
    start_time = time.time()

    for specific_square_name in layer_one_features.keys():
        
        specific_square_idx_gt = 0
        for key, value in layer_one_features.items():
            if key == specific_square_name:
                break
            specific_square_idx_gt += 1
        
        # print(specific_square_name, torch.sum(filtered_all_ground_truths, dim=1)[specific_square_idx_gt])
        if torch.sum(filtered_all_ground_truths, dim=1)[specific_square_idx_gt] < 20: # not enough samples
            continue
        
        intervened_pred_list = []
        
        legal_moves_list = []
        for fen in filtered_special_fens:
            board = chess.Board(fen)
            legal_moves_list.append(get_legal_moves_idx(board, all_moves_dict))
        legal_moves = torch.stack(legal_moves_list)
        
        column_indexes = torch.where(filtered_all_ground_truths[specific_square_idx_gt] == 1)[0]
        target_boards = board_inputs[column_indexes, :, :, :]
        target_legal_moves = legal_moves[column_indexes, :]
        target_all_best_moves = [filtered_all_best_transitional_moves[i] for i in column_indexes.tolist()]
        target_transition_points = [filtered_transition_points[i] for i in column_indexes.tolist()]

        for intervention_strength in candidate_strength:
            elos_self = torch.zeros(len(target_boards))
            elos_oppo = torch.zeros(len(target_boards))
            epsilon = 0.005
            intervened_pred_list = []
            
            for elo in elo_range:
            
                elos_self = elos_self.fill_(elo).long()
                elos_oppo = elos_oppo.fill_(elo).long()
            
                # clean run
                _enable_activation_hook(model, cfg)
                with torch.no_grad():
                    logits_maia, logits_side_info, logits_value = model(target_boards, elos_self, elos_oppo)
                    activations = getattr(_thread_local, 'residual_streams', {})
                    sae_activations = apply_sae_to_activations(sae, activations, target_key_list)
                    sae_reconstruct_activations = apply_sae_to_reconstruction(sae, activations, target_key_list)
                    logits_maia_legal = logits_maia * target_legal_moves
                    preds = logits_maia_legal.argmax(dim=-1)
            
                # mediated intervention
                intervened_sae_activations = {}
                for key in sae_activations:
                    intervened_sae_activations[key] = sae_activations[key].clone()
                
                layer, feature_idx = intervention_site[specific_square_idx_gt]
                intervened_sae_activations[layer][:, feature_idx] += epsilon
                intervened_sae_activations[layer][:, feature_idx] *= intervention_strength
                    
                reconstructed_activations = {}
                for key in intervened_sae_activations:
                    reconstructed_activations[key] = nn.functional.linear(intervened_sae_activations[key], sae[key]['decoder_FD.weight'], 
                                                                          sae[key]['decoder_FD.bias']).unsqueeze(1).expand(-1, 8, -1)
            
                _enable_intervention_hook(model, cfg)
                set_modified_values(reconstructed_activations)
                with torch.no_grad():
                    intervened_logits_maia, intervened_logits_side_info, intervened_logits_value = model(target_boards, elos_self, elos_oppo)
                    intervened_logits_maia_legal = intervened_logits_maia * target_legal_moves
                    intervened_preds = intervened_logits_maia_legal.argmax(dim=-1)
                intervened_pred_list.append(intervened_preds)
                clear_modified_values()
                
                # Best move rate

                original_cnt = 0
                denom = 0
                for i in range(target_boards.shape[0]):
                    pred = move_dict[preds[i].item()]
                    if (not cfg.mistakes_only or target_transition_points[i] > elo):
                        denom += 1
                        if pred in target_all_best_moves[i]:
                            original_cnt += 1
                            if elo:
                                original_results_dict[specific_square_name][intervention_strength][elo][1] += 1
                        else:
                            original_results_dict[specific_square_name][intervention_strength][elo][0] += 1
                                
                # original_cnt /= target_boards.shape[0]
                original_cnt /= denom
                # print(f"Original rate for predicting the best {specific_square_name} move: {original_results_dict[specific_square_name][intervention_strength][elo][1]/(original_results_dict[specific_square_name][intervention_strength][elo][0]+original_results_dict[specific_square_name][intervention_strength][elo][1])}")
            
                intervened_cnt = 0
                for i in range(target_boards.shape[0]):
                    pred = move_dict[intervened_preds[i].item()]
                    if (not cfg.mistakes_only or target_transition_points[i] > elo):
                        if pred in target_all_best_moves[i]:
                            intervened_cnt += 1
                            intervened_results_dict[specific_square_name][intervention_strength][elo][1] += 1
                        else:
                            intervened_results_dict[specific_square_name][intervention_strength][elo][0] += 1
                #intervened_cnt /= target_boards.shape[1]
                intervened_cnt /= denom
                # print(f"Intervened rate for predicting the best {specific_square_name} move:{intervened_results_dict[specific_square_name][intervention_strength][elo][1]/(intervened_results_dict[specific_square_name][intervention_strength][elo][0]+intervened_results_dict[specific_square_name][intervention_strength][elo][1])}")
            
            intervened_pred_list = torch.stack(intervened_pred_list, dim=0)
            intervened_transition_points = []
            for i in range(intervened_pred_list.shape[1]):
                intervened = False
                for j in range(1, len(elo_dict) - 1):
                    if all(move_dict[intervened_pred_list[k][i].item()] not in target_all_best_moves[i] for k in range(j)) and \
                       all(move_dict[intervened_pred_list[k][i].item()] in target_all_best_moves[i] for k in range(j, len(elo_dict) - 1)):
                           intervened_transition_points.append(j)
                           intervened = True
                if not intervened:
                    intervened_transition_points.append(-1)
            
            cnt = 0
            tot_sum = 0
            for i in range(intervened_pred_list.shape[1]):
                if intervened_transition_points[i] != -1:
                    cnt += 1
                    tot_sum += (target_transition_points[i] - intervened_transition_points[i])
            
            if cnt:
                deviation = tot_sum/cnt
            else:
                deviation = 0
            transition_points_deviation[specific_square_name][intervention_strength] = deviation

    end_time = time.time()
    print(f"Total time for intervention run: {(end_time - start_time):.2f}s")
    return original_results_dict, intervened_results_dict, transition_points_deviation

# activation-patching style intervention pipeline

def patching_style_intervention():
    pass

# random intervention pipeline

def random_intervention(cfg, layer_one_features, elo_range, candidate_strength, intervention_site, filtered_all_ground_truths, filtered_special_fens, filtered_all_best_transitional_moves, filtered_transition_points, board_inputs, sae, model, move_dict, all_moves_dict):
    target_key_list = ['transformer block 1 hidden states']
    original_results_dict = {
        k: {
            strength: {elo: [0, 0] for elo in elo_range}
            for strength in candidate_strength
        }
        for k in [key for key, value in layer_one_features.items()]
    }
    random_results_dict = {
        k: {
            strength: {elo: [0, 0] for elo in elo_range}
            for strength in candidate_strength
        }
        for k in [key for key, value in layer_one_features.items()]
    }
    start_time = time.time()

    for specific_square_name in layer_one_features.keys():
        
        specific_square_idx_gt = 0
        for key, value in layer_one_features.items():
            if key == specific_square_name:
                break
            specific_square_idx_gt += 1
        
        if torch.sum(filtered_all_ground_truths, dim=1)[specific_square_idx_gt] < 20: # not enough samples
            continue
        
        legal_moves_list = []
        for fen in filtered_special_fens:
            board = chess.Board(fen)
            legal_moves_list.append(get_legal_moves_idx(board, all_moves_dict))
        legal_moves = torch.stack(legal_moves_list)
        
        column_indexes = torch.where(filtered_all_ground_truths[specific_square_idx_gt] == 1)[0]
        target_boards = board_inputs[column_indexes, :, :, :]
        target_legal_moves = legal_moves[column_indexes, :]
        target_all_best_moves = [filtered_all_best_transitional_moves[i] for i in column_indexes.tolist()]
        target_transition_points = [filtered_transition_points[i] for i in column_indexes.tolist()]

        for intervention_strength in candidate_strength:
            elos_self = torch.zeros(len(target_boards))
            elos_oppo = torch.zeros(len(target_boards))
            
            for elo in elo_range:
            
                elos_self = elos_self.fill_(elo).long()
                elos_oppo = elos_oppo.fill_(elo).long()
            
                # clean run
                _enable_activation_hook(model, cfg)
                with torch.no_grad():
                    logits_maia, logits_side_info, logits_value = model(target_boards, elos_self, elos_oppo)
                    activations = getattr(_thread_local, 'residual_streams', {})
                    sae_activations = apply_sae_to_activations(sae, activations, target_key_list)
                    sae_reconstruct_activations = apply_sae_to_reconstruction(sae, activations, target_key_list)
                    logits_maia_legal = logits_maia * target_legal_moves
                    preds = logits_maia_legal.argmax(dim=-1)
            
                # random intervention
                random_sae_activations = {}
                for key in sae_activations:
                    random_sae_activations[key] = sae_activations[key].clone()
                
                layer, feature_idx = intervention_site[specific_square_idx_gt]
                feature_idx = random.randint(0, 2047)
                # random_sae_activations[layer][:, feature_idx] += epsilon
                random_sae_activations[layer][:, feature_idx] *= intervention_strength

                random_intervened_activations = {}
                for key in random_sae_activations:
                    random_intervened_activations[key] = nn.functional.linear(random_sae_activations[key], sae[key]['decoder_FD.weight'], 
                                                                          sae[key]['decoder_FD.bias']).unsqueeze(1).expand(-1, 8, -1)
            
                _enable_intervention_hook(model, cfg)
                set_modified_values(random_intervened_activations)
                with torch.no_grad():
                    random_logits_maia, random_logits_side_info, random_logits_value = model(target_boards, elos_self, elos_oppo)
                    random_logits_maia_legal = random_logits_maia * target_legal_moves
                    random_preds = random_logits_maia_legal.argmax(dim=-1)
                clear_modified_values()
                
                # Best move rate

                original_cnt = 0
                denom = 0
                for i in range(target_boards.shape[0]):
                    pred = move_dict[preds[i].item()]
                    if (not cfg.mistakes_only or target_transition_points[i] > elo):
                        denom += 1
                        if pred in target_all_best_moves[i]:
                            original_cnt += 1
                            if elo:
                                original_results_dict[specific_square_name][intervention_strength][elo][1] += 1
                        else:
                            original_results_dict[specific_square_name][intervention_strength][elo][0] += 1
                                
                original_cnt /= denom
                # print(f"Original rate for predicting the best {specific_square_name} move: {original_results_dict[specific_square_name][intervention_strength][elo][1]/(original_results_dict[specific_square_name][intervention_strength][elo][0]+original_results_dict[specific_square_name][intervention_strength][elo][1])}")
            
                random_cnt = 0
                for i in range(target_boards.shape[0]):
                    pred = move_dict[random_preds[i].item()]
                    if (not cfg.mistakes_only or target_transition_points[i] > elo):
                        if pred in target_all_best_moves[i]:
                            random_cnt += 1
                            random_results_dict[specific_square_name][intervention_strength][elo][1] += 1
                        else:
                            random_results_dict[specific_square_name][intervention_strength][elo][0] += 1
                    random_cnt /= denom

    end_time = time.time()
    print(f"Total time for intervention run: {(end_time - start_time):.2f}s")
    return original_results_dict, random_results_dict


#TODO define linear probe and SV interventions

def linear_probe_intervention(cfg, layer_one_features, elo_range, candidate_strength, intervention_site, filtered_all_ground_truths, filtered_special_fens, filtered_all_best_transitional_moves, filtered_transition_points, board_inputs, sae, model, move_dict, all_moves_dict, probe_weights):
    
    target_key_list = ['transformer block 1 hidden states']
    original_results_dict = {
        k: {
            strength: {elo: [0, 0] for elo in elo_range}
            for strength in candidate_strength
        }
        for k in [key for key, value in layer_one_features.items()]
    }
    intervened_results_dict = {
        k: {
            strength: {elo: [0, 0] for elo in elo_range}
            for strength in candidate_strength
        }
        for k in [key for key, value in layer_one_features.items()]
    }

    transition_points_deviation = {k:{strength: {} for strength in candidate_strength} for k in [key for key, value in layer_one_features.items()]}
    start_time = time.time()

    for specific_square_name in layer_one_features.keys():
        
        specific_square_idx_gt = 0
        for key, value in layer_one_features.items():
            if key == specific_square_name:
                break
            specific_square_idx_gt += 1
        
        if torch.sum(filtered_all_ground_truths, dim=1)[specific_square_idx_gt] < 20: # not enough samples
            continue
        
        legal_moves_list = []
        for fen in filtered_special_fens:
            board = chess.Board(fen)
            legal_moves_list.append(get_legal_moves_idx(board, all_moves_dict))
        legal_moves = torch.stack(legal_moves_list)
        
        column_indexes = torch.where(filtered_all_ground_truths[specific_square_idx_gt] == 1)[0]
        target_boards = board_inputs[column_indexes, :, :, :]
        target_legal_moves = legal_moves[column_indexes, :]
        target_all_best_moves = [filtered_all_best_transitional_moves[i] for i in column_indexes.tolist()]
        target_transition_points = [filtered_transition_points[i] for i in column_indexes.tolist()]

        for intervention_strength in candidate_strength:
            elos_self = torch.zeros(len(target_boards))
            elos_oppo = torch.zeros(len(target_boards))
            
            for elo in elo_range:
            
                elos_self = elos_self.fill_(elo).long()
                elos_oppo = elos_oppo.fill_(elo).long()
            
                # clean run
                _enable_activation_hook(model, cfg)
                with torch.no_grad():
                    logits_maia, logits_side_info, logits_value = model(target_boards, elos_self, elos_oppo)
                    activations = getattr(_thread_local, 'residual_streams', {})
                    sae_activations = apply_sae_to_activations(sae, activations, target_key_list)
                    sae_reconstruct_activations = apply_sae_to_reconstruction(sae, activations, target_key_list)
                    logits_maia_legal = logits_maia * target_legal_moves
                    preds = logits_maia_legal.argmax(dim=-1)

                intervened_internal_activations = {}
                for key in activations:
                    intervened_internal_activations[key] = activations[key]
                                
                for i in range(target_boards.shape[0]):
                    for layer in target_key_list:
                        intervened_internal_activations[layer][i] += intervention_strength * probe_weights[specific_square_name]

                _enable_intervention_hook(model, cfg)
                set_modified_values(intervened_internal_activations)
                with torch.no_grad():
                    intervened_logits_maia, intervened_logits_side_info, intervened_logits_value = model(target_boards, elos_self, elos_oppo)
                    intervened_logits_maia_legal = intervened_logits_maia * legal_moves
                    intervened_preds = intervened_logits_maia_legal.argmax(dim=-1)
                clear_modified_values()

                                # Best move rate

                original_cnt = 0
                denom = 0
                for i in range(target_boards.shape[0]):
                    pred = move_dict[preds[i].item()]
                    if (not cfg.mistakes_only or target_transition_points[i] > elo):
                        denom += 1
                        if pred in target_all_best_moves[i]:
                            original_cnt += 1
                            if elo:
                                original_results_dict[specific_square_name][intervention_strength][elo][1] += 1
                        else:
                            original_results_dict[specific_square_name][intervention_strength][elo][0] += 1
                                
                # original_cnt /= target_boards.shape[0]
                original_cnt /= denom
                # print(f"Original rate for predicting the best {specific_square_name} move: {original_results_dict[specific_square_name][intervention_strength][elo][1]/(original_results_dict[specific_square_name][intervention_strength][elo][0]+original_results_dict[specific_square_name][intervention_strength][elo][1])}")
            
                intervened_cnt = 0
                for i in range(target_boards.shape[0]):
                    pred = move_dict[intervened_preds[i].item()]
                    if (not cfg.mistakes_only or target_transition_points[i] > elo):
                        if pred in target_all_best_moves[i]:
                            intervened_cnt += 1
                            intervened_results_dict[specific_square_name][intervention_strength][elo][1] += 1
                        else:
                            intervened_results_dict[specific_square_name][intervention_strength][elo][0] += 1
                #intervened_cnt /= target_boards.shape[1]
                intervened_cnt /= denom
                # print(f"Intervened rate for predicting the best {specific_square_name} move:{intervened_results_dict[specific_square_name][intervention_strength][elo][1]/(intervened_results_dict[specific_square_name][intervention_strength][elo][0]+intervened_results_dict[specific_square_name][intervention_strength][elo][1])}")
            
            intervened_pred_list = torch.stack(intervened_pred_list, dim=0)
            intervened_transition_points = []
            for i in range(intervened_pred_list.shape[1]):
                intervened = False
                for j in range(1, len(elo_dict) - 1):
                    if all(move_dict[intervened_pred_list[k][i].item()] not in target_all_best_moves[i] for k in range(j)) and \
                       all(move_dict[intervened_pred_list[k][i].item()] in target_all_best_moves[i] for k in range(j, len(elo_dict) - 1)):
                           intervened_transition_points.append(j)
                           intervened = True
                if not intervened:
                    intervened_transition_points.append(-1)
            
            cnt = 0
            tot_sum = 0
            for i in range(intervened_pred_list.shape[1]):
                if intervened_transition_points[i] != -1:
                    cnt += 1
                    tot_sum += (target_transition_points[i] - intervened_transition_points[i])
            
            if cnt:
                deviation = tot_sum/cnt
            else:
                deviation = 0
            transition_points_deviation[specific_square_name][intervention_strength] = deviation

    end_time = time.time()
    print(f"Total time for intervention run: {(end_time - start_time):.2f}s")
    return original_results_dict, intervened_results_dict, transition_points_deviation


def steering_intervention(cfg, layer_one_features, elo_range, candidate_strength, intervention_site, filtered_all_ground_truths, filtered_special_fens, filtered_all_best_transitional_moves, filtered_transition_points, board_inputs, sae, model, move_dict, all_moves_dict, steering_vectors):
    
    target_key_list = ['transformer block 1 hidden states']
    original_results_dict = {
        k: {
            strength: {elo: [0, 0] for elo in elo_range}
            for strength in candidate_strength
        }
        for k in [key for key, value in layer_one_features.items()]
    }
    intervened_results_dict = {
        k: {
            strength: {elo: [0, 0] for elo in elo_range}
            for strength in candidate_strength
        }
        for k in [key for key, value in layer_one_features.items()]
    }
    
    transition_points_deviation = {k:{strength: {} for strength in candidate_strength} for k in [key for key, value in layer_one_features.items()]}
    start_time = time.time()
    for specific_square_name in layer_one_features.keys():
        
        specific_square_idx_gt = 0
        for key, value in layer_one_features.items():
            if key == specific_square_name:
                break
            specific_square_idx_gt += 1
        
        if torch.sum(filtered_all_ground_truths, dim=1)[specific_square_idx_gt] < 20: # not enough samples
            continue
        
        legal_moves_list = []
        for fen in filtered_special_fens:
            board = chess.Board(fen)
            legal_moves_list.append(get_legal_moves_idx(board, all_moves_dict))
        legal_moves = torch.stack(legal_moves_list)
        
        column_indexes = torch.where(filtered_all_ground_truths[specific_square_idx_gt] == 1)[0]
        target_boards = board_inputs[column_indexes, :, :, :]
        target_legal_moves = legal_moves[column_indexes, :]
        target_all_best_moves = [filtered_all_best_transitional_moves[i] for i in column_indexes.tolist()]
        target_transition_points = [filtered_transition_points[i] for i in column_indexes.tolist()]

        for intervention_strength in candidate_strength:
            elos_self = torch.zeros(len(target_boards))
            elos_oppo = torch.zeros(len(target_boards))
            
            for elo in elo_range:
            
                elos_self = elos_self.fill_(elo).long()
                elos_oppo = elos_oppo.fill_(elo).long()
            
                # clean run
                _enable_activation_hook(model, cfg)
                with torch.no_grad():
                    logits_maia, logits_side_info, logits_value = model(target_boards, elos_self, elos_oppo)
                    activations = getattr(_thread_local, 'residual_streams', {})
                    sae_activations = apply_sae_to_activations(sae, activations, target_key_list)
                    sae_reconstruct_activations = apply_sae_to_reconstruction(sae, activations, target_key_list)
                    logits_maia_legal = logits_maia * target_legal_moves
                    preds = logits_maia_legal.argmax(dim=-1)

                intervened_internal_activations = {}
                for key in activations:
                    intervened_internal_activations[key] = activations[key]
                                
                for i in range(target_boards.shape[0]):
                    for layer in target_key_list:
                        intervened_internal_activations[layer][i] += intervention_strength * steering_vectors[specific_square_name]

                _enable_intervention_hook(model, cfg)
                set_modified_values(intervened_internal_activations)
                with torch.no_grad():
                    intervened_logits_maia, intervened_logits_side_info, intervened_logits_value = model(target_boards, elos_self, elos_oppo)
                    intervened_logits_maia_legal = intervened_logits_maia * legal_moves
                    intervened_preds = intervened_logits_maia_legal.argmax(dim=-1)
                clear_modified_values()

                                # Best move rate

                original_cnt = 0
                denom = 0
                for i in range(target_boards.shape[0]):
                    pred = move_dict[preds[i].item()]
                    if (not cfg.mistakes_only or target_transition_points[i] > elo):
                        denom += 1
                        if pred in target_all_best_moves[i]:
                            original_cnt += 1
                            if elo:
                                original_results_dict[specific_square_name][intervention_strength][elo][1] += 1
                        else:
                            original_results_dict[specific_square_name][intervention_strength][elo][0] += 1
                                
                # original_cnt /= target_boards.shape[0]
                original_cnt /= denom
                # print(f"Original rate for predicting the best {specific_square_name} move: {original_results_dict[specific_square_name][intervention_strength][elo][1]/(original_results_dict[specific_square_name][intervention_strength][elo][0]+original_results_dict[specific_square_name][intervention_strength][elo][1])}")
            
                intervened_cnt = 0
                for i in range(target_boards.shape[0]):
                    pred = move_dict[intervened_preds[i].item()]
                    if (not cfg.mistakes_only or target_transition_points[i] > elo):
                        if pred in target_all_best_moves[i]:
                            intervened_cnt += 1
                            intervened_results_dict[specific_square_name][intervention_strength][elo][1] += 1
                        else:
                            intervened_results_dict[specific_square_name][intervention_strength][elo][0] += 1
                #intervened_cnt /= target_boards.shape[1]
                intervened_cnt /= denom
                # print(f"Intervened rate for predicting the best {specific_square_name} move:{intervened_results_dict[specific_square_name][intervention_strength][elo][1]/(intervened_results_dict[specific_square_name][intervention_strength][elo][0]+intervened_results_dict[specific_square_name][intervention_strength][elo][1])}")
            
            intervened_pred_list = torch.stack(intervened_pred_list, dim=0)
            intervened_transition_points = []
            for i in range(intervened_pred_list.shape[1]):
                intervened = False
                for j in range(1, len(elo_dict) - 1):
                    if all(move_dict[intervened_pred_list[k][i].item()] not in target_all_best_moves[i] for k in range(j)) and \
                       all(move_dict[intervened_pred_list[k][i].item()] in target_all_best_moves[i] for k in range(j, len(elo_dict) - 1)):
                           intervened_transition_points.append(j)
                           intervened = True
                if not intervened:
                    intervened_transition_points.append(-1)
            
            cnt = 0
            tot_sum = 0
            for i in range(intervened_pred_list.shape[1]):
                if intervened_transition_points[i] != -1:
                    cnt += 1
                    tot_sum += (target_transition_points[i] - intervened_transition_points[i])
            
            if cnt:
                deviation = tot_sum/cnt
            else:
                deviation = 0
            transition_points_deviation[specific_square_name][intervention_strength] = deviation

    end_time = time.time()
    print(f"Total time for intervention run: {(end_time - start_time):.2f}s")
    return original_results_dict, intervened_results_dict, transition_points_deviation


def find_best_intervention(original_results_dict, intervened_results_dict, elo_range):
    best_accuracies = {}
    original_accuracies = {}
    
    for square in original_results_dict.keys():
        best_accuracies[square] = {}
        original_accuracies[square] = {}
        
        for elo in elo_range:
            best_acc = None
            original_acc = None
            
            for strength in original_results_dict[square].keys():
                if sum(original_results_dict[square][strength][elo]) and sum(intervened_results_dict[square][strength][elo]):
                    current_original_acc = original_results_dict[square][strength][elo][1] / sum(original_results_dict[square][strength][elo])
                    current_intervened_acc = intervened_results_dict[square][strength][elo][1] / sum(intervened_results_dict[square][strength][elo])
                    
                    if best_acc is None or current_intervened_acc > best_acc:
                        best_acc = current_intervened_acc
                    
                    if original_acc is None:
                        original_acc = current_original_acc
            
            if best_acc is not None:
                best_accuracies[square][elo] = best_acc
            if original_acc is not None:
                original_accuracies[square][elo] = original_acc
    return original_accuracies, best_accuracies

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # Supporting Arguments
    parser.add_argument('--data_root', default='maia2_sae/pgn', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--max_ply', default=300, type=int)
    parser.add_argument('--clock_threshold', default=30, type=int)
    parser.add_argument('--chunk_size', default=8000, type=int)
    parser.add_argument('--start_year', default=2013, type=int)
    parser.add_argument('--start_month', default=1, type=int)
    parser.add_argument('--end_year', default=2019, type=int)
    parser.add_argument('--end_month', default=12, type=int)
    parser.add_argument('--from_checkpoint', default=False, type=bool)
    parser.add_argument('--checkpoint_year', default=2018, type=int)
    parser.add_argument('--checkpoint_month', default=12, type=int)
    parser.add_argument('--test_year', default=2024, type=int)
    parser.add_argument('--test_month', default=1, type=int)
    parser.add_argument('--num_cpu_left', default=4, type=int)
    parser.add_argument('--model', default='ViT', type=str)
    parser.add_argument('--max_games_per_elo_range', default=20, type=int)

    # Tunable Arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=30000, type=int)
    parser.add_argument('--first_n_moves', default=10, type=int)
    parser.add_argument('--last_n_moves', default=10, type=int)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=2, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=8, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    parser.add_argument('--side_info', default=True, type=bool)
    parser.add_argument('--value', default=True, type=bool)
    parser.add_argument('--value_coefficient', default=1, type=float)
    parser.add_argument('--side_info_coefficient', default=1, type=float)
    parser.add_argument('--mistakes_only', default = True, type = bool)
    parser.add_argument('--intervention', default='vanilla', type=str)

    return parser.parse_args(args)

def main() -> None:
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    # num_processes = cpu_count() - cfg.num_cpu_left
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    move_dict = {v: k for k, v in all_moves_dict.items()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model_path = "maia2-sae/weights.v2.pt"
    ckpt = torch.load(trained_model_path, map_location=torch.device(device))
    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    special_fens = defaultdict(list)
    counters = {'total': 0, 'monotonic': 0, 'transitional': 0, 'both': 0, 'all_correct': 0}
    transition_points = []
    best_transitional_moves = []
    all_best_transitional_moves = []
    transitional_maia_moves = []

    # dataset_dir = 'maia2-sae/dataset/maia2_trans_mono_dataset'
    # file_pattern = os.path.join(dataset_dir, 'lichess_db_eval_chunk_*.jsonl')
    # file_list = glob.glob(file_pattern)

    # for file_path in tqdm(file_list, desc="Processing Maia-2 trans-mono data"):
    #     special_fens, counters, transition_points, best_transitional_moves, transitional_maia_moves = process_positions(
    #         file_path, special_fens, best_transitional_moves, all_best_transitional_moves, transitional_maia_moves,
    #         counters, transition_points)
        
    tot_dataset_cnt = 10
    for i in tqdm(range(tot_dataset_cnt), desc="Processing Maia-2 trans-mono data"):
        file_path = f'/grace/cache/huggingface/lichess_data/lichess_db_evals_with_maia2/lichess_db_eval_chunk_{i}.jsonl'
        special_fens, counters, transition_points, best_transitional_moves, transitional_maia_moves = process_positions(
            file_path, 
            special_fens, 
            best_transitional_moves, 
            all_best_transitional_moves, 
            transitional_maia_moves,
            counters, 
            transition_points
        )

    total_non_all_correct = counters['total'] - counters['all_correct']
    print("Total positions:", counters['total'])
    print("Monotonic positions:", counters['monotonic'], f"({counters['monotonic'] / counters['total']:.2%})")
    print("Transitional positions:", counters['transitional'],
        f"({counters['transitional'] / total_non_all_correct:.2%} of non-all-correct positions)")
    print("Both monotonic and transitional:", counters['both'],
        f"({counters['both'] / total_non_all_correct:.2%} of non-all-correct positions)")

    intervention_data_dir = f'maia2-sae/dataset/intervention'
    if not os.path.exists(intervention_data_dir):
        os.makedirs(intervention_data_dir)

    if os.path.exists(f'{intervention_data_dir}/blunder_list.json'):
        with open(f'{intervention_data_dir}/blunder_list.json', 'r') as f:
            blunder_list = json.load(f)

    else:
        stockfish_path = 'maia2_sae/Stockfish-sf_16/src/stockfish'
        threshold = 150
        args = [(fen, move[0], stockfish_path, threshold) for fen, move in zip(special_fens['transitional'], transitional_maia_moves)]
        with Pool(processes=cpu_count()) as pool:
            blunder_list = list(tqdm(pool.imap(is_blunder, args), total=len(args)))

        with open(f'{intervention_data_dir}/blunder_list.json', 'w') as f:
            json.dump(blunder_list, f)

    with open('maia2-sae/dataset/intervention/layer7_squarewise_features.pickle', 'rb') as f:
        layer7_features = pickle.load(f)

    # Mediated intervention on the last transformer block hidden states
    target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states']
    squarewise_alarmbells = {}
    for concept_name, feature_results in layer7_features.items():
        square = concept_name[-2:]  
        best_feature = feature_results[0]  
        feature_idx, auc_score = best_feature
        squarewise_alarmbells[square] = (feature_idx, auc_score)

    print(squarewise_alarmbells)

    all_ground_truths = []
    intervention_site = []

    for key, value in squarewise_alarmbells.items():
        layer, feature_idx = target_key_list[1], value[0]
        intervention_site.append((layer, feature_idx))
        ground_truth = []
        for i in tqdm(range(len(special_fens['transitional']))):
            fen = special_fens['transitional'][i]
            move = best_transitional_moves[i]
            square_idx = square_index(key)
            try:
                no_longer_under_attack = is_piece_no_longer_under_attack(fen, move, square_idx)
                if no_longer_under_attack:
                    # is_blundered = np.all(is_blunder(fen, transitional_maia_moves[i][j]) for j in range(transition_points[i]))
                    # is_blundered = is_blunder(fen, transitional_maia_moves[i][0])
                    is_blundered = blunder_list[i]
                    micro_gt = no_longer_under_attack and \
                                is_blundered and \
                                not np.any([is_piece_no_longer_under_attack(fen, transitional_maia_moves[i][j], square_idx) for j in range((transition_points[i]))])
                    ground_truth.append(micro_gt)
                else:
                    ground_truth.append(0)
            except ValueError:
                ground_truth.append(0)
        # print(np.sum(ground_truth))
        all_ground_truths.append(ground_truth)
    all_ground_truths = torch.tensor(all_ground_truths, dtype=torch.int)

    valid_indices = [i for i in range(all_ground_truths.shape[1]) if torch.sum(all_ground_truths[:, i]) != 0]
    filtered_special_fens = [special_fens['transitional'][i] for i in valid_indices]
    filtered_transition_points = [transition_points[i] for i in valid_indices]
    filtered_best_transitional_moves = [best_transitional_moves[i] for i in valid_indices]
    filtered_all_ground_truths = all_ground_truths[:, valid_indices]
    filtered_all_best_transitional_moves = [all_best_transitional_moves[i] for i in valid_indices]

    filtered_data = {
        "filtered_special_fens": [special_fens['transitional'][i] for i in valid_indices],
        "filtered_transition_points": [transition_points[i] for i in valid_indices],
        "filtered_best_transitional_moves": [best_transitional_moves[i] for i in valid_indices],
        "filtered_all_ground_truths": filtered_all_ground_truths.tolist(), 
        "filtered_all_best_transitional_moves": [all_best_transitional_moves[i] for i in valid_indices]
    }

    board_inputs = []
    for fen in filtered_special_fens:
        board_tensor = board_to_tensor(chess.Board(fen))
        board_inputs.append(board_tensor)
    board_inputs = torch.stack(board_inputs, dim=0)

    output_path = f'{intervention_data_dir}/filtered_positions.json'
    with open(output_path, 'w') as json_file:
        json.dump(filtered_data, json_file)

    sae_dim = 2048
    sae_lr = 1e-5
    sae_site = "res"
    sae_date = "2023-12"
    sae = torch.load(f'maia2-sae/sae/{sae_date}-{sae_dim}-{sae_lr}-{sae_site}.pt')['sae_state_dicts']

    layer_one_features = squarewise_alarmbells.copy()
    elo_range = range(len(elo_dict) - 1) # excluding top level because all the time it predicts correctly in transitional positions
    candidate_strength = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

    if cfg.intervention == 'vanilla':
        original_results_dict, intervened_results_dict, transition_points_deviation = mediated_intervention(cfg, layer_one_features, elo_range, candidate_strength, intervention_site, filtered_all_ground_truths, filtered_special_fens, filtered_all_best_transitional_moves, filtered_transition_points, board_inputs, sae, model, move_dict, all_moves_dict)
        # For further visualization use
        original_acc, best_intervened_acc = find_best_intervention(original_results_dict, intervened_results_dict, elo_range)

        avg_best_accuracies = {}
        avg_original_accuracies = {}

        for elo in elo_range:
            best_acc_sum = sum(square_acc[elo] for square_acc in best_intervened_acc.values() if elo in square_acc)
            best_acc_count = sum(1 for square_acc in best_intervened_acc.values() if elo in square_acc)
            
            original_acc_sum = sum(square_acc[elo] for square_acc in original_acc.values() if elo in square_acc)
            original_acc_count = sum(1 for square_acc in original_acc.values() if elo in square_acc)
            
            if best_acc_count > 0:
                avg_best_accuracies[elo] = best_acc_sum / best_acc_count
            
            if original_acc_count > 0:
                avg_original_accuracies[elo] = original_acc_sum / original_acc_count

        print(avg_best_accuracies, avg_original_accuracies)

        save_dir='maia2-sae/dataset/intervention'
        results = {
            'best_accuracies': avg_best_accuracies,
            'original_accuracies': avg_original_accuracies
        }
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'vanilla_intervention_accuracies.pickle')
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

    if cfg.intervention == 'random':
        original_results_dict, random_results_dict = random_intervention(cfg, layer_one_features, elo_range, candidate_strength, intervention_site, filtered_all_ground_truths, filtered_special_fens, filtered_all_best_transitional_moves, filtered_transition_points, board_inputs, sae, model, move_dict, all_moves_dict)
        # For random tests; it's not the final version of random test
        original_acc, best_random_acc = find_best_intervention(original_results_dict, random_results_dict, elo_range)

    if cfg.intervention == 'probe':

        probe_weights = torch.load("reverse_pooled_weights.pth")


        original_results_dict, intervened_results_dict, transition_points_deviation = mediated_intervention(cfg, layer_one_features, elo_range, candidate_strength, intervention_site, filtered_all_ground_truths, filtered_special_fens, filtered_all_best_transitional_moves, filtered_transition_points, board_inputs, sae, model, move_dict, all_moves_dict, probe_weights)
        # For further visualization use
        original_acc, best_intervened_acc = find_best_intervention(original_results_dict, intervened_results_dict, elo_range)

        avg_best_accuracies = {}
        avg_original_accuracies = {}

        for elo in elo_range:
            best_acc_sum = sum(square_acc[elo] for square_acc in best_intervened_acc.values() if elo in square_acc)
            best_acc_count = sum(1 for square_acc in best_intervened_acc.values() if elo in square_acc)
            
            original_acc_sum = sum(square_acc[elo] for square_acc in original_acc.values() if elo in square_acc)
            original_acc_count = sum(1 for square_acc in original_acc.values() if elo in square_acc)
            
            if best_acc_count > 0:
                avg_best_accuracies[elo] = best_acc_sum / best_acc_count
            
            if original_acc_count > 0:
                avg_original_accuracies[elo] = original_acc_sum / original_acc_count

                print(avg_best_accuracies, avg_original_accuracies)

        save_dir='maia2-sae/dataset/intervention'
        results = {
            'best_accuracies': avg_best_accuracies,
            'original_accuracies': avg_original_accuracies
        }
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'linear_probe_intervention_accuracies.pickle')
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

    if cfg.intervention == 'steering_vector':

        steering_vectors = torch.load("steering_vectors.pth")


        original_results_dict, intervened_results_dict, transition_points_deviation = mediated_intervention(cfg, layer_one_features, elo_range, candidate_strength, intervention_site, filtered_all_ground_truths, filtered_special_fens, filtered_all_best_transitional_moves, filtered_transition_points, board_inputs, sae, model, move_dict, all_moves_dict, steering_vectors)
        # For further visualization use
        original_acc, best_intervened_acc = find_best_intervention(original_results_dict, intervened_results_dict, elo_range)

        avg_best_accuracies = {}
        avg_original_accuracies = {}

        for elo in elo_range:
            best_acc_sum = sum(square_acc[elo] for square_acc in best_intervened_acc.values() if elo in square_acc)
            best_acc_count = sum(1 for square_acc in best_intervened_acc.values() if elo in square_acc)
            
            original_acc_sum = sum(square_acc[elo] for square_acc in original_acc.values() if elo in square_acc)
            original_acc_count = sum(1 for square_acc in original_acc.values() if elo in square_acc)
            
            if best_acc_count > 0:
                avg_best_accuracies[elo] = best_acc_sum / best_acc_count
            
            if original_acc_count > 0:
                avg_original_accuracies[elo] = original_acc_sum / original_acc_count

                print(avg_best_accuracies, avg_original_accuracies)

        save_dir='maia2-sae/dataset/intervention'
        results = {
            'best_accuracies': avg_best_accuracies,
            'original_accuracies': avg_original_accuracies
        }
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'steering_vector_intervention_accuracies.pickle')
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)



    if cfg.intervention == 'patching':
        # TODO later
        pass





if __name__=="__main__":
    main()
