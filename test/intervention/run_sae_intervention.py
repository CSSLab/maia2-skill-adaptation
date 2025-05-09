import os
import pickle
import torch
import chess
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
import csv
import threading
import hashlib

from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves, get_side_info
from maia2.main import MAIA2Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ELO_DICT = create_elo_dict()
ELO_RANGE = range(len(ELO_DICT) - 1)
all_moves = get_all_possible_moves()
MOVE_DICT = {move: i for i, move in enumerate(all_moves)}
ALL_MOVE_DICT = {i: move for i, move in enumerate(all_moves)}

_thread_local = threading.local()

def is_piece_no_longer_under_attack(fen: str, move: str, square_index: int) -> bool:
    """
    Defensive willingness function.
    Returns True if the move resolves a threatening situation for a white piece.
    """
    board = chess.Board(fen)
    piece_before_move = board.piece_at(square_index)

    def is_under_attack(board: chess.Board, square_index: int) -> bool:
        attackers = board.attackers(chess.BLACK, square_index) 
        defenders = board.attackers(chess.WHITE, square_index)  
        return len(attackers) > len(defenders)

    was_under_attack_before = (is_under_attack(board, square_index) and 
                             piece_before_move is not None and 
                             piece_before_move.color is not chess.BLACK)
    
    move_obj = chess.Move.from_uci(move)
    if not board.is_legal(move_obj):
        raise ValueError(f"Illegal move: {move}")

    board.push(move_obj)
    piece_after_move = board.piece_at(square_index)
    is_under_attack_after = is_under_attack(board, square_index) and piece_after_move is not None

    return was_under_attack_before and not is_under_attack_after

def is_square_under_defensive_threat(fen: str, square_index: int) -> bool:
    board = chess.Board(fen)
    piece = board.piece_at(square_index)
    if piece is None or piece.color != chess.WHITE:
        return False
    return len(board.attackers(chess.BLACK, square_index)) > len(board.attackers(chess.WHITE, square_index))

class SAEIntervention:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self._load_model()
        self.sae = self._load_sae()
        self.best_features = self._load_best_features()
        self.intervention_strengths = [1, 2, 5, 10, 20, 50]
        self.scenarios = ['amplify_awareness']
        self.top_ns = [1, 5, 10]

        self._enable_intervention_hook()

    def _load_model(self):
        model = MAIA2Model(len(get_all_possible_moves()), ELO_DICT, self.cfg)
        ckpt = torch.load("maia2-sae/weights.v2.pt", map_location=DEVICE)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.eval().to(DEVICE)
    
    def _load_sae(self):
        sae = torch.load('maia2-sae/sae/best_jrsaes_2023-11-16384-1-res.pt', map_location=DEVICE)['sae_state_dicts']
        return {k: {name: param.to(DEVICE) for name, param in v.items()} 
                for k, v in sae.items() if 'transformer block' in k}
    
    def _load_best_features(self):
        features = {}
        layer = 'layer7'
        features[layer] = {
            'awareness': pickle.load(open(f'maia2-sae/dataset/intervention/{layer}_defensive_awareness.pickle', 'rb'))
        }
        return self._process_best_features(features)
    
    def _process_best_features(self, raw_features):
        processed = {}
        layer = 'layer7'
        processed[layer] = {}
        for concept in ['awareness']:
            for square, feature_list in raw_features[layer][concept].items():
                if feature_list:
                    processed_key = f"awareness_{square[-2:]}"
                    processed[layer][processed_key] = {
                        'indices': [feat[0] for feat in feature_list],
                        'aucs': [feat[1] for feat in feature_list],
                        'layer_key': 'transformer block 1 hidden states'
                    }
        return processed

    def _generate_batches(self, data_type='train', batch_size=256):
        base_path = "maia2-sae/dataset/blundered-transitional-dataset"
        cache_dir = os.path.join(base_path, f'all_final_cache_defensive_willingness_{data_type}')
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(f"{base_path}/{data_type}_moves.csv", 'r') as f:
            data = [line for line in csv.DictReader(f)]
        
        squares = [chess.square_name(sq) for sq in range(64)]
        for square in tqdm(squares, desc=f"Processing {data_type} squares"):
            square_hash = hashlib.md5(square.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{data_type}_{square_hash}.pkl")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    square_batches = pickle.load(f)
                for batch in square_batches:
                    yield batch
                continue
                
            square_idx = chess.parse_square(square)
            square_data = []
            
            for line in data:
                fen = line['fen']
                move_dict = eval(line['moves'])
                correct_move = move_dict['10']
                if self._is_square_relevant(fen, correct_move, square_idx):
                    board = chess.Board(fen)
                    legal_moves, _ = get_side_info(board, correct_move, MOVE_DICT)
                    
                    square_data.append({
                        'fen': fen,
                        'correct_move': correct_move,
                        'transition_point': int(line['transition_point']),
                        'legal_moves': legal_moves
                    })
            
            square_batches = []
            for i in range(0, len(square_data), batch_size):
                batch_data = square_data[i:i+batch_size]
                batch = {
                    'boards': torch.stack([board_to_tensor(chess.Board(line['fen'])) for line in batch_data]).to(DEVICE),
                    'correct_moves': [line['correct_move'] for line in batch_data],
                    'transition_points': [line['transition_point'] for line in batch_data],
                    'legal_moves': torch.stack([line['legal_moves'] for line in batch_data]).to(DEVICE),
                    'square': square,
                    'fens': [line['fen'] for line in batch_data]
                }
                square_batches.append(batch)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(square_batches, f)
            
            for batch in square_batches:
                yield batch

    def _is_square_relevant(self, fen, move, square_idx):
        board = chess.Board(fen)
        piece = board.piece_at(square_idx)
        if piece is None or piece.color != chess.WHITE:
            return False
        return is_piece_no_longer_under_attack(fen, move, square_idx)

    def _create_intervention(self, scenario, strength, square, top_n=1):
        interventions = defaultdict(dict)
        layer = 'layer7'
        feature_info = self.best_features[layer].get(f"awareness_{square}")
        if feature_info:
            layer_key = feature_info['layer_key']
            feature_indices = feature_info['indices'][:top_n]
            for idx in feature_indices:
                if scenario == 'amplify_awareness':
                    interventions[layer_key][idx] = strength
                elif scenario == 'ablate_awareness':
                    interventions[layer_key][idx] = -strength
        return interventions

    def _enable_intervention_hook(self):
        def get_intervention_hook(name):
            def hook(module, input, output):
                if not hasattr(_thread_local, 'residual_streams'):
                    _thread_local.residual_streams = {}
                _thread_local.residual_streams[name] = output.detach()
                if hasattr(_thread_local, 'modified_values') and name in _thread_local.modified_values:
                    return _thread_local.modified_values[name]
                return output
            return hook
        
        for i in range(self.cfg.num_blocks_vit):
            feedforward_module = self.model.module.transformer.elo_layers[i][1]
            feedforward_module.register_forward_hook(get_intervention_hook(f'transformer block {i} hidden states'))

    def _apply_intervention(self, board_inputs, elos_self, elos_oppo, interventions):
        def set_modified_values(modified_dict):
            _thread_local.modified_values = modified_dict

        def clear_modified_values():
            if hasattr(_thread_local, 'modified_values'):
                del _thread_local.modified_values

        board_inputs = board_inputs.to(DEVICE)
        elos_self = elos_self.to(DEVICE)
        elos_oppo = elos_oppo.to(DEVICE)

        with torch.no_grad():
            _, _, _ = self.model(board_inputs, elos_self, elos_oppo)
            clean_activations = getattr(_thread_local, 'residual_streams', {}).copy()

        sae_activations = {}
        for key in clean_activations:
            if key in self.sae:
                act = torch.mean(clean_activations[key], dim=1)
                encoded = torch.nn.functional.linear(act, self.sae[key]['encoder_DF.weight'], 
                                            self.sae[key]['encoder_DF.bias'])
                sae_activations[key] = torch.nn.functional.relu(encoded)

        modified_sae_activations = {}
        for key in interventions:
            if key not in sae_activations:
                continue
                
            modified = sae_activations[key].clone()
            for idx, strength in interventions[key].items():
                modified[:, idx] *= strength
                
            modified_sae_activations[key] = modified

        reconstructed_activations = {}
        for key in modified_sae_activations:
            decoded = torch.nn.functional.linear(
                modified_sae_activations[key],
                self.sae[key]['decoder_FD.weight'],
                self.sae[key]['decoder_FD.bias']
            )
            
            reconstructed_activations[key] = decoded.unsqueeze(1).expand(-1, 8, -1)

        set_modified_values(reconstructed_activations)
        with torch.no_grad():
            logits, _, _ = self.model(board_inputs, elos_self, elos_oppo)
        clear_modified_values()
        
        return logits

    def find_optimal_parameters(self):
        print("Finding optimal parameters on training set...")
        train_results = defaultdict(lambda: defaultdict(list))
        
        train_batches = list(self._generate_batches(data_type='train'))
        for batch in tqdm(train_batches, desc="Processing training batches"):
            square = batch['square']
            legal_moves = batch['legal_moves']
            
            for scenario in self.scenarios:
                for strength in self.intervention_strengths:
                    for top_n in self.top_ns:
                        interventions = self._create_intervention(scenario, strength, square, top_n)
                        
                        for elo in ELO_RANGE:
                            key = (scenario, square, elo)
                            elos_self = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                            elos_oppo = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                            
                            logits = self._apply_intervention(batch['boards'], elos_self, elos_oppo, interventions)
                            probs = (logits * legal_moves.to(DEVICE)).softmax(-1)
                            
                            correct = sum(
                                ALL_MOVE_DICT[prob.argmax().item()] == batch['correct_moves'][i]
                                for i, prob in enumerate(probs)
                            )
                            accuracy = correct / len(batch['boards'])
                            
                            train_results[key][f"{strength}_{top_n}"].append(accuracy)
        
        self.optimal_params = {}
        for key in train_results:
            scenario, square, elo = key
            param_accuracies = {param: np.mean(accs) for param, accs in train_results[key].items()}
            
            if scenario == 'amplify_awareness':
                best_param = max(param_accuracies, key=param_accuracies.get)
            else:
                best_param = min(param_accuracies, key=param_accuracies.get)
                
            strength, top_n = map(int, best_param.split('_'))
            self.optimal_params[key] = (strength, top_n)
        
        print("Optimal parameters found!")

    def evaluate_test_set(self):
        print("Evaluating on test set with optimal parameters...")
        test_results = {}
        position_results = {}
        
        test_batches = list(self._generate_batches(data_type='test'))
        for batch_idx, batch in enumerate(tqdm(test_batches, desc="Processing test batches")):
            square = batch['square']
            legal_moves = batch['legal_moves']
            correct_moves = batch['correct_moves']
            
            if 'original_tps' not in test_results:
                test_results['original_tps'] = []
            test_results['original_tps'].extend(batch['transition_points'])
            
            for elo in ELO_RANGE:
                elos_self = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                elos_oppo = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                
                with torch.no_grad():
                    logits, _, _ = self.model(batch['boards'].to(DEVICE), elos_self, elos_oppo)
                    probs = (logits * legal_moves.to(DEVICE)).softmax(-1)
                    
                    for i, prob in enumerate(probs):
                        position_key = f"{batch_idx}_{i}"
                        predicted_move = ALL_MOVE_DICT[prob.argmax().item()]
                        is_correct = predicted_move == correct_moves[i]
                        
                        if position_key not in position_results:
                            position_results[position_key] = {}
                        if 'baseline' not in position_results[position_key]:
                            position_results[position_key]['baseline'] = {}
                        
                        position_results[position_key]['baseline'][elo] = is_correct
                
                for scenario in self.scenarios:
                    key = (scenario, square, elo)
                    if key not in self.optimal_params:
                        continue
                    
                    strength, top_n = self.optimal_params[key]
                    interventions = self._create_intervention(scenario, strength, square, top_n)
                    
                    logits = self._apply_intervention(batch['boards'], elos_self, elos_oppo, interventions)
                    probs = (logits * legal_moves.to(DEVICE)).softmax(-1)
                    
                    for i, prob in enumerate(probs):
                        position_key = f"{batch_idx}_{i}"
                        predicted_move = ALL_MOVE_DICT[prob.argmax().item()]
                        is_correct = predicted_move == correct_moves[i]
                        
                        if position_key not in position_results:
                            position_results[position_key] = {}
                        if scenario not in position_results[position_key]:
                            position_results[position_key][scenario] = {}
                        
                        position_results[position_key][scenario][elo] = is_correct
                    
                    correct = sum(
                        ALL_MOVE_DICT[prob.argmax().item()] == correct_moves[i]
                        for i, prob in enumerate(probs)
                    )
                    
                    result_key = (scenario, square, elo)
                    if result_key not in test_results:
                        test_results[result_key] = {'total': 0, 'correct': 0, 'params': (strength, top_n)}
                    test_results[result_key]['total'] += len(batch['boards'])
                    test_results[result_key]['correct'] += correct
        
        transition_points = {}
        position_contexts = {}
        
        for position_key, result_dict in position_results.items():
            batch_idx, pos_idx = map(int, position_key.split('_'))
            batch = test_batches[batch_idx]
            correct_move = batch['correct_moves'][pos_idx]
            
            position_contexts[position_key] = {
                'correct_move': correct_move,
                'square': batch['square']
            }
            
            for scenario in ['baseline'] + self.scenarios:
                if scenario not in result_dict:
                    continue
                    
                skill_results = result_dict[scenario]
                
                elo_levels = sorted(skill_results.keys())
                
                if all(skill_results[elo] for elo in elo_levels):
                    if scenario not in transition_points:
                        transition_points[scenario] = []
                    transition_points[scenario].append(0)
                    continue
                    
                # Check if all ELO levels made incorrect predictions
                if all(not skill_results[elo] for elo in elo_levels):
                    if scenario not in transition_points:
                        transition_points[scenario] = []
                    transition_points[scenario].append(10)  
                    continue
                    
                found_transition = False
                for level_idx in range(len(elo_levels) - 1):
                    curr_level = elo_levels[level_idx]
                    next_level = elo_levels[level_idx + 1]
                    
                    if not skill_results[curr_level] and skill_results[next_level]:
                        if all(skill_results[elo_levels[i]] for i in range(level_idx + 1, len(elo_levels))):
                            if scenario not in transition_points:
                                transition_points[scenario] = []
                            transition_points[scenario].append(next_level)
                            found_transition = True
                            break
                            
                if not found_transition:
                    continue
        
        test_results['transition_points'] = transition_points
        test_results['position_results'] = dict(position_results)
        test_results['position_contexts'] = dict(position_contexts)
        
        for scenario, points in transition_points.items():
            if points:
                avg_tp = sum(points) / len(points)
                num_positions = len(points)
                print(f"{scenario}: Transition Point = {avg_tp}, Total Positions {num_positions}")
                test_results[f'avg_transition_{scenario}'] = avg_tp
                test_results[f'num_positions_{scenario}'] = num_positions
        
        return test_results

    def run_experiment(self):
        self.find_optimal_parameters()
        
        with open('maia2-sae/intervention_results/defensive_optimal_params.pkl', 'wb') as f:
            pickle.dump(self.optimal_params, f)

        test_results = self.evaluate_test_set()
        self._save_results(test_results, 'maia2-sae/new_intervention_results/defensive_intervention_results.pkl')
        return test_results

    def _save_results(self, results, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            'test_results': results,
            'optimal_params': self.optimal_params,
            'original_tps': results.get('original_tps', [])
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

class InterventionConfig:
    def __init__(self):
        self.input_channels = 18  
        self.dim_cnn = 256
        self.dim_vit = 1024      
        self.num_blocks_cnn = 5  
        self.num_blocks_vit = 2   
        self.vit_length = 8
        self.elo_dim = 128
        self.side_info = True
        self.value = True
        self.value_coefficient = 1.0
        self.side_info_coefficient = 1.0
        self.mistakes_only = True
        self.dropout_rate = 0.1
        self.cnn_dropout = 0.5
        self.sae_dim = 16384
        self.sae_lr = 1
        self.sae_site = "res"

if __name__ == "__main__":
    cfg = InterventionConfig()
    intervention = SAEIntervention(cfg)
    results = intervention.run_experiment()
    print("\nExperiment completed. Results saved.")