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
        self.intervention_strengths = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
        self.scenarios = ['amplify_awareness', 'ablate_awareness']
        self.top_ns = [1, 5, 10, 20]

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

    def _generate_batches(self, batch_size=256):
        base_path = "maia2-sae/dataset/blundered-transitional-dataset"
        cache_dir = os.path.join(base_path, 'cache_defensive_willingness')
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(f"{base_path}/test_moves.csv", 'r') as f:
            data = [line for line in csv.DictReader(f)]
        
        squares = [chess.square_name(sq) for sq in range(64)]
        for square in tqdm(squares, desc="Processing squares"):
            square_hash = hashlib.md5(square.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"test_{square_hash}.pkl")
            
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
                    'square': square
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
        # return is_square_under_defensive_threat(fen, square_idx)
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

    def _evaluate_batch(self, batch, results):
        square = batch['square']
        legal_moves = batch['legal_moves']
        
        if 'original_tps' not in results:
            results['original_tps'] = []
            results['transition_shifts'] = {}
        
        results['original_tps'].extend(batch['transition_points'])
        
        best_amplify = defaultdict(lambda: (-1, -1, 0))  
        worst_ablate = defaultdict(lambda: (float('inf'), -1, 0))

        for scenario in self.scenarios:
            for strength in self.intervention_strengths:
                for top_n in self.top_ns:
                    interventions = self._create_intervention(scenario, strength, square, top_n)
                    
                    for elo in ELO_RANGE:
                        elos_self = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                        elos_oppo = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                        
                        logits = self._apply_intervention(batch['boards'], elos_self, elos_oppo, interventions)
                        probs = (logits * legal_moves.to(DEVICE)).softmax(-1)
                        
                        correct = sum(
                            ALL_MOVE_DICT[prob.argmax().item()] == batch['correct_moves'][i]
                            for i, prob in enumerate(probs)
                        )
                        accuracy = correct / len(batch['boards'])

                        if scenario == 'amplify_awareness':
                            current_best = best_amplify[(square, elo)]
                            if accuracy > current_best[0]:
                                best_amplify[(square, elo)] = (accuracy, strength, top_n)
                        else:
                            current_worst = worst_ablate[(square, elo)]
                            if accuracy < current_worst[0]:
                                worst_ablate[(square, elo)] = (accuracy, strength, top_n)
                        
                        result_key = (scenario, strength, square, elo, top_n)
                        if result_key not in results:
                            results[result_key] = {'total': 0, 'correct': 0}
                        results[result_key]['total'] += len(batch['boards'])
                        results[result_key]['correct'] += correct

        for (square, elo), (acc, strength, top_n) in best_amplify.items():
            result_key = ('amplify_awareness', 'best', square, elo)
            source_key = ('amplify_awareness', strength, square, elo, top_n)
            results[result_key] = results[source_key].copy()
            results[result_key].update({'top_n': top_n, 'strength': strength})
            
        for (square, elo), (acc, strength, top_n) in worst_ablate.items():
            result_key = ('ablate_awareness', 'worst', square, elo)
            source_key = ('ablate_awareness', strength, square, elo, top_n)
            results[result_key] = results[source_key].copy()
            results[result_key].update({'top_n': top_n, 'strength': strength})

    def run_experiment(self):
        results = {}
        
        print("Running experiment with all strengths and top_n features...")
        test_batches = list(self._generate_batches())
        for batch in tqdm(test_batches, desc="Processing batches"):
            self._evaluate_batch(batch, results)
            
        self._save_results(results, 'maia2-sae/intervention_results/final_intervention_results_defensive_multi.pkl')
        return results

    def _save_results(self, results, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'test_results': results,
                'original_tps': results.get('original_tps', [])
            }, f)

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