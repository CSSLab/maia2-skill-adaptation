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
        self.intervention_strengths = [1, 2, 5, 10, 20]
        self.scenarios = ['amplify_awareness', 'ablate_awareness']
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
        cache_dir = os.path.join(base_path, f'cache_defensive_willingness_{data_type}')
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
            else:  # ablate_awareness
                best_param = min(param_accuracies, key=param_accuracies.get)
                
            strength, top_n = map(int, best_param.split('_'))
            self.optimal_params[key] = (strength, top_n)
        
        print("Optimal parameters found!")

    def evaluate_test_set(self):
        print("Evaluating on test set with optimal parameters...")
        test_results = {}
        
        test_batches = list(self._generate_batches(data_type='test'))
        for batch in tqdm(test_batches, desc="Processing test batches"):
            square = batch['square']
            legal_moves = batch['legal_moves']
            
            if 'original_tps' not in test_results:
                test_results['original_tps'] = []
            test_results['original_tps'].extend(batch['transition_points'])
            
            for scenario in self.scenarios:
                for elo in ELO_RANGE:
                    key = (scenario, square, elo)
                    if key not in self.optimal_params:
                        continue
                        
                    strength, top_n = self.optimal_params[key]
                    interventions = self._create_intervention(scenario, strength, square, top_n)
                    
                    elos_self = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                    elos_oppo = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                    
                    logits = self._apply_intervention(batch['boards'], elos_self, elos_oppo, interventions)
                    probs = (logits * legal_moves.to(DEVICE)).softmax(-1)
                    
                    correct = sum(
                        ALL_MOVE_DICT[prob.argmax().item()] == batch['correct_moves'][i]
                        for i, prob in enumerate(probs)
                    )
                    
                    result_key = (scenario, square, elo)
                    if result_key not in test_results:
                        test_results[result_key] = {'total': 0, 'correct': 0, 'params': (strength, top_n)}
                    test_results[result_key]['total'] += len(batch['boards'])
                    test_results[result_key]['correct'] += correct

        return test_results

        # intervention with post-intervention probing 
        def evaluate_test_set_with_probing(self):
            print("Evaluating on test set with optimal parameters...")
            test_results = {}
            activation_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            
            test_batches = list(self._generate_batches(data_type='test'))
            for batch in tqdm(test_batches, desc="Processing test batches"):
                square = batch['square']
                legal_moves = batch['legal_moves']
                
                if 'original_tps' not in test_results:
                    test_results['original_tps'] = []
                test_results['original_tps'].extend(batch['transition_points'])
                
                for scenario in self.scenarios:
                    for elo in ELO_RANGE:
                        key = (scenario, square, elo)
                        if key not in self.optimal_params:
                            continue
                        
                        strength, top_n = self.optimal_params[key]
                        interventions = self._create_intervention(scenario, strength, square, top_n)
                        
                        elos_self = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                        elos_oppo = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                        
                        logits, decoded_vectors = self._apply_intervention(
                            batch['boards'], elos_self, elos_oppo, interventions
                        )
                        
                        probs = (logits * legal_moves.to(DEVICE)).softmax(-1)
                        correct = sum(
                            ALL_MOVE_DICT[prob.argmax().item()] == batch['correct_moves'][i]
                            for i, prob in enumerate(probs)
                        )
                        
                        result_key = (scenario, square, elo)
                        if result_key not in test_results:
                            test_results[result_key] = {'total': 0, 'correct': 0, 'params': (strength, top_n)}
                        test_results[result_key]['total'] += len(batch['boards'])
                        test_results[result_key]['correct'] += correct

                        for i in range(len(batch['boards'])):
                            activation = decoded_vectors[i].detach().cpu().numpy()
                            activation_data[scenario][elo][square].append(activation)

            print("\nActivation Data Statistics:")
            for scenario in self.scenarios:
                print(f"\n{scenario}:")
                for elo in ELO_RANGE:
                    if activation_data[scenario][elo]:
                        squares = list(activation_data[scenario][elo].keys())
                        sizes = [len(activation_data[scenario][elo][sq]) for sq in squares]
                        print(f"ELO {ELO_DICT[elo]}: {len(squares)} squares, "
                            f"avg {np.mean(sizes):.1f} samples/square, "
                            f"min {min(sizes)} samples, max {max(sizes)} samples")

            probe_results = defaultdict(lambda: defaultdict(list))
            
            for scenario in tqdm(self.scenarios, desc="Processing scenarios"):
                for elo in ELO_RANGE:
                    if not activation_data[scenario][elo]:
                        continue
                        
                    squares = list(activation_data[scenario][elo].keys())
                    print(f"\n{scenario} - ELO {ELO_DICT[elo]}: Processing {len(squares)} squares")
                    
                    for square_idx, square in enumerate(squares):
                        positive_activations = activation_data[scenario][elo][square]
                        if not positive_activations:
                            continue
                            
                        # Get multiple negative squares for better balance
                        neg_indices = [(square_idx + i + 1) % len(squares) for i in range(3)]
                        neg_squares = [squares[i] for i in neg_indices]
                        negative_activations = []
                        for neg_square in neg_squares:
                            negative_activations.extend(activation_data[scenario][elo][neg_square])
                        
                        if not negative_activations:
                            continue
                        
                        min_size = min(len(positive_activations), len(negative_activations))
                        print(f"Square {square}: {len(positive_activations)} positive, {len(negative_activations)} negative samples")
                        if min_size < 10:
                            print(f"Skipping {square} due to insufficient samples")
                            continue
                        
                        X_positive = np.array(positive_activations[:min_size])
                        X_negative = np.array(negative_activations[:min_size])
                        
                        X = np.vstack([X_positive, X_negative])
                        y = np.array([1] * min_size + [0] * min_size)
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        model = LogisticRegression(max_iter=32)
                        model.fit(X_train, y_train)
                        
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        y_pred = (y_pred_proba > 0.5).astype(int)
                        
                        try:
                            auc = roc_auc_score(y_test, y_pred_proba)
                            acc = accuracy_score(y_test, y_pred)
                            probe_results[scenario][elo].append((auc, acc))
                            print(f"Square {square}: AUC = {auc:.4f}, Accuracy = {acc:.4f}")
                        except Exception as e:
                            print(f"Error evaluating square {square}: {str(e)}")
                            continue

            # Print and store results
            print("\nIntervention Probe Results:")
            scenario_averages = {}
            for scenario in self.scenarios:
                print(f"\n{scenario.capitalize()}:")
                elo_scores = {}
                for elo in ELO_RANGE:
                    if probe_results[scenario][elo]:
                        aucs, accs = zip(*probe_results[scenario][elo])
                        avg_auc = np.mean(aucs)
                        avg_acc = np.mean(accs)
                        elo_scores[elo] = {'auc': avg_auc, 'accuracy': avg_acc}
                        print(f"ELO {ELO_DICT[elo]}: Average AUC = {avg_auc:.4f}, Average Accuracy = {avg_acc:.4f}")
                
                if elo_scores:
                    overall_auc = np.mean([scores['auc'] for scores in elo_scores.values()])
                    overall_acc = np.mean([scores['accuracy'] for scores in elo_scores.values()])
                    print(f"Overall average for {scenario}: AUC = {overall_auc:.4f}, Accuracy = {overall_acc:.4f}")
                    scenario_averages[scenario] = {
                        'elo_scores': elo_scores,
                        'overall_auc': overall_auc,
                        'overall_accuracy': overall_acc
                    }

            test_results['probe_results'] = {
                'per_scenario_elo': dict(probe_results),
                'scenario_averages': scenario_averages
            }
            
            return test_results

    def run_experiment(self):
        self.find_optimal_parameters()
        
        with open('maia2-sae/intervention_results/optimal_params.pkl', 'wb') as f:
            pickle.dump(self.optimal_params, f)

        test_results = self.evaluate_test_set()
        self._save_results(test_results, 'maia2-sae/intervention_results/final_optimized_intervention_results.pkl')
        return test_results

    def _save_results(self, results, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'test_results': results,
                'optimal_params': self.optimal_params,
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