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
import json

from ..maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves, get_side_info
from ..maia2.main import MAIA2Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ELO_DICT = create_elo_dict()
ELO_RANGE = range(len(ELO_DICT) - 1)
all_moves = get_all_possible_moves()
MOVE_DICT = {move: i for i, move in enumerate(all_moves)}
ALL_MOVE_DICT = {i: move for i, move in enumerate(all_moves)}
_thread_local = threading.local()

def is_attacking_opportunity_taken(fen: str, move: str, square_index: int) -> bool:
    board = chess.Board(fen)
    piece_before_move = board.piece_at(square_index)
    def has_attacking_advantage(board: chess.Board, square_index: int) -> bool:
        attackers = board.attackers(chess.WHITE, square_index)
        defenders = board.attackers(chess.BLACK, square_index)
        return len(attackers) > len(defenders)
    had_advantage_before = (
        has_attacking_advantage(board, square_index)
        and piece_before_move is not None
        and piece_before_move.color == chess.BLACK
    )
    move_obj = chess.Move.from_uci(move)
    if not board.is_legal(move_obj):
        return False
    return had_advantage_before and move_obj.to_square == square_index

def is_piece_no_longer_under_attack(fen: str, move: str, square_index: int) -> bool:
    board = chess.Board(fen)
    piece_before_move = board.piece_at(square_index)
    def is_under_attack(board: chess.Board, square_index: int) -> bool:
        attackers = board.attackers(chess.BLACK, square_index)
        defenders = board.attackers(chess.WHITE, square_index)
        return len(attackers) > len(defenders)
    was_under_attack_before = (
        is_under_attack(board, square_index)
        and piece_before_move is not None
        and piece_before_move.color == chess.WHITE
    )
    move_obj = chess.Move.from_uci(move)
    if not board.is_legal(move_obj):
        return False
    board.push(move_obj)
    piece_after_move = board.piece_at(square_index)
    is_under_attack_after = is_under_attack(board, square_index) and piece_after_move is not None
    return was_under_attack_before and not is_under_attack_after

class ProbeIntervention:
    def __init__(self, cfg, offensive_probe_path, defensive_probe_path):
        self.cfg = cfg
        self.model = self._load_model()
        # self.offensive_probes = self._load_probe_weights(offensive_probe_path)
        self.defensive_probes = self._load_probe_weights(defensive_probe_path)
        self.intervention_strengths = [1, 5, 25]
        self.best_strengths = {
            # 'offensive': {},
            'defensive': {}
        }

        self._enable_intervention_hook()

    def _load_model(self):
        model = MAIA2Model(len(get_all_possible_moves()), ELO_DICT, self.cfg)
        ckpt = torch.load("maia2-sae/weights.v2.pt", map_location=DEVICE)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.eval().to(DEVICE)

    def _load_probe_weights(self, path):
        return torch.load(path)
    
    def _is_square_relevant_offensive(self, fen, move, square_idx):
        board = chess.Board(fen)
        piece = board.piece_at(square_idx)
        if piece is None or piece.color != chess.BLACK:
            return False
        return is_attacking_opportunity_taken(fen, move, square_idx)

    def _is_square_relevant_defensive(self, fen, move, square_idx):
        board = chess.Board(fen)
        piece = board.piece_at(square_idx)
        if piece is None or piece.color != chess.WHITE:
            return False
        return is_piece_no_longer_under_attack(fen, move, square_idx)

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
            ff = self.model.module.transformer.elo_layers[i][1]
            ff.register_forward_hook(get_intervention_hook(f'transformer block {i} hidden states'))
    
    def _generate_batches(self, data_type='train', batch_size=256, concept='defensive'):
        base_path = "../../../../../../ada1/projects/chess/maia_interp/maia2-sae/dataset/blundered-transitional-dataset"
        with open(f"{base_path}/{data_type}_moves.csv", 'r') as f:
            data = [line for line in csv.DictReader(f)]

        squares = [chess.square_name(sq) for sq in range(64)]
        for square in tqdm(squares, desc=f"Generating {data_type} batches ({concept})"):
            square_idx = chess.parse_square(square)
            square_data = []

            max_samples = int(0.25 * len(data))
            data_subset = data[:max_samples]

            for line in data_subset:
                fen = line['fen']
                move_dict = eval(line['moves'])
                correct_move = move_dict['10']

                is_relevant = (
                    self._is_square_relevant_offensive(fen, correct_move, square_idx)
                    if concept == 'offensive'
                    else self._is_square_relevant_defensive(fen, correct_move, square_idx)
                )

                if is_relevant:
                    board = chess.Board(fen)
                    legal_moves, _ = get_side_info(board, correct_move, MOVE_DICT)
                    square_data.append({
                        'fen': fen,
                        'correct_move': correct_move,
                        'legal_moves': legal_moves
                    })


            for i in range(0, len(square_data), batch_size):
                batch_data = square_data[i:i+batch_size]
                yield {
                    'boards': torch.stack([board_to_tensor(chess.Board(x['fen'])) for x in batch_data]).to(DEVICE),
                    'correct_moves': [x['correct_move'] for x in batch_data],
                    'legal_moves': torch.stack([x['legal_moves'] for x in batch_data]).to(DEVICE),
                    'square': square,
                    'concept': concept
                }

    def _create_intervention(self, concept, square, strength):
        interventions = defaultdict(dict)
        layer_key = 'transformer block 1 hidden states'
        probe_weights = self.offensive_probes if concept == 'offensive' else self.defensive_probes
        if square in probe_weights:
            interventions[layer_key][square] = strength
        return interventions

    def _apply_probe_intervention(self, board_inputs, elos_self, elos_oppo, interventions):
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

        reconstructed_activations = {}
        for layer_key, square_dict in interventions.items():
            if layer_key not in clean_activations:
                continue
            batch_activations = clean_activations[layer_key].clone()
            for i in range(batch_activations.shape[0]):
                for square_name, strength in square_dict.items():
                    # vec = self.offensive_probes[square_name] if square_name in self.offensive_probes else self.defensive_probes[square_name]
                    vec = self.defensive_probes[square_name]
                    probe_vec = torch.tensor(vec, device=DEVICE)
                    batch_activations[i] += strength * probe_vec
            reconstructed_activations[layer_key] = batch_activations

        set_modified_values(reconstructed_activations)
        with torch.no_grad():
            logits, decoded_vectors, _ = self.model(board_inputs, elos_self, elos_oppo)
        clear_modified_values()
        return logits, decoded_vectors
    def find_optimal_strengths(self, concept):
        print(f"Finding optimal strengths for {concept} concepts...")
        train_batches = list(self._generate_batches(data_type='train', concept=concept))
        accuracy_by_strength = defaultdict(lambda: defaultdict(list))  # [square][strength] → accuracies

        for batch in tqdm(train_batches, desc=f"Training batches ({concept})"):
            square = batch['square']
            legal_moves = batch['legal_moves']
            correct_moves = batch['correct_moves']

            for strength in self.intervention_strengths:
                interventions = self._create_intervention(concept, square, strength)
                for elo in ELO_RANGE:
                    elos_self = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                    elos_oppo = torch.full((len(batch['boards']),), elo, device=DEVICE).long()

                    logits, _ = self._apply_probe_intervention(batch['boards'], elos_self, elos_oppo, interventions)
                    probs = (logits * legal_moves).softmax(-1)
                    correct = sum(
                        ALL_MOVE_DICT[prob.argmax().item()] == correct_moves[i]
                        for i, prob in enumerate(probs)
                    )
                    accuracy = correct / len(correct_moves)
                    accuracy_by_strength[square][strength].append(accuracy)

        for square in accuracy_by_strength:
            strength_scores = {
                s: np.mean(accs)
                for s, accs in accuracy_by_strength[square].items()
            }
            best_strength = max(strength_scores, key=strength_scores.get)
            self.best_strengths[concept][square] = best_strength

        print(f"Completed strength optimization for {concept}.")

    def evaluate_test_set(self, concept):
        print(f"Evaluating test set for {concept}...")
        test_batches = list(self._generate_batches(data_type='test', concept=concept))
        accuracy_by_elo = defaultdict(lambda: {'correct': 0, 'total': 0})
        transition_points = []

        for batch in tqdm(test_batches, desc=f"Evaluating ({concept})"):
            square = batch['square']
            strength = self.best_strengths[concept].get(square)
            if strength is None:
                continue
            
            legal_moves = batch['legal_moves']
            correct_moves = batch['correct_moves']
            interventions = self._create_intervention(concept, square, strength)

            cross_elo_preds = []

            for elo in ELO_RANGE:
                elos_self = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                elos_oppo = torch.full((len(batch['boards']),), elo, device=DEVICE).long()
                logits, _ = self._apply_probe_intervention(batch['boards'], elos_self, elos_oppo, interventions)
                probs = (logits * legal_moves).softmax(-1)

                preds = [ALL_MOVE_DICT[prob.argmax().item()] for prob in probs]
                cross_elo_preds.append(preds)

                correct = sum(
                    [pred == correct_moves[i]
                    for i, pred in enumerate(preds)]
                )
                accuracy_by_elo[elo]['correct'] += correct
                accuracy_by_elo[elo]['total'] += len(correct_moves)

            for j in range(len(correct_moves)):
                found_transition = False
                
                if all(cross_elo_preds[elo][j] == correct_moves[j] for elo in ELO_RANGE):
                    transition_points.append(0)
                    continue

                if all(cross_elo_preds[elo][j] != correct_moves[j] for elo in ELO_RANGE):
                    transition_points.append(10)
                    continue

                for elo in ELO_RANGE[:-1]:
                    if (cross_elo_preds[elo][j] != correct_moves[j]):
                        if all(cross_elo_preds[lo][j] == correct_moves[j] for lo in ELO_RANGE[elo+1:]):
                            transition_points.append(elo+1)
                            found_transition = True
                            break

                if not found_transition:
                    continue

        return accuracy_by_elo, transition_points 
    
    def _save_results(self, results, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(results, f)

    def run_experiment(self):
        all_results = {}

        for concept in ['defensive']:
            self.find_optimal_strengths(concept)
            results, transition_points = self.evaluate_test_set(concept)
            all_results[concept] = results

            print(f"\n{concept.capitalize()} awareness accuracy by ELO:")
            for elo in sorted(results):
                correct = results[elo]['correct']
                total = results[elo]['total']
                acc = correct / total if total > 0 else 0.0
                print(f"Elo {elo}: {acc:.3f} ({correct}/{total})")

        print("\nOverall Combined Accuracy by ELO:")
        for elo in sorted(ELO_RANGE):
            # c1 = all_results['offensive'][elo]['correct']
            c2 = all_results['defensive'][elo]['correct']
            # t1 = all_results['offensive'][elo]['total']
            t2 = all_results['defensive'][elo]['total']
            total_correct = c2
            total_total = t2

            acc = total_correct / total_total if total_total > 0 else 0.0
            print(f"Elo {elo}: {acc:.3f} ({total_correct}/{total_total})")
        
        with open("probe_tp_2.json", "w") as outfile:
            json.dump(transition_points, outfile)

        avg_trans_point = sum(transition_points)/len(transition_points)
        print("Post-Intervention Average Transition Point: " + str(avg_trans_point))

        save_path = './rebuttal_final_probe_intervention_results.pkl'
        self._save_results(all_results, save_path)
        print(f"\nExperiment completed. Results saved to {save_path}")

        return all_results

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
        self.mistakes_only = False
        self.dropout_rate = 0.1
        self.cnn_dropout = 0.5
        self.sae_dim = 16384
        self.sae_lr = 1
        self.sae_site = "res"


if __name__ == "__main__":
    cfg = InterventionConfig()
    intervention = ProbeIntervention(
        cfg,
        offensive_probe_path='rebuttal_oa_probes_reverse_pooled_weights.pth',
        defensive_probe_path='rebuttal_da_probes_reverse_pooled_weights.pth'
    )
    intervention.run_experiment()
