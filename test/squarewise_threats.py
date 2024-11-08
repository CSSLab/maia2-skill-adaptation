import pickle
import torch
import tqdm
import chess
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Tuple, Any
import pdb

from ..maia2.main import *
from ..maia2.utils import *

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

def is_piece_no_longer_under_attack(fen: str, move: str, square_index: int) -> bool:
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

def initialize_concept_functions() -> Tuple[List[str], Dict[str, Any]]:
    move_dependent_header = []
    function_map = {}

    for square in range(64):
        defensive_root_func = 'is_piece_no_longer_under_attack'
        defensive_key = f"{defensive_root_func}_{chess.square_name(square)}"
        defensive_partial_function = partial(is_piece_no_longer_under_attack, square_index=square)

        function_map[defensive_key] = defensive_partial_function
        move_dependent_header.append(defensive_key)

    return move_dependent_header, function_map

def select_indices(lst: List[int], max_positive: int = 200, max_negative: int = 500) -> Tuple[List[int], List[int]]:
    positive_indices = [i for i, val in enumerate(lst) if val == 1]
    negative_indices = [i for i, val in enumerate(lst) if val == 0]

    selected_positive = random.sample(positive_indices, min(max_positive, len(positive_indices)))
    selected_negative = random.sample(negative_indices, min(max_negative, len(negative_indices)))

    print(f"Selected {len(selected_positive)} positive and {len(selected_negative)} negative examples")
    return selected_positive, selected_negative

def evaluate_sae_feature(feature_activations: torch.Tensor, ground_truth: torch.Tensor) -> float:
    if torch.sum(feature_activations) == 0:
        return 0
    return roc_auc_score(ground_truth.cpu().numpy(), feature_activations.cpu().numpy())

def process_sae_feature(args: Tuple[int, torch.Tensor, torch.Tensor]) -> Tuple[int, float]:
    feature_index, feature_activations, ground_truth = args
    if torch.sum(feature_activations) == 0:
        return feature_index, 0
    return feature_index, evaluate_sae_feature(feature_activations, ground_truth)

def cache_examples_and_activations(concept_functions: Dict[str, Any],
                                 concept_names: List[str],
                                 sae_activations: Dict[str, torch.Tensor],
                                 board_fens: List[str],
                                 preds: List[str],
                                 move_dependent_header: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    samples_and_activations = {}
    
    for concept_name, concept_func in concept_functions.items():
        if concept_name not in move_dependent_header:
            continue

        print(f"Processing {concept_name}")
        labels = [concept_func(board_fens[i], preds[i]) for i in range(len(board_fens))]
        
        positive_indices, negative_indices = select_indices(labels)
        temp_dic = {
            "positives": torch.tensor([1 for _ in positive_indices]),
            "negatives": torch.tensor([0 for _ in negative_indices])
        }

        for key in sae_activations:
            temp_dic[key] = torch.cat((
                sae_activations[key][positive_indices],
                sae_activations[key][negative_indices]
            ))

        samples_and_activations[concept_name] = temp_dic

    return samples_and_activations

def analyze_concepts(sae_activations: Dict[str, torch.Tensor],
                    samples_and_activations: Dict[str, Dict[str, torch.Tensor]],
                    move_dependent_header: List[str],
                    target_key_list: List[str],
                    num_cores: int = min(72, cpu_count()),
                    top_k: int = 20) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    results = {key: {} for key in target_key_list}

    for concept_name in move_dependent_header:
        print(f"\nEvaluating concept: {concept_name}")

        for layer_key in target_key_list:
            print(f"Processing {layer_key}")
            sampled_activations = samples_and_activations[concept_name][layer_key]
            ground_truth = torch.cat((
                samples_and_activations[concept_name]["positives"],
                samples_and_activations[concept_name]["negatives"]
            ))

            n_features = sampled_activations.shape[1]
            args_list = [(i, sampled_activations[:, i], ground_truth) 
                        for i in range(n_features)]

            with Pool(num_cores) as pool:
                feature_results = list(tqdm.tqdm(
                    pool.imap(process_sae_feature, args_list),
                    total=n_features
                ))

            feature_results.sort(key=lambda x: x[1], reverse=True)
            results[layer_key][concept_name] = feature_results[:top_k]

            top_feature = feature_results[0]
            print(f"Best feature: {top_feature[0]}, AUC: {top_feature[1]:.4f}")
            
            print(f"\nTop 5 features:")
            for i, (feature_index, auc) in enumerate(feature_results[:5], 1):
                print(f"{i}. Feature {feature_index}: AUC={auc:.4f}")

    return results

def select_best_features(results: Dict[str, List[Tuple[int, float]]], 
                        min_auc: float = 0.7) -> Dict[str, Dict[str, Tuple[int, float]]]:
    square_features = {}
    
    for concept_name, concept_results in results.items():
        square_name = concept_name[-2:]
        
        for feature_index, auc_score in concept_results[:5]:
            if auc_score > min_auc:
                square_features[square_name] = (feature_index, auc_score)
                break

    print(f"\nFound {len(square_features)} significant features")
    if square_features:
        print("\nSquare mappings:")
        for square, (feature_idx, auc) in square_features.items():
            print(f"{square}: Feature {feature_idx} (AUC: {auc:.3f})")
    
    return square_features

def main():
    test_dim = 8192
    sae_lr = 1
    sae_site = "res"
    
    with open(f'maia2-sae/activations/maia2_activations_for_sae_{test_dim}_{sae_lr}_{sae_site}.pickle', 'rb') as f:
        maia2_activations = pickle.load(f)

    target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states']
    
    move_dependent_header, function_map = initialize_concept_functions()
    sae_activations = maia2_activations['all_sae_activations']
    board_fens = maia2_activations['board_fen']
    all_moves = get_all_possible_moves()
    all_moves_dict = {i: move for i, move in enumerate(all_moves)}

    preds = [all_moves_dict[int(idx)] for idx in maia2_activations['preds']]
    
    samples_and_activations = cache_examples_and_activations(
        function_map, move_dependent_header, sae_activations, 
        board_fens, preds, move_dependent_header
    )
    
    results = analyze_concepts(
        sae_activations, samples_and_activations, 
        move_dependent_header, target_key_list,
        top_k=20
    )

    for layer_key in target_key_list:
        layer_name = 'layer6' if 'block 0' in layer_key else 'layer7'
        with open(f'maia2-sae/dataset/intervention/{layer_name}_squarewise_features.pickle', 'wb') as f:
            pickle.dump(results[layer_key], f)
    
    # layer6_features = select_best_features(results['transformer block 0 hidden states'])
    # layer7_features = select_best_features(results['transformer block 1 hidden states'])
    
    # with open('maia2-sae/dataset/intervention/layer6_squarewise_features.pickle', 'wb') as f:
    #     pickle.dump(layer6_features, f)
    
    # with open('maia2-sae/dataset/intervention/layer7_squarewise_features.pickle', 'wb') as f:
    #     pickle.dump(layer7_features, f)


if __name__ == "__main__":
    main()