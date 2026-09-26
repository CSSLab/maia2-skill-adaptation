import os
import sys
import pickle
import torch
import torch.nn as nn
import chess
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
import threading
import hashlib

ROOT = os.environ.get('SKILL_ADAPTATION_ROOT',
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from maia2.main import MAIA2Model
from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves, get_side_info

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_thread_local = threading.local()

def load_filtered_concepts(min_train_samples=500):
    summary_path = os.path.join(ROOT, 'dataset', 'concept-filtered-externalization', 'concept_counts_summary.pkl')
    with open(summary_path, 'rb') as f:
        data = pickle.load(f)

    filtered = {}
    for concept_name, counts in data.items():
        if counts.get('train', 0) >= min_train_samples:
            filtered[concept_name] = counts

    return filtered

def load_sae_features_for_concept(concept_name, features_path, layer_key):
    with open(features_path, 'r') as f:
        all_features = json.load(f)

    if concept_name not in all_features:
        return None

    if layer_key not in all_features[concept_name]:
        return None

    return all_features[concept_name][layer_key]

def load_model(model_path):
    elo_dict = create_elo_dict()
    all_moves = get_all_possible_moves()

    from argparse import Namespace
    cfg = Namespace(
        input_channels=18, dim_cnn=256, dim_vit=1024,
        num_blocks_cnn=5, num_blocks_vit=2, vit_length=8, elo_dim=128
    )

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model.to(DEVICE).eval(), elo_dict, all_moves, cfg

def load_sae(sae_path):
    sae_checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
    sae = sae_checkpoint['sae_state_dicts']
    return {k: {name: param.to(DEVICE) for name, param in v.items()} for k, v in sae.items()}

def enable_intervention_hooks(model, cfg):
    def get_hook(name):
        def hook(module, input, output):
            if not hasattr(_thread_local, 'activations'):
                _thread_local.activations = {}
            _thread_local.activations[name] = output.detach()

            if hasattr(_thread_local, 'interventions') and name in _thread_local.interventions:
                output = _thread_local.interventions[name]
            # residual stream after the block (post-intervention if steering is active), used by the probes
            if not hasattr(_thread_local, 'residual_streams'):
                _thread_local.residual_streams = {}
            _thread_local.residual_streams[name] = (output + input[0]).detach()
            return output
        return hook

    layer_names = []
    for i in range(cfg.num_blocks_vit):
        ff = model.transformer.elo_layers[i][1]
        layer_name = f'transformer block {i} hidden states'
        ff.register_forward_hook(get_hook(layer_name))
        layer_names.append(layer_name)

    return layer_names

def load_concept_data(concept_name, data_dir, split='train'):
    csv_path = os.path.join(data_dir, concept_name, f'{split}_moves.csv')
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)

def apply_sae_intervention(model, sae, boards, elos_self, elos_oppo, layer_name, feature_indices, strength):
    _thread_local.activations = {}

    with torch.no_grad():
        _ = model(boards, elos_self, elos_oppo)
        clean_acts = _thread_local.activations[layer_name].clone()

    act_pooled = torch.mean(clean_acts, dim=1)

    encoder_weight = sae[layer_name]['encoder_DF.weight']
    encoder_bias = sae[layer_name]['encoder_DF.bias']
    pre_activation = nn.functional.linear(act_pooled, encoder_weight, encoder_bias)

    thresholds = sae[layer_name]['threshold']
    sae_acts = pre_activation * (pre_activation >= thresholds.unsqueeze(0))

    modified_sae_acts = sae_acts.clone()
    for feat_idx in feature_indices:
        modified_sae_acts[:, feat_idx] *= strength

    decoder_weight = sae[layer_name]['decoder_FD.weight']
    decoder_bias = sae[layer_name]['decoder_FD.bias']
    reconstructed = nn.functional.linear(modified_sae_acts, decoder_weight, decoder_bias)

    intervened_acts = reconstructed.unsqueeze(1).expand(-1, clean_acts.size(1), -1)

    _thread_local.interventions = {layer_name: intervened_acts}

    with torch.no_grad():
        logits, _, _ = model(boards, elos_self, elos_oppo)

    if hasattr(_thread_local, 'interventions'):
        del _thread_local.interventions

    return logits

def get_cached_activations_and_labels(model, df, layer_name, all_moves_dict, batch_size=1024):
    cached_data = []

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]

        boards_list = []
        optimal_moves = []
        legal_moves_list = []

        for _, row in batch_df.iterrows():
            fen = row['fen']
            board = chess.Board(fen)
            boards_list.append(board_to_tensor(board))

            move_dict = eval(row['moves']) if isinstance(row['moves'], str) else row['moves']
            optimal_move = move_dict['10']
            optimal_moves.append(optimal_move)

            legal_moves, _ = get_side_info(board, optimal_move, all_moves_dict)
            legal_moves_list.append(legal_moves)

        boards = torch.stack(boards_list).to(DEVICE)
        legal_moves = torch.stack(legal_moves_list).to(DEVICE)

        activations_per_elo = {}
        for elo in range(11):
            _thread_local.activations = {}
            elos_self = torch.full((len(boards),), elo, device=DEVICE).long()
            elos_oppo = torch.full((len(boards),), elo, device=DEVICE).long()

            with torch.no_grad():
                _ = model(boards, elos_self, elos_oppo)
                activations_per_elo[elo] = _thread_local.activations[layer_name].clone().cpu()

        cached_data.append({
            'boards': boards.cpu(),
            'activations_per_elo': activations_per_elo,
            'optimal_moves': optimal_moves,
            'legal_moves': legal_moves.cpu()
        })

    return cached_data

def evaluate_with_cached_activations(model, sae, cached_data, elo, feature_indices, strength, layer_name, all_moves_dict, idx_to_move):
    correct = 0
    total = 0

    for batch_cache in cached_data:
        boards = batch_cache['boards'].to(DEVICE)
        clean_acts = batch_cache['activations_per_elo'][elo].to(DEVICE)
        legal_moves = batch_cache['legal_moves'].to(DEVICE)
        optimal_moves = batch_cache['optimal_moves']

        act_pooled = torch.mean(clean_acts, dim=1)

        encoder_weight = sae[layer_name]['encoder_DF.weight']
        encoder_bias = sae[layer_name]['encoder_DF.bias']
        pre_activation = nn.functional.linear(act_pooled, encoder_weight, encoder_bias)

        thresholds = sae[layer_name]['threshold']
        sae_acts = pre_activation * (pre_activation >= thresholds.unsqueeze(0))

        modified_sae_acts = sae_acts.clone()
        for feat_idx in feature_indices:
            modified_sae_acts[:, feat_idx] *= strength

        decoder_weight = sae[layer_name]['decoder_FD.weight']
        decoder_bias = sae[layer_name]['decoder_FD.bias']
        reconstructed = nn.functional.linear(modified_sae_acts, decoder_weight, decoder_bias)

        intervened_acts = reconstructed.unsqueeze(1).expand(-1, clean_acts.size(1), -1)

        _thread_local.interventions = {layer_name: intervened_acts}

        elos_self = torch.full((len(boards),), elo, device=DEVICE).long()
        elos_oppo = torch.full((len(boards),), elo, device=DEVICE).long()

        with torch.no_grad():
            logits, _, _ = model(boards, elos_self, elos_oppo)

        if hasattr(_thread_local, 'interventions'):
            del _thread_local.interventions

        probs = (logits * legal_moves).softmax(-1)

        for j, prob in enumerate(probs):
            pred_idx = prob.argmax().item()
            pred_move = idx_to_move[pred_idx]
            if pred_move == optimal_moves[j]:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0

def find_best_hyperparams(model, sae, concept_name, sae_features, train_df, layer_name, all_moves_dict, idx_to_move, batch_size=1024):
    top_k_values = [1, 2, 5, 10, 20]
    strength_values = [2.0, 5.0, 10.0, 20.0]

    print('    Caching activations for all ELOs...')
    cached_data = get_cached_activations_and_labels(model, train_df, layer_name, all_moves_dict, batch_size)

    best_params = {}

    total_iterations = 10 * len(top_k_values) * len(strength_values)
    pbar = tqdm(total=total_iterations, desc='  Hyperparameter search', leave=False)

    for elo in range(10):
        best_acc = 0.0
        best_k = None
        best_strength = None

        for k in top_k_values:
            feature_indices = sae_features['top_100'][:k]

            for strength in strength_values:
                acc = evaluate_with_cached_activations(
                    model, sae, cached_data, elo, feature_indices, strength,
                    layer_name, all_moves_dict, idx_to_move
                )

                if acc > best_acc:
                    best_acc = acc
                    best_k = k
                    best_strength = strength

                pbar.update(1)
                pbar.set_postfix({'elo': elo, 'best_acc': f'{best_acc:.4f}'})

        best_params[elo] = {
            'k': best_k,
            'strength': best_strength,
            'train_acc': best_acc
        }

    pbar.close()
    return best_params

def find_transition_point(predictions, labels, elo_levels):
    skill_results = {}
    for i, elo in enumerate(elo_levels):
        skill_results[elo] = (predictions[i] == labels[i]).item()

    if all(skill_results[elo] for elo in elo_levels):
        return 0
    if all(not skill_results[elo] for elo in elo_levels):
        return 10

    for level_idx in range(len(elo_levels) - 1):
        curr_level = elo_levels[level_idx]
        next_level = elo_levels[level_idx + 1]
        if not skill_results[curr_level] and skill_results[next_level]:
            if all(skill_results[elo_levels[i]] for i in range(level_idx + 1, len(elo_levels))):
                return next_level
    return -1

def evaluate_on_test(model, sae, sae_features, test_df, best_params, layer_name, all_moves_dict, idx_to_move, batch_size=1024):
    results_per_elo = {elo: {'correct': 0, 'total': 0} for elo in range(10)}
    all_transition_points = []

    test_fens = []
    baseline_predictions_per_elo = {elo: [] for elo in range(10)}
    intervention_predictions_per_elo = {elo: [] for elo in range(10)}

    for start_idx in tqdm(range(0, len(test_df), batch_size), desc='  Testing', leave=False):
        batch_df = test_df.iloc[start_idx:start_idx+batch_size]

        boards_list = []
        optimal_moves_list = []
        legal_moves_list = []
        fens_list = []

        for _, row in batch_df.iterrows():
            fen = row['fen']
            board = chess.Board(fen)
            boards_list.append(board_to_tensor(board))
            fens_list.append(fen)

            move_dict = eval(row['moves']) if isinstance(row['moves'], str) else row['moves']
            optimal_move = move_dict['10']
            optimal_moves_list.append(optimal_move)

            legal_moves, _ = get_side_info(board, optimal_move, all_moves_dict)
            legal_moves_list.append(legal_moves)

        boards = torch.stack(boards_list).to(DEVICE)
        legal_moves_batch = torch.stack(legal_moves_list).to(DEVICE)

        batch_baseline_predictions_all_elos = []
        batch_intervention_predictions_all_elos = []
        batch_labels_all_elos = []

        for elo in range(11):
            elos_self = torch.full((len(boards),), elo, device=DEVICE).long()
            elos_oppo = torch.full((len(boards),), elo, device=DEVICE).long()

            with torch.no_grad():
                logits_baseline, _, _ = model(boards, elos_self, elos_oppo)

            probs_baseline = (logits_baseline * legal_moves_batch).softmax(-1)
            pred_indices_baseline = probs_baseline.argmax(dim=1).cpu().tolist()
            batch_baseline_predictions_all_elos.append(pred_indices_baseline)

        for elo in range(11):
            params = best_params[elo] if elo < 10 else best_params[9]
            feature_indices = sae_features['top_100'][:params['k']]

            elos_self = torch.full((len(boards),), elo, device=DEVICE).long()
            elos_oppo = torch.full((len(boards),), elo, device=DEVICE).long()

            logits = apply_sae_intervention(
                model, sae, boards, elos_self, elos_oppo, layer_name,
                feature_indices, params['strength']
            )

            probs = (logits * legal_moves_batch).softmax(-1)
            pred_indices = probs.argmax(dim=1).cpu().tolist()
            batch_intervention_predictions_all_elos.append(pred_indices)

            if elo == 0:
                batch_labels_all_elos = [all_moves_dict[move] for move in optimal_moves_list]

        test_fens.extend(fens_list)

        for pos_idx in range(len(boards_list)):
            predictions_per_elo = [batch_intervention_predictions_all_elos[elo][pos_idx] for elo in range(11)]
            baseline_preds_per_elo = [batch_baseline_predictions_all_elos[elo][pos_idx] for elo in range(11)]
            labels_per_elo = [batch_labels_all_elos[pos_idx]] * 11

            for elo in range(10):
                baseline_predictions_per_elo[elo].append(idx_to_move[baseline_preds_per_elo[elo]])
                intervention_predictions_per_elo[elo].append(idx_to_move[predictions_per_elo[elo]])

            for elo in range(10):
                is_correct = (predictions_per_elo[elo] == labels_per_elo[elo])
                results_per_elo[elo]['total'] += 1
                if is_correct:
                    results_per_elo[elo]['correct'] += 1

            tp = find_transition_point(
                torch.tensor(baseline_preds_per_elo),
                torch.tensor(labels_per_elo),
                list(range(11))
            )

            if tp != -1:
                all_transition_points.append(tp)

    test_accs = {}
    for elo in range(10):
        if results_per_elo[elo]['total'] > 0:
            test_accs[elo] = results_per_elo[elo]['correct'] / results_per_elo[elo]['total']
        else:
            test_accs[elo] = 0.0

    avg_tp = sum(all_transition_points) / len(all_transition_points) if all_transition_points else -1
    num_transitional = len(all_transition_points)

    return test_accs, avg_tp, num_transitional, test_fens, baseline_predictions_per_elo, intervention_predictions_per_elo

def load_pretrained_probes(concept_name, probe_base_dir):
    layer0_path = os.path.join(probe_base_dir, 'layer0_efficient', f'{concept_name}_probes.pkl')
    layer1_path = os.path.join(probe_base_dir, 'layer1_efficient', f'{concept_name}_probes.pkl')

    if not os.path.exists(layer0_path) or not os.path.exists(layer1_path):
        return None

    with open(layer0_path, 'rb') as f:
        layer0_data = pickle.load(f)
    with open(layer1_path, 'rb') as f:
        layer1_data = pickle.load(f)

    return {'layer0': layer0_data, 'layer1': layer1_data}

def get_activations_for_df(model, df, elo_category, device, layer_key,
                           sae=None, steer_layer_name=None, feature_indices=None, strength=None):
    """Mean-pooled residual stream after `layer_key` for every position in `df`.
    When `sae`, `steer_layer_name`, `feature_indices` and `strength` are given, the
    activations are collected with SAE feature steering applied at `steer_layer_name`."""
    boards_list = []
    for _, row in df.iterrows():
        fen = row['fen']
        board = chess.Board(fen)
        board_input = board_to_tensor(board)
        boards_list.append(board_input)

    boards = torch.stack(boards_list).to(device)
    elos_self = torch.tensor([elo_category] * len(df)).to(device)
    elos_oppo = torch.tensor([elo_category] * len(df)).to(device)

    _thread_local.residual_streams = {}
    if sae is not None and steer_layer_name is not None:
        apply_sae_intervention(model, sae, boards, elos_self, elos_oppo,
                               steer_layer_name, feature_indices, strength)
    else:
        with torch.no_grad():
            logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)

    activations = _thread_local.residual_streams[layer_key]
    activations = torch.mean(activations, dim=1)
    return activations.cpu().numpy()

def _probe_predict(probe_state, X, device):
    """Apply a saved linear probe (state dict with 'weight' and 'bias') to features X.
    A single-logit probe (as trained by train/train_probes.py) is thresholded at 0;
    a two-logit probe uses argmax."""
    weight = torch.as_tensor(probe_state['weight']).float().to(device)
    bias = torch.as_tensor(probe_state['bias']).float().to(device)
    X_tensor = torch.as_tensor(X).float().to(device)
    logits = torch.nn.functional.linear(X_tensor, weight, bias)
    if logits.shape[1] == 1:
        return (logits[:, 0] > 0).long().cpu().numpy()
    return torch.argmax(logits, dim=1).cpu().numpy()

def evaluate_with_pretrained_probes(concept_name, model, device, cfg, probe_base_dir, layer_names,
                                    sae=None, sae_features=None, best_params=None, steer_layer_name=None):
    """Evaluate the probes trained before steering on the steered representations.
    Each skill condition uses the (k, strength) selected for it on the training split;
    the highest condition reuses the setting of the adjacent one."""
    probe_data = load_pretrained_probes(concept_name, probe_base_dir)
    if probe_data is None:
        print(f"  Warning: Pretrained probes not found for {concept_name}")
        return None

    concept_positions_csv = os.path.join(ROOT, 'concept_positions', f'{concept_name}.csv')
    if not os.path.exists(concept_positions_csv):
        print(f"  Warning: Concept positions not found for {concept_name}")
        return None

    concept_df = pd.read_csv(concept_positions_csv)
    if len(concept_df) < 100:
        print(f"  Warning: Insufficient concept data for {concept_name} ({len(concept_df)} samples)")
        return None

    n_pos = (concept_df['label'] == 1).sum()
    n_neg = (concept_df['label'] == 0).sum()
    print(f"  Probe data: {len(concept_df)} positions ({n_pos} pos, {n_neg} neg)")

    results_layer0 = {}
    results_layer1 = {}

    print(f"  Evaluating pretrained probes for 11 ELO levels...")
    for elo_category in tqdm(range(11), desc="  ELO levels", leave=False):
        if sae is not None and best_params is not None:
            params = best_params[elo_category] if elo_category < 10 else best_params[9]
            feature_indices = sae_features['top_100'][:params['k']]
            steer_kwargs = dict(sae=sae, steer_layer_name=steer_layer_name,
                                feature_indices=feature_indices, strength=params['strength'])
        else:
            steer_kwargs = {}
        X_test_l0 = get_activations_for_df(model, concept_df, elo_category, device, layer_names[0], **steer_kwargs)
        X_test_l1 = get_activations_for_df(model, concept_df, elo_category, device, layer_names[1], **steer_kwargs)
        y_test = concept_df['label'].values

        probe_weights_l0 = probe_data['layer0']['probe_weights_per_elo'][elo_category]
        probe_weights_l1 = probe_data['layer1']['probe_weights_per_elo'][elo_category]

        with torch.no_grad():
            predictions_l0 = _probe_predict(probe_weights_l0, X_test_l0, device)
            acc_l0 = np.mean(predictions_l0 == y_test)

            predictions_l1 = _probe_predict(probe_weights_l1, X_test_l1, device)
            acc_l1 = np.mean(predictions_l1 == y_test)

        results_layer0[elo_category] = acc_l0
        results_layer1[elo_category] = acc_l1

    return {'layer0': results_layer0, 'layer1': results_layer1, 'n_test': len(concept_df)}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--mode', type=str, default='random', choices=['salient', 'random'])
    parser.add_argument('--min_train_samples', type=int, default=500)
    args = parser.parse_args()

    layer_idx = args.layer
    mode = args.mode

    print('='*80)
    print(f'SAE-Based Feature Intervention Experiment (Layer {layer_idx}, Mode: {mode})')
    print('='*80)

    model_path = os.path.join(ROOT, 'weights.v2.pt')
    sae_path = os.path.join(ROOT, 'sae', 'best_jrsaes_2023-11-16384-1-res.pt')
    features_path = os.path.join(ROOT, 'extern', 'feature_steering', 'sae_feature_selection_results.json')
    data_dir = os.path.join(ROOT, 'dataset', 'concept-filtered-externalization')
    probe_base_dir = os.path.join(ROOT, 'probes')
    output_dir = os.path.join(ROOT, 'extern', 'feature_steering', 'results', f'sae_results_layer{layer_idx}_{mode}')

    os.makedirs(output_dir, exist_ok=True)

    print(f'\nLoading model from: {model_path}')
    model, elo_dict, all_moves, cfg = load_model(model_path)
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    idx_to_move = {i: move for i, move in enumerate(all_moves)}

    print(f'\nLoading SAE from: {sae_path}')
    sae = load_sae(sae_path)

    layer_names = enable_intervention_hooks(model, cfg)
    layer_name = layer_names[layer_idx]
    print(f'Intervention hook enabled at: {layer_name}')

    print(f'\nLoading filtered concepts (>={args.min_train_samples} train samples)...')
    filtered_concepts = load_filtered_concepts(min_train_samples=args.min_train_samples)
    print(f'Found {len(filtered_concepts)} filtered concepts')

    all_results = {}

    for concept_idx, concept_name in enumerate(filtered_concepts.keys()):
        print(f'\n{"="*80}')
        print(f'Concept {concept_idx+1}/{len(filtered_concepts)}: {concept_name}')
        print(f'{"="*80}')

        sae_features = load_sae_features_for_concept(concept_name, features_path, layer_name)
        if sae_features is None:
            print(f'  SAE features not found, skipping...')
            continue

        train_df = load_concept_data(concept_name, data_dir, 'train')
        test_df = load_concept_data(concept_name, data_dir, 'test')

        if train_df is None or test_df is None:
            print(f'  Data not found, skipping...')
            continue

        print(f'  Train samples: {len(train_df)}, Test samples: {len(test_df)}')

        if mode == 'random':
            num_features = len(sae_features['top_100'])
            concept_hash = int(hashlib.md5(concept_name.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(concept_hash)
            dict_size = sae[layer_name]['encoder_DF.weight'].shape[0]
            random_indices = np.random.choice(dict_size, size=num_features, replace=False).tolist()
            sae_features = {'top_100': random_indices}

        print(f'\n  Finding best hyperparameters on train set...')
        best_params = find_best_hyperparams(
            model, sae, concept_name, sae_features, train_df, layer_name, all_moves_dict, idx_to_move
        )

        print(f'\n  Best params per ELO:')
        for elo in range(10):
            p = best_params[elo]
            print(f'    ELO {elo}: k={p["k"]}, strength={p["strength"]:.1f}, train_acc={p["train_acc"]:.4f}')

        print(f'\n  Evaluating on test set...')
        test_accs, avg_tp, num_transitional, test_fens, baseline_preds, intervention_preds = evaluate_on_test(
            model, sae, sae_features, test_df, best_params, layer_name, all_moves_dict, idx_to_move
        )

        print(f'\n  Test results per ELO:')
        for elo in range(10):
            p = best_params[elo]
            acc = test_accs[elo]
            print(f'    ELO {elo}: test_acc={acc:.4f} (k={p["k"]}, strength={p["strength"]:.1f})')

        print(f'\n  Avg transition point (with 0,10): {avg_tp:.2f} ({num_transitional}/{len(test_df)} transitional)')

        print(f'\n  Evaluating pretrained probes on steered representations...')
        probe_results = evaluate_with_pretrained_probes(
            concept_name, model, DEVICE, cfg, probe_base_dir, layer_names,
            sae=sae, sae_features=sae_features, best_params=best_params, steer_layer_name=layer_name
        )

        if probe_results is not None:
            print(f'\n  Probe test accuracies (layer 0):')
            for elo in range(10):
                print(f'    ELO {elo}: {probe_results["layer0"][elo]:.4f}')
            print(f'\n  Probe test accuracies (layer 1):')
            for elo in range(10):
                print(f'    ELO {elo}: {probe_results["layer1"][elo]:.4f}')

        all_results[concept_name] = {
            'best_params': best_params,
            'test_accs': test_accs,
            'avg_transition_point': avg_tp,
            'num_transitional_samples': num_transitional,
            'num_test_samples': len(test_df),
            'probe_layer0': probe_results['layer0'] if probe_results else None,
            'probe_layer1': probe_results['layer1'] if probe_results else None,
            'test_fens': test_fens,
            'baseline_predictions_per_elo': baseline_preds,
            'intervention_predictions_per_elo': intervention_preds,
        }

        results_path = os.path.join(output_dir, f'{concept_name}_sae_intervention_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(all_results[concept_name], f)

    print(f'\n{"="*80}')
    print('Summary')
    print(f'{"="*80}')

    print(f'\nTotal concepts processed: {len(all_results)}')

    weighted_accs_per_elo = {elo: [] for elo in range(10)}
    weighted_tps = []
    weights_acc = {elo: [] for elo in range(10)}
    weights_tp = []

    for concept_name, results in all_results.items():
        num_samples = results['num_test_samples']

        for elo in range(10):
            weighted_accs_per_elo[elo].append(results['test_accs'][elo])
            weights_acc[elo].append(num_samples)

        if results['avg_transition_point'] != -1 and results['num_transitional_samples'] > 0:
            weighted_tps.append(results['avg_transition_point'])
            weights_tp.append(results['num_transitional_samples'])

    print(f'\nWeighted average test accuracy per ELO across all concepts:')
    for elo in range(10):
        if weighted_accs_per_elo[elo]:
            avg_acc = np.average(weighted_accs_per_elo[elo], weights=weights_acc[elo])
            std_acc = np.sqrt(np.average((np.array(weighted_accs_per_elo[elo]) - avg_acc)**2, weights=weights_acc[elo]))
            print(f'  ELO {elo}: {avg_acc:.4f} ± {std_acc:.4f}')

    if weighted_tps:
        overall_avg_tp = np.average(weighted_tps, weights=weights_tp)
        overall_std_tp = np.sqrt(np.average((np.array(weighted_tps) - overall_avg_tp)**2, weights=weights_tp))
        print(f'\nWeighted average transition point across all concepts: {overall_avg_tp:.2f} ± {overall_std_tp:.2f}')
    else:
        print(f'\nNo transitional samples found')

    results_with_probes = [r for r in all_results.values() if r['probe_layer0'] is not None]
    if len(results_with_probes) > 0:
        avg_probe_layer0 = {elo: 0.0 for elo in range(11)}
        avg_probe_layer1 = {elo: 0.0 for elo in range(11)}
        for elo in range(11):
            avg_probe_layer0[elo] = np.mean([r['probe_layer0'][elo] for r in results_with_probes])
            avg_probe_layer1[elo] = np.mean([r['probe_layer1'][elo] for r in results_with_probes])

        print(f'\n{"="*80}')
        print(f'PRETRAINED PROBE ACCURACIES (avg across {len(results_with_probes)} concepts)')
        print(f'{"="*80}')
        print(f'\nLayer 0:')
        for elo in range(10):
            print(f'  ELO {elo}: {avg_probe_layer0[elo]:.4f}')

        print(f'\nLayer 1:')
        for elo in range(10):
            print(f'  ELO {elo}: {avg_probe_layer1[elo]:.4f}')

    summary_path = os.path.join(output_dir, 'sae_intervention_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f'\nResults saved to: {output_dir}')
    print('Done!')

if __name__ == '__main__':
    main()
