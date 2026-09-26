import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import json
import threading

ROOT = os.environ.get('SKILL_ADAPTATION_ROOT',
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from maia2.main import MAIA2Model
from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves
import chess

_thread_local = threading.local()

def _enable_activation_hook(model, cfg):
    def get_activation(name):
        def hook(model, input, output):
            if not hasattr(_thread_local, 'residual_streams'):
                _thread_local.residual_streams = {}
            _thread_local.residual_streams[name] = output.detach()
        return hook

    for i in range(cfg.num_blocks_vit):
        feedforward_module = model.transformer.elo_layers[i][1]
        feedforward_module.register_forward_hook(get_activation(f'transformer block {i} hidden states'))

def apply_sae_to_activations(sae, activations, layer_key, device):
    act = activations.to(device)
    encoder_weight = sae[layer_key]['encoder_DF.weight'].to(device)
    encoder_bias = sae[layer_key]['encoder_DF.bias'].to(device)
    pre_activation = nn.functional.linear(act, encoder_weight, encoder_bias)
    thresholds = sae[layer_key]['threshold'].to(device)
    encoded = pre_activation * (pre_activation >= thresholds.unsqueeze(0))
    return encoded.cpu()

def get_activations_for_fens(model, fens, elo_category, batch_size, all_moves_dict, elo_dict, cfg, device):
    all_acts = {}

    for i in range(0, len(fens), batch_size):
        batch_fens = fens[i:i+batch_size]
        boards_list = []
        elos_self_list = []
        elos_oppo_list = []

        for fen in batch_fens:
            board = chess.Board(fen)
            board_input = board_to_tensor(board)
            boards_list.append(board_input)
            elos_self_list.append(elo_category)
            elos_oppo_list.append(elo_category)

        boards = torch.stack(boards_list).to(device)
        elos_self = torch.tensor(elos_self_list).to(device)
        elos_oppo = torch.tensor(elos_oppo_list).to(device)

        _thread_local.residual_streams = {}

        with torch.no_grad():
            logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)

        batch_acts = {}
        for key, val in _thread_local.residual_streams.items():
            batch_acts[key] = torch.mean(val, dim=1).cpu()

        for key in batch_acts:
            if key not in all_acts:
                all_acts[key] = []
            all_acts[key].append(batch_acts[key])

    return {k: torch.cat(v, dim=0) for k, v in all_acts.items()}

def compute_feature_metrics(pos_acts, neg_acts):
    n_features = pos_acts.shape[1]
    aucs = []
    f1s = []

    for feat_idx in range(n_features):
        pos_vals = pos_acts[:, feat_idx].numpy()
        neg_vals = neg_acts[:, feat_idx].numpy()

        if pos_vals.std() == 0 and neg_vals.std() == 0:
            aucs.append(0.5)
            f1s.append(0.0)
            continue

        y_true = np.concatenate([np.ones(len(pos_vals)), np.zeros(len(neg_vals))])
        y_score = np.concatenate([pos_vals, neg_vals])

        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = 0.5

        threshold = np.median(y_score)
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        aucs.append(auc)
        f1s.append(f1)

    return np.array(aucs), np.array(f1s)

class Config:
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    cfg = Config()
    device = torch.device(cfg.device)
    batch_size = 1024
    elo_category = 5

    print(f"Loading Maia-2 model...")
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    ckpt = torch.load(os.path.join(ROOT, 'weights.v2.pt'), map_location=device, weights_only=False)

    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    _enable_activation_hook(model, cfg)

    print(f"Loading SAE...")
    sae_checkpoint = torch.load(os.path.join(ROOT, 'sae', 'best_jrsaes_2023-11-16384-1-res.pt'), map_location='cpu', weights_only=False)
    sae = sae_checkpoint['sae_state_dicts']
    layer_keys = list(sae.keys())

    print(f"SAE layers: {layer_keys}")

    concept_dir = os.path.join(ROOT, 'concept_positions')
    concept_files = sorted([f for f in os.listdir(concept_dir) if f.endswith('.csv')])

    print(f"\nTotal concepts with positions: {len(concept_files)}\n")

    results = {}

    for concept_idx, concept_file in enumerate(concept_files, 1):
        concept_name = concept_file.replace('.csv', '')

        print("=" * 80)
        print(f"Concept {concept_idx}/{len(concept_files)}: {concept_name}")
        print("=" * 80)

        df = pd.read_csv(os.path.join(concept_dir, concept_file))

        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]

        if len(pos_df) == 0 or len(neg_df) == 0:
            print(f"Skipping (insufficient samples)")
            continue

        pos_fens = pos_df['fen'].tolist()
        neg_fens = neg_df['fen'].tolist()

        print(f"Positive samples: {len(pos_fens)}, Negative samples: {len(neg_fens)}")

        pos_acts_maia2 = get_activations_for_fens(model, pos_fens, elo_category, batch_size,
                                                   all_moves_dict, elo_dict, cfg, device)
        neg_acts_maia2 = get_activations_for_fens(model, neg_fens, elo_category, batch_size,
                                                   all_moves_dict, elo_dict, cfg, device)

        results[concept_name] = {}

        for layer_idx, layer_key in enumerate(layer_keys):
            print(f"Layer {layer_idx}:")
            print(f"  Encoding with SAE...")

            pos_sae = apply_sae_to_activations(sae, pos_acts_maia2[layer_key], layer_key, device)
            neg_sae = apply_sae_to_activations(sae, neg_acts_maia2[layer_key], layer_key, device)

            print(f"  SAE features shape: {pos_sae.shape}")
            print(f"  Computing feature saliency...")

            aucs, f1s = compute_feature_metrics(pos_sae, neg_sae)

            combined_scores = aucs + f1s
            top_100_indices = np.argsort(combined_scores)[::-1][:100].tolist()
            top_5_indices = top_100_indices[:5]

            results[concept_name][layer_key] = {
                'top_100': top_100_indices,
                'top_100_aucs': aucs[top_100_indices].tolist(),
                'top_100_f1s': f1s[top_100_indices].tolist()
            }

            print(f"\n  Top 5 features by AUC:")
            for rank, feat_idx in enumerate(top_5_indices, 1):
                print(f"    {rank}. Feature {feat_idx:5d}: AUC={aucs[feat_idx]:.4f}, F1={f1s[feat_idx]:.4f}")
            print()

    results_path = os.path.join(ROOT, 'extern', 'feature_steering', 'sae_feature_selection_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
