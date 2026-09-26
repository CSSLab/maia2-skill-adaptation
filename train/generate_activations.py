import os
import sys
import torch
import pickle
import threading
import chess
import json
import random
import argparse
import pandas as pd

ROOT = os.environ.get('SKILL_ADAPTATION_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from maia2.main import MAIA2Model
from maia2.utils import get_all_possible_moves, create_elo_dict, board_to_tensor, map_to_category

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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', default=100000, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--model', default='ViT', type=str)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=2, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=8, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    parser.add_argument('--data_path', default=os.path.join(ROOT, 'dataset', 'blundered-transitional-dataset', 'train_moves.csv'), type=str)
    parser.add_argument('--model_path', default=os.path.join(ROOT, 'weights.v2.pt'), type=str)
    parser.add_argument('--output_path', default=os.path.join(ROOT, 'activations', 'maia2_activations.pickle'), type=str)
    parser.add_argument('--use_csv', default=True, type=bool)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    return parser.parse_args(args)

def load_fens_with_metadata(data_path, num_samples, use_csv=True):
    if use_csv:
        df = pd.read_csv(data_path)
        total_available = len(df)
        num_samples = min(num_samples, total_available)

        random.seed(42)
        selected_indices = random.sample(range(total_available), num_samples)
        selected_df = df.iloc[selected_indices].reset_index(drop=True)

        print(f"Loaded {num_samples} FENs randomly sampled from {total_available} blundered-transitional positions")
        print(f"Transition point distribution: {selected_df['transition_point'].value_counts().sort_index().to_dict()}")
        return selected_df
    else:
        with open(data_path, 'r') as f:
            all_fens = json.load(f)

        total_available = len(all_fens)
        num_samples = min(num_samples, total_available)

        random.seed(42)
        selected_indices = random.sample(range(total_available), num_samples)
        selected_fens = [all_fens[i] for i in selected_indices]

        print(f"Loaded {num_samples} FENs randomly sampled from {total_available} total positions")
        return selected_fens

def process_batch(data, model, all_moves_dict, elo_dict, cfg, elo_category, use_csv=True):
    boards_list = []
    elos_self_list = []
    elos_oppo_list = []
    legal_moves_list = []
    board_fens_list = []
    metadata_list = []

    if use_csv:
        for _, row in data.iterrows():
            fen = row['fen']
            board = chess.Board(fen)

            board_input = board_to_tensor(board)
            boards_list.append(board_input)
            board_fens_list.append(fen)

            elos_self_list.append(elo_category)
            elos_oppo_list.append(elo_category)

            legal_moves = torch.zeros(len(all_moves_dict))
            legal_moves_idx = torch.tensor([all_moves_dict[move.uci()] for move in board.legal_moves])
            legal_moves[legal_moves_idx] = 1
            legal_moves_list.append(legal_moves)

            metadata_list.append({
                'correct_move': row['correct_move'],
                'transition_point': row['transition_point'],
                'is_blunder': row['is_blunder']
            })
    else:
        for fen in data:
            original_board = chess.Board(fen)

            if original_board.turn == chess.WHITE:
                board = original_board
            else:
                board = original_board.mirror()

            board_input = board_to_tensor(board)
            boards_list.append(board_input)
            board_fens_list.append(fen)

            elos_self_list.append(elo_category)
            elos_oppo_list.append(elo_category)

            legal_moves = torch.zeros(len(all_moves_dict))
            legal_moves_idx = torch.tensor([all_moves_dict[move.uci()] for move in board.legal_moves])
            legal_moves[legal_moves_idx] = 1
            legal_moves_list.append(legal_moves)

    boards = torch.stack(boards_list).to(cfg.device)
    elos_self = torch.tensor(elos_self_list).to(cfg.device)
    elos_oppo = torch.tensor(elos_oppo_list).to(cfg.device)
    legal_moves = torch.stack(legal_moves_list).to(cfg.device)

    _thread_local.residual_streams = {}

    with torch.no_grad():
        logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)

    activations = {}
    for key, val in _thread_local.residual_streams.items():
        activations[key] = torch.mean(val, dim=1)
    logits_maia_legal = logits_maia * legal_moves
    preds = logits_maia_legal.argmax(dim=-1)

    result = {
        'board_fens': board_fens_list,
        'logits_maia': logits_maia.clone().detach().cpu(),
        'logits_value': logits_value.clone().detach().cpu(),
        'preds': preds.clone().detach().cpu(),
        'elos_self': elos_self.cpu(),
        'elos_oppo': elos_oppo.cpu(),
        'activations': {k: v.clone().detach().cpu() for k, v in activations.items()},
    }

    if use_csv and metadata_list:
        result['metadata'] = metadata_list

    return result

def main():
    args = parse_args()

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    device = torch.device(args.device)
    ckpt = torch.load(args.model_path, map_location=device)

    model = MAIA2Model(len(all_moves), elo_dict, args)

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

    _enable_activation_hook(model, args)

    data = load_fens_with_metadata(args.data_path, args.num_samples, use_csv=args.use_csv)

    all_board_fens = []
    all_metadata = []

    all_activations_per_elo = {elo: {} for elo in range(11)}
    all_logits_maia_per_elo = {elo: [] for elo in range(11)}
    all_logits_value_per_elo = {elo: [] for elo in range(11)}
    all_preds_per_elo = {elo: [] for elo in range(11)}

    data_len = len(data) if args.use_csv else len(data)
    num_batches = (data_len + args.batch_size - 1) // args.batch_size
    print(f"Processing {data_len} positions across 11 ELO categories (0-10)")
    print(f"Batch size: {args.batch_size}, Total batches per ELO: {num_batches}")

    for elo_category in range(11):
        print(f"\n{'='*80}")
        print(f"Processing ELO category {elo_category}/10")
        print(f"{'='*80}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, data_len)

            if args.use_csv:
                batch_data = data.iloc[start_idx:end_idx]
            else:
                batch_data = data[start_idx:end_idx]

            print(f"  Batch {batch_idx + 1}/{num_batches} ({len(batch_data)} positions)", end='\r')

            batch_results = process_batch(batch_data, model, all_moves_dict, elo_dict, args,
                                        elo_category=elo_category, use_csv=args.use_csv)

            if elo_category == 0:
                all_board_fens.extend(batch_results['board_fens'])
                if 'metadata' in batch_results:
                    all_metadata.extend(batch_results['metadata'])

            all_logits_maia_per_elo[elo_category].append(batch_results['logits_maia'])
            all_logits_value_per_elo[elo_category].append(batch_results['logits_value'])
            all_preds_per_elo[elo_category].append(batch_results['preds'])

            for key, val in batch_results['activations'].items():
                if key not in all_activations_per_elo[elo_category]:
                    all_activations_per_elo[elo_category][key] = []
                all_activations_per_elo[elo_category][key].append(val)

        print()

    print("\nConcatenating results per ELO category...")
    for elo_category in range(11):
        all_activations_per_elo[elo_category] = {
            k: torch.cat(v, dim=0) for k, v in all_activations_per_elo[elo_category].items()
        }
        all_logits_maia_per_elo[elo_category] = torch.cat(all_logits_maia_per_elo[elo_category], dim=0)
        all_logits_value_per_elo[elo_category] = torch.cat(all_logits_value_per_elo[elo_category], dim=0)
        all_preds_per_elo[elo_category] = torch.cat(all_preds_per_elo[elo_category], dim=0)

    data_to_save = {
        'board_fen': all_board_fens,
        'all_maia2_activations_per_elo': all_activations_per_elo,
        'logits_maia_per_elo': all_logits_maia_per_elo,
        'logits_value_per_elo': all_logits_value_per_elo,
        'preds_per_elo': all_preds_per_elo,
    }

    if all_metadata:
        data_to_save['metadata'] = all_metadata

    print(f"Saving activations to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"\n{'='*80}")
    print(f"Total positions saved: {len(all_board_fens)}")
    print(f"Per-ELO activation shapes:")
    for elo_category in range(11):
        print(f"\n  ELO category {elo_category}:")
        for key, val in data_to_save['all_maia2_activations_per_elo'][elo_category].items():
            print(f"    {key}: {val.shape}")
        print(f"    logits_maia: {data_to_save['logits_maia_per_elo'][elo_category].shape}")
        print(f"    logits_value: {data_to_save['logits_value_per_elo'][elo_category].shape}")
        print(f"    preds: {data_to_save['preds_per_elo'][elo_category].shape}")

if __name__ == "__main__":
    main()
