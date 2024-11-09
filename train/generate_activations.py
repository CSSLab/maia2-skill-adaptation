import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import threading
import chess
from torch.utils.data import DataLoader
import argparse
from ..maia2.main import *
from ..maia2.utils import get_all_possible_moves, create_elo_dict

_thread_local = threading.local()

def get_chunks(pgn_path, chunk_size):
    chunk_count = 0
    chunks = []
    
    with open(pgn_path) as pgn_file:
        while True:
            start_pos = pgn_file.tell()
            game_count = 0
            
            while game_count < chunk_size:
                line = pgn_file.readline()
                if not line:
                    break
                    
                if line[-4:] == "1-0\n" or line[-4:] == "0-1\n":
                    game_count += 1
                if line[-8:] == "1/2-1/2\n":
                    game_count += 1
                if line[-2:] == "*\n":
                    game_count += 1
                    
            line = pgn_file.readline()
            if line not in ["\n", ""]:
                raise ValueError
                
            end_pos = pgn_file.tell()
            chunks.append((start_pos, end_pos))
            chunk_count += 1
            
            if chunk_count >= 100:
                break
            if not line:
                break
                
    return chunks

class MAIA2Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, all_moves_dict, cfg):
        
        self.all_moves_dict = all_moves_dict
        self.data = data
        self.cfg = cfg
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        board_fen, move_uci, elo_self, elo_oppo, active_win = self.data[idx]
        
        board = chess.Board(board_fen)
        board_input = board_to_tensor(board)
        
        legal_moves, side_info = get_side_info(board, move_uci, self.all_moves_dict)
        
        move_input = self.all_moves_dict[move_uci]
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info, active_win, board_fen

def apply_sae_to_activations(sae, activations, target_key_list):
    sae_activations = {}
    for key in target_key_list:
        act = torch.mean(activations[key], dim=1)
        pre_activation = nn.functional.linear(act, sae[key]['encoder_DF.weight'], sae[key]['encoder_DF.bias'])

        if 'threshold' in sae[key]:
            thresholds = sae[key]['threshold']
            encoded = pre_activation * (pre_activation >= thresholds.unsqueeze(0))
        else:
            encoded = nn.functional.relu(pre_activation)
            
        sae_activations[key] = encoded
    
    return sae_activations

def _enable_activation_hook(model, cfg):
    def get_activation(name):
        def hook(model, input, output):
            if not hasattr(_thread_local, 'residual_streams'):
                _thread_local.residual_streams = {}
            _thread_local.residual_streams[name] = output.detach()
        return hook
        
    def get_attention_head(name):
        def hook(module, input, output):
            if not hasattr(_thread_local, 'attention_heads'):
                _thread_local.attention_heads = {}
            batch_size, seq_len, _ = output.shape
            # 16 attention heads. each 64 dim
            reshaped_output = output.view(batch_size, seq_len, 16, 64)
            for head_idx in range(16):
                head_activation = reshaped_output[:, :, head_idx, :]
                _thread_local.attention_heads[f'{name}_head_{head_idx}'] = head_activation.detach()
        return hook

    def get_mlp_output(name):
        def hook(module, input, output):
            if not hasattr(_thread_local, 'mlp_outputs'):
                _thread_local.mlp_outputs = {}
            _thread_local.mlp_outputs[name] = output.detach()
        return hook
        
    for i in range(cfg.num_blocks_vit):
        if cfg.sae_residual_streams:
            feedforward_module = model.module.transformer.elo_layers[i][1]
            feedforward_module.register_forward_hook(get_activation(f'transformer block {i} hidden states'))
        if cfg.sae_attention_heads:
            attention_module = model.module.transformer.elo_layers[i][0]
            attention_module.register_forward_hook(get_attention_head(f'transformer block {i} attention heads'))
        if cfg.sae_mlp_outputs:
            mlp_module = model.module.transformer.elo_layers[i][1].net
            mlp_module.register_forward_hook(get_mlp_output(f'transformer block {i} MLP outputs'))

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='maia2_sae/pgn', type=str)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--max_games_per_elo_range', default=20, type=int)
    parser.add_argument('--first_n_moves', default=10, type=int)
    parser.add_argument('--last_n_moves', default=10, type=int)
    parser.add_argument('--max_ply', default=300, type=int)
    parser.add_argument('--clock_threshold', default=30, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--chunk_size', default=8000, type=int)
    parser.add_argument('--test_year', default=2023, type=int)
    parser.add_argument('--test_month', default=12, type=int)
    parser.add_argument('--model', default='ViT', type=str)
    parser.add_argument('--batch_size', default=30000, type=int)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=2, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=8, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    parser.add_argument('--sae_dim', default=8192, type=int)

    parser.add_argument('--sae_attention_heads', default=False, type=bool)
    parser.add_argument('--sae_residual_streams', default=True, type=bool)
    parser.add_argument('--sae_mlp_outputs', default=False, type=bool)
    
    return parser.parse_args(args)

def main():
    args = parse_args()

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    move_dict = {v: k for k, v in all_moves_dict.items()}
    
    trained_model_path = "maia2_sae/weights.v2.pt"
    ckpt = torch.load(trained_model_path, map_location=torch.device('cpu'))
    model = MAIA2Model(len(all_moves), elo_dict, args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states']
    formatted_month = f"{args.test_month:02d}"
    pgn_path = os.path.join(args.data_root, f"lichess_db_standard_rated_{args.test_year}-{formatted_month}.pgn")
    pgn_chunks = get_chunks(pgn_path, args.chunk_size)
    
    print(f'Testing Mixed Elo with {len(pgn_chunks)} chunks from {pgn_path}: ', flush=True)

    for i in range(0, len(pgn_chunks), args.num_workers):
        pgn_chunks_sublist = pgn_chunks[i:i + args.num_workers]
        data, game_count, chunk_count = process_chunks(args, pgn_path, pgn_chunks_sublist, elo_dict)
        
        dataset = MAIA2Dataset(data, all_moves_dict, args)
        dataloader = DataLoader(dataset, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              drop_last=False,
                              num_workers=args.num_workers)
        break

    sae_lr = 1
    sae_site = "res"
    sae_date = "2023-11"
    sae = torch.load(f'maia2-sae/sae/best_jrsaes_{sae_date}-{args.sae_dim}-{sae_lr}-{sae_site}.pt')['sae_state_dicts']
        
    _enable_activation_hook(model, args)
    
    logits_value_list = {i: [] for i in range(len(elo_dict))}
    # all_sae_activations = {key: [] for key in target_key_list}
    
    for boards, next_labels, elos_self, elos_oppo, legal_moves, side_info, wdl, board_fen in dataloader:
        logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)
        activations = getattr(_thread_local, 'residual_streams', {})
        sae_activations = apply_sae_to_activations(sae, activations, target_key_list)
        # for key in target_key_list:
        #     if key in sae_activations:
        #         all_sae_activations[key].append(sae_activations[key])
        
        logits_value_list[i].append(logits_value.clone().detach())
        logits_maia_legal = logits_maia * legal_moves
        preds = logits_maia_legal.argmax(dim=-1)
        break
    
    data_to_save = {
        'board_fen': board_fen,
        'next_labels': next_labels,
        'logits_maia': logits_maia.clone().detach(),
        'logits_value': logits_value.clone().detach(),
        'preds': preds.clone().detach(),
        'elos_self': elos_self,
        'elos_oppo': elos_oppo,
        'all_sae_activations': sae_activations,
    }
    
    with open(f'maia2-sae/activations/maia2_activations_for_sae_{args.sae_dim}_{sae_lr}_{sae_site}.pickle', 'wb') as f:
        pickle.dump(data_to_save, f)

if __name__ == "__main__":
    main()