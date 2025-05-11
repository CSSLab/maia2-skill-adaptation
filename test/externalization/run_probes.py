from maia2 import model, dataset, inference, utils, main
import chess
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import random
import pickle
import numpy as np
from functools import partial
from maia2.main import MAIA2Model
import threading
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

_thread_local = threading.local()

def is_square_under_defensive_threat(fen: str, square_index: int) -> bool:
    board = chess.Board(fen)
    piece = board.piece_at(square_index)
    if piece is None or piece.color != chess.WHITE:
        return False
    return len(board.attackers(chess.BLACK, square_index)) > len(board.attackers(chess.WHITE, square_index))

class SquareDataset(Dataset):
    def __init__(self, data, all_moves_dict):
        self.data = data
        self.all_moves_dict = all_moves_dict
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fen, move, turn, transition_point, is_positive = self.data[idx]
        board = chess.Board(fen)
        board_input = utils.board_to_tensor(board)
        move_input = self.all_moves_dict[move]
        legal_moves, side_info = utils.get_side_info(board, move, self.all_moves_dict)
        return board_input, move_input, legal_moves, side_info, is_positive

def _enable_activation_hook(model):
    def get_activation(name):
        def hook(model, input, output):
            if not hasattr(_thread_local, 'residual_streams'):
                _thread_local.residual_streams = {}
            _thread_local.residual_streams[name] = output.detach()
        return hook
    
    for i in range(2):
        feedforward_module = model.transformer.elo_layers[i][1]
        feedforward_module.register_forward_hook(get_activation(f'transformer block {i} hidden states'))

def extract_activations(model, dataloader, device, elo_level):
    model.eval()
    all_activations = {}
    all_labels = []
    
    _enable_activation_hook(model)
    
    with torch.no_grad():
        for batch in dataloader:
            boards, moves, legal_moves, side_info, labels = batch
            boards = boards.to(device)
            
            elos_self = torch.full((len(boards),), elo_level, dtype=torch.long).to(device)
            elos_oppo = torch.full((len(boards),), elo_level, dtype=torch.long).to(device)
            
            _, _, _ = model(boards, elos_self, elos_oppo)
            
            activations = getattr(_thread_local, 'residual_streams', {})
            
            for key, activation in activations.items():
                avg_activation = torch.mean(activation, dim=1)
                
                if key not in all_activations:
                    all_activations[key] = []
                all_activations[key].append(avg_activation.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())
            
            _thread_local.residual_streams = {}
    
    for key in all_activations:
        all_activations[key] = np.vstack(all_activations[key])
    
    return all_activations, np.array(all_labels)

def train_probe(X_train, y_train, X_val, y_val, X_test, y_test):
    l1_values = [1e-2, 1e-1, 1, 5, 10]
    
    best_r2 = -float('inf')
    best_model = None
    best_params = None
    
    for l1 in l1_values:
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=1/l1, max_iter=32)
        clf.fit(X_train, y_train)
        
        val_pred_proba = clf.predict_proba(X_val)[:, 1]
        val_r2 = r2_score(y_val, val_pred_proba)
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_model = clf
            best_params = {'l1': l1}
    
    test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_r2 = r2_score(y_test, test_pred_proba)
    
    return test_r2, best_params

def create_square_dataset(square_idx, split='train'):
    if split == 'train':
        csv_path = '../../blundered-transitional-dataset/train_moves.csv'
    elif split == 'val':
        csv_path = '../../blundered-transitional-dataset/val_moves.csv'
    else:
        csv_path = '../../blundered-transitional-dataset/test_moves.csv'
    
    cache_path = f'../../square_datasets/{split}/square_{square_idx}_balanced.pkl'
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            balanced_data = pickle.load(f)
        print(f"Loaded cached data for {split} square {chess.square_name(square_idx)}")
        return balanced_data
    
    df = pd.read_csv(csv_path)
    
    positive_data = []
    negative_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split} for {chess.square_name(square_idx)}"):
        board = chess.Board(row['fen'])
        
        if not board.turn:
            board.apply_mirror()
            correct_move = utils.mirror_move(row['correct_move'])
        else:
            correct_move = row['correct_move']
        
        is_threatened = is_square_under_defensive_threat(board.fen(), square_idx)
        
        move_data = {
            'fen': board.fen(),
            'correct_move': correct_move,
            'turn': board.turn,
            'transition_point': row['transition_point'] if 'transition_point' in row and not pd.isna(row['transition_point']) else -1
        }
        
        if is_threatened:
            positive_data.append(move_data)
        else:
            negative_data.append(move_data)
    
    num_positives = len(positive_data)
    if num_positives == 0:
        print(f"No positive examples for square {chess.square_name(square_idx)}, skipping...")
        return None
    
    if num_positives > len(negative_data):
        negative_samples = negative_data
    else:
        negative_samples = random.sample(negative_data, num_positives)
    
    balanced_data = []
    for data, is_positive in [(positive_data, True), (negative_samples, False)]:
        for item in data:
            balanced_data.append((
                item['fen'],
                item['correct_move'],
                item['turn'],
                item['transition_point'],
                is_positive
            ))
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(balanced_data, f)
    
    print(f"Cached data for {split} square {chess.square_name(square_idx)}")
    
    return balanced_data

def process_square(square_idx):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    all_moves = utils.get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = utils.create_elo_dict()
    
    with open('../finetune_config.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    cfg = Config(cfg_dict)
    
    print(f"\nCreating datasets for square {chess.square_name(square_idx)}...")
    
    train_data = create_square_dataset(square_idx, 'train')
    val_data = create_square_dataset(square_idx, 'val')
    test_data = create_square_dataset(square_idx, 'test')
    
    if train_data is None or val_data is None or test_data is None:
        print(f"Skipping square {chess.square_name(square_idx)} due to missing data")
        return None
    
    print(f"Created datasets: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    train_dataset = SquareDataset(train_data, all_moves_dict)
    val_dataset = SquareDataset(val_data, all_moves_dict)
    test_dataset = SquareDataset(test_data, all_moves_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8192, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8192, shuffle=False)
    
    print(f"Loading models...")
    
    pretrained_path = "../../weights.v2.pt"
    finetuned_path = "../ft_model/defensive_threats_finetune_5e-05_8192_1e-05/best_model.pt"
    
    pretrained_model = MAIA2Model(len(all_moves), elo_dict, cfg)
    finetuned_model = MAIA2Model(len(all_moves), elo_dict, cfg)
    
    pretrained_ckpt = torch.load(pretrained_path, map_location=device)
    finetuned_ckpt = torch.load(finetuned_path, map_location=device)
    
    if all(k.startswith('module.') for k in pretrained_ckpt['model_state_dict'].keys()):
        pretrained_model.load_state_dict({k.replace('module.', ''): v for k, v in pretrained_ckpt['model_state_dict'].items()})
    else:
        pretrained_model.load_state_dict(pretrained_ckpt['model_state_dict'])
    
    if all(k.startswith('module.') for k in finetuned_ckpt['model_state_dict'].keys()):
        finetuned_model.load_state_dict({k.replace('module.', ''): v for k, v in finetuned_ckpt['model_state_dict'].items()})
    else:
        finetuned_model.load_state_dict(finetuned_ckpt['model_state_dict'])
    
    pretrained_model.eval().to(device)
    finetuned_model.eval().to(device)
    
    results = {
        'pretrained': {},
        'finetuned': {}
    }
    
    for elo_level in range(11):
        print(f"Processing ELO level {elo_level} for square {chess.square_name(square_idx)}")
        
        pretrained_activations_train, train_labels = extract_activations(pretrained_model, train_loader, device, elo_level)
        pretrained_activations_val, val_labels = extract_activations(pretrained_model, val_loader, device, elo_level)
        pretrained_activations_test, test_labels = extract_activations(pretrained_model, test_loader, device, elo_level)
        
        finetuned_activations_train, _ = extract_activations(finetuned_model, train_loader, device, elo_level)
        finetuned_activations_val, _ = extract_activations(finetuned_model, val_loader, device, elo_level)
        finetuned_activations_test, _ = extract_activations(finetuned_model, test_loader, device, elo_level)
        
        results['pretrained'][elo_level] = {}
        results['finetuned'][elo_level] = {}
        
        for layer_name in ['transformer block 0 hidden states', 'transformer block 1 hidden states']:
            pretrained_r2, pretrained_params = train_probe(
                pretrained_activations_train[layer_name], train_labels,
                pretrained_activations_val[layer_name], val_labels,
                pretrained_activations_test[layer_name], test_labels
            )
            
            finetuned_r2, finetuned_params = train_probe(
                finetuned_activations_train[layer_name], train_labels,
                finetuned_activations_val[layer_name], val_labels,
                finetuned_activations_test[layer_name], test_labels
            )
            
            results['pretrained'][elo_level][layer_name] = pretrained_r2
            results['finetuned'][elo_level][layer_name] = finetuned_r2
            
            print(f"  Pretrained {layer_name}: {pretrained_r2:.4f}")
            print(f"  Finetuned {layer_name}: {finetuned_r2:.4f}")
    
    averaged_results = {
        'pretrained': {},
        'finetuned': {}
    }
    
    for elo_level in range(11):
        averaged_results['pretrained'][elo_level] = np.mean([
            results['pretrained'][elo_level]['transformer block 0 hidden states'],
            results['pretrained'][elo_level]['transformer block 1 hidden states']
        ])
        averaged_results['finetuned'][elo_level] = np.mean([
            results['finetuned'][elo_level]['transformer block 0 hidden states'],
            results['finetuned'][elo_level]['transformer block 1 hidden states']
        ])
    
    print(f"\nAverage results for square {chess.square_name(square_idx)}:")
    for elo_level in range(11):
        print(f"  ELO {elo_level}: Pretrained {averaged_results['pretrained'][elo_level]:.4f}, Finetuned {averaged_results['finetuned'][elo_level]:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    elo_levels = list(range(11))
    pretrained_values = [averaged_results['pretrained'][elo] for elo in elo_levels]
    finetuned_values = [averaged_results['finetuned'][elo] for elo in elo_levels]
    
    plt.plot(elo_levels, pretrained_values, marker='o', linewidth=2, markersize=8, label='Original')
    plt.plot(elo_levels, finetuned_values, marker='s', linewidth=2, markersize=8, label='Fine-tuned')
    
    plt.xlabel('Skill Level', fontsize=20)
    plt.ylabel('Understanding Level', fontsize=20)
    plt.ylim(0.5, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    plt.title(f'{chess.square_name(square_idx)} under attack', fontsize=28)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/square_{square_idx}_understanding.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(f'results/square_{square_idx}_results.pkl', 'wb') as f:
        pickle.dump(averaged_results, f)
    
    return averaged_results

def main():
    all_results = {}
    
    for square_idx in range(64):
        print(f"\n{'='*50}")
        print(f"Processing square {chess.square_name(square_idx)} (index {square_idx})")
        print(f"{'='*50}")
        
        results = process_square(square_idx)
        if results is not None:
            all_results[square_idx] = results
            print(f"Completed square {chess.square_name(square_idx)}")
    
    print(f"\nCompleted processing all squares. Processed {len(all_results)}/64 squares.")
    
    with open('results/all_squares_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\nGenerating summary visualization...")
    
    plt.figure(figsize=(12, 8))
    
    pretrained_all = {elo: [] for elo in range(11)}
    finetuned_all = {elo: [] for elo in range(11)}
    
    for square_idx, results in all_results.items():
        for elo in range(11):
            pretrained_all[elo].append(results['pretrained'][elo])
            finetuned_all[elo].append(results['finetuned'][elo])
    
    elo_levels = list(range(11))
    pretrained_means = [np.mean(pretrained_all[elo]) for elo in elo_levels]
    finetuned_means = [np.mean(finetuned_all[elo]) for elo in elo_levels]
    
    pretrained_stds = [np.std(pretrained_all[elo]) for elo in elo_levels]
    finetuned_stds = [np.std(finetuned_all[elo]) for elo in elo_levels]
    
    plt.plot(elo_levels, pretrained_means, marker='o', linewidth=3, markersize=10, label='Original')
    plt.plot(elo_levels, finetuned_means, marker='s', linewidth=3, markersize=10, label='Fine-tuned')
    
    plt.fill_between(elo_levels, 
                    np.array(pretrained_means) - np.array(pretrained_stds),
                    np.array(pretrained_means) + np.array(pretrained_stds),
                    alpha=0.2)
    plt.fill_between(elo_levels,
                    np.array(finetuned_means) - np.array(finetuned_stds),
                    np.array(finetuned_means) + np.array(finetuned_stds),
                    alpha=0.2)
    
    plt.xlabel('Skill Level', fontsize=20)
    plt.ylabel('Understanding Level', fontsize=20)
    plt.ylim(0.5, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    
    plt.tight_layout()
    plt.savefig('results/average_understanding.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Process completed! Results and visualizations saved in the 'results' directory.")

if __name__ == "__main__":
    main()