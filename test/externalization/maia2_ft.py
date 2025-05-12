from maia2 import model, dataset, inference, utils, main
import chess
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import random
import pdb
import pickle
from functools import partial
from maia2.main import MAIA2Model

class ThreatAwareDataset(Dataset):
    def __init__(self, csv_path, all_moves_dict, cfg, is_test=False):
        self.cfg = cfg
        self.all_moves_dict = all_moves_dict
        self.elo_dict = utils.create_elo_dict()
        
        cache_path = csv_path.replace('.csv', '_defensive_threat_filtered.pkl')
        # if is_test:
        #     cache_path = csv_path.replace('.csv', '_defensive_threat_filtered_exclude_tp10.pkl')
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded cached filtered data from {cache_path}")
            return

        df = pd.read_csv(csv_path)
        expanded_data = []
        
        self.threat_functions = []
        for square in range(64):
            self.threat_functions.append(partial(is_square_under_defensive_threat, square_index=square))

        print("Filtering positions with defensive threats...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing positions"):
                
            board = chess.Board(row['fen'])
            
            if not board.turn:
                board.apply_mirror()
                correct_move = utils.mirror_move(row['correct_move'])
            else:
                correct_move = row['correct_move']

            has_threat = any(
                def_func(board.fen())
                for def_func in self.threat_functions
            )
            
            if has_threat:
                for elo_level in range(len(self.elo_dict)):
                    expanded_data.append((
                        board.fen(),
                        correct_move,
                        elo_level,
                        elo_level,
                        board.turn,
                        row['transition_point'] if 'transition_point' in row and not pd.isna(row['transition_point']) else -1
                    ))

        with open(cache_path, 'wb') as f:
            pickle.dump(expanded_data, f)
        print(f"Saved filtered data to {cache_path}")
        
        self.data = expanded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move, elo_self, elo_oppo, white_active, transition_point = self.data[idx]
        board = chess.Board(fen)
        board_input = utils.board_to_tensor(board)
        move_input = self.all_moves_dict[move]
        
        legal_moves, side_info = utils.get_side_info(board, move, self.all_moves_dict)
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info, transition_point

def is_square_under_offensive_threat(fen: str, square_index: int) -> bool:
    board = chess.Board(fen)
    piece = board.piece_at(square_index)
    if piece is None or piece.color != chess.BLACK:
        return False
    return len(board.attackers(chess.WHITE, square_index)) > len(board.attackers(chess.BLACK, square_index))

def is_square_under_defensive_threat(fen: str, square_index: int) -> bool:
    board = chess.Board(fen)
    piece = board.piece_at(square_index)
    if piece is None or piece.color != chess.WHITE:
        return False
    return len(board.attackers(chess.BLACK, square_index)) > len(board.attackers(chess.WHITE, square_index))

def calculate_accuracy(logits, labels, legal_moves):
    logits = logits * legal_moves
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float().sum()
    total = labels.size(0)
    return (correct / total).item(), predictions

def calculate_baseline_accuracy(transition_points, num_elo_levels):
    accuracy_per_elo = {i: 0 for i in range(num_elo_levels)}
    counts_per_elo = {i: 0 for i in range(num_elo_levels)}
    
    for tp in transition_points:
        if tp == -1:
            continue
        for elo in range(num_elo_levels):
            counts_per_elo[elo] += 1
            if tp == 0:  
                accuracy_per_elo[elo] += 1
            elif tp == 10:  
                pass  
            else:  
                if elo >= tp:
                    accuracy_per_elo[elo] += 1
    
    for elo in range(num_elo_levels):
        if counts_per_elo[elo] > 0:
            accuracy_per_elo[elo] = accuracy_per_elo[elo] / counts_per_elo[elo]
        else:
            accuracy_per_elo[elo] = 0
    
    return accuracy_per_elo

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

def run_ft():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open('finetune_config.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    cfg = Config(cfg_dict)
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    
    utils.seed_everything(cfg.seed)
    save_root = f'ft_model/defensive_threats_finetune_{cfg.lr}_{cfg.batch_size}_{cfg.wd}/'
    os.makedirs(save_root, exist_ok=True)

    all_moves = utils.get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = utils.create_elo_dict()
    move_dict = {v: k for k, v in all_moves_dict.items()}

    trained_model_path = "../weights.v2.pt"
    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    ckpt = torch.load(trained_model_path, map_location=device)
    
    if all(k.startswith('module.') for k in ckpt['model_state_dict'].keys()):
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()})
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        
    model.eval().to(device)
    print("Freezing ELO embedding layer...")
    for param in model.elo_embedding.parameters():
        param.requires_grad = False

    train_dataset = ThreatAwareDataset(
        '../blundered-transitional-dataset/train_moves.csv',
        all_moves_dict,
        cfg
    )

    val_dataset = ThreatAwareDataset(
        '../blundered-transitional-dataset/val_moves.csv',
        all_moves_dict,
        cfg
    )

    test_dataset = ThreatAwareDataset(
        '../blundered-transitional-dataset/test_moves.csv',
        all_moves_dict,
        cfg,
        is_test=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(elo_dict),
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False
    )

    criterion_maia = nn.CrossEntropyLoss()
    criterion_side_info = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    N_params = utils.count_parameters(model)
    print(f'Trainable Parameters: {N_params}', flush=True)
    best_val_loss = float('inf')

    print("Computing baseline performance...")
    transition_points = [tp for _, _, _, _, _, tp in test_dataset.data][::len(elo_dict)]
    transition_points = [tp for tp in transition_points if tp != -1]
    baseline_accuracy = calculate_baseline_accuracy(transition_points, len(elo_dict))
    
    print("Baseline accuracy per ELO level:")
    for elo, acc in baseline_accuracy.items():
        print(f"ELO level {elo}: {acc:.4f}")
    
    if transition_points:
        avg_transition_point = sum(tp for tp in transition_points if tp not in [-1, 0, 10]) / sum(1 for tp in transition_points if tp not in [-1, 0, 10]) if any(tp not in [-1, 0, 10] for tp in transition_points) else 0
        print(f"Average transition point (baseline): {avg_transition_point:.2f}")

    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}', flush=True)
        start_time = time.time()
        
        model.train()
        window_loss = 0
        window_acc = 0
        window_steps = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            boards = batch[0].to(device)
            labels = batch[1].to(device)
            elos_self = batch[2].to(device)
            elos_oppo = batch[3].to(device)
            legal_moves = batch[4].to(device)
            side_info = batch[5].to(device)
            
            logits_maia, logits_side_info, _ = model(boards, elos_self, elos_oppo)
            
            loss = criterion_maia(logits_maia, labels)
            if cfg.side_info:
                loss_side_info = criterion_side_info(logits_side_info, side_info) * cfg.side_info_coefficient
                loss += loss_side_info
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc, _ = calculate_accuracy(logits_maia, labels, legal_moves)
            window_loss += loss.item()
            window_acc += acc
            window_steps += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_window_loss = window_loss / window_steps
                avg_window_acc = window_acc / window_steps
                print(f'Batch {batch_idx + 1}/{len(train_loader)}, '
                      f'Loss: {avg_window_loss:.4f}, Accuracy: {avg_window_acc:.4f}', flush=True)
                window_loss = 0
                window_acc = 0
                window_steps = 0
        
        model.eval()
        val_loss = 0
        val_acc = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                boards, labels, elos_self, elos_oppo, legal_moves, side_info, _ = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                
                logits_maia, logits_side_info, _ = model(boards, elos_self, elos_oppo)
                
                loss = criterion_maia(logits_maia, labels)
                if cfg.side_info:
                    loss_side_info = criterion_side_info(logits_side_info, side_info) * cfg.side_info_coefficient
                    loss += loss_side_info
                
                acc, _ = calculate_accuracy(logits_maia, labels, legal_moves)
                
                val_loss += loss.item()
                val_acc += acc
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        avg_val_acc = val_acc / val_steps
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
            }, f'{save_root}best_model.pt')
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc,
        }, f'{save_root}epoch_{epoch + 1}.pt')
        
        print(f'Epoch {epoch + 1} completed in {utils.readable_time(time.time() - start_time)}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}\n')

    print("Testing best model...")
    best_model = MAIA2Model(len(all_moves), elo_dict, cfg)
    best_ckpt = torch.load(f'{save_root}best_model.pt', map_location=device)
    best_model.load_state_dict(best_ckpt['model_state_dict'])
    best_model.eval().to(device)

    elo_accuracies = {i: {'correct': 0, 'total': 0} for i in range(len(elo_dict))}
    new_transition_points = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            boards, labels, elos_self, elos_oppo, legal_moves, side_info, tps = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            
            logits_maia, _, _ = best_model(boards, elos_self, elos_oppo)
            _, predictions = calculate_accuracy(logits_maia, labels, legal_moves)
            
            for i in range(0, len(predictions), len(elo_dict)):
                if i + len(elo_dict) <= len(predictions):
                    batch_predictions = predictions[i:i+len(elo_dict)]
                    batch_labels = labels[i:i+len(elo_dict)]
                    batch_elos = elos_self[i:i+len(elo_dict)]
                    
                    tp = find_transition_point(batch_predictions, batch_labels, batch_elos.tolist())
                    if tp != -1:
                        new_transition_points.append(tp)
                    
                    for j in range(len(elo_dict)):
                        elo_level = j
                        is_correct = (batch_predictions[j] == batch_labels[j]).item()
                        elo_accuracies[elo_level]['total'] += 1
                        if is_correct:
                            elo_accuracies[elo_level]['correct'] += 1
    
    print("\nModel accuracy per ELO level:")
    for elo_level, stats in elo_accuracies.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"ELO level {elo_level}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")
    
    print("\nTransition point distribution for best model:")
    tp_counts = {}
    for tp in new_transition_points:
        if tp not in tp_counts:
            tp_counts[tp] = 0
        tp_counts[tp] += 1
    
    for tp, count in sorted(tp_counts.items()):
        percentage = count / len(new_transition_points) * 100
        print(f"Transition point {tp}: {count} positions ({percentage:.2f}%)")
    
    avg_transition_point = sum(tp for tp in new_transition_points if tp not in [0, 10]) / sum(1 for tp in new_transition_points if tp not in [0, 10]) if any(tp not in [0, 10] for tp in new_transition_points) else 0
    print(f"Average transition point (model): {avg_transition_point:.2f}")
    
    print("\nComparison of baseline vs model:")
    for elo_level in range(len(elo_dict)):
        baseline_acc = baseline_accuracy[elo_level]
        model_acc = elo_accuracies[elo_level]['correct'] / elo_accuracies[elo_level]['total'] if elo_accuracies[elo_level]['total'] > 0 else 0
        diff = model_acc - baseline_acc
        print(f"ELO level {elo_level}: Baseline {baseline_acc:.4f}, Model {model_acc:.4f}, Diff {diff:.4f}")

if __name__=="__main__":
    run_ft()