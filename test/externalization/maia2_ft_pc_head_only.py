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
import pickle
from functools import partial

class ThreatAwareDataset(Dataset):
    def __init__(self, csv_path, all_moves_dict, cfg, is_test=False):
        self.cfg = cfg
        self.all_moves_dict = all_moves_dict
        self.elo_dict = utils.create_elo_dict()
        
        cache_path = csv_path.replace('.csv', '_defensive_threat_filtered.pkl')
        
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

def is_square_under_defensive_threat(fen: str, square_index: int) -> bool:
    board = chess.Board(fen)
    piece = board.piece_at(square_index)
    if piece is None or piece.color != chess.WHITE:
        return False
    return len(board.attackers(chess.BLACK, square_index)) > len(board.attackers(chess.WHITE, square_index))

def is_square_under_offensive_threat(fen: str, square_index: int) -> bool:
    board = chess.Board(fen)
    piece = board.piece_at(square_index)
    if piece is None or piece.color != chess.BLACK:
        return False
    return len(board.attackers(chess.WHITE, square_index)) > len(board.attackers(chess.BLACK, square_index))

def calculate_accuracy(logits, labels, legal_moves):
    logits = logits * legal_moves
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float().sum()
    total = labels.size(0)
    return (correct / total).item()

class SinglePolicyHeadModel(nn.Module):
    """
    A wrapper model that keeps the base model frozen and only trains a single policy head for all skill levels
    """
    def __init__(self, base_model, fc_new):
        super(SinglePolicyHeadModel, self).__init__()
        self.base_model = base_model
        self.fc_new = fc_new
    
    def forward(self, boards, elos_self=None, elos_oppo=None):
        with torch.no_grad():
            _, _, _ = self.base_model(boards, elos_self, elos_oppo)

            # Get features from the model just before policy head
            features = self.base_model.module.last_ln(self.base_model.module.transformer(
                self.base_model.module.to_patch_embedding(
                    self.base_model.module.chess_cnn(boards).view(
                        boards.size(0), self.base_model.module.cfg.vit_length, 8 * 8
                    )
                ) + self.base_model.module.pos_embedding, 
                torch.cat((
                    self.base_model.module.elo_embedding(elos_self), 
                    self.base_model.module.elo_embedding(elos_oppo)
                ), dim=1)
            ).mean(dim=1))

        logits = self.fc_new(features)
        return logits

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

    save_root = f'ft_model/single_policy_head_finetune_{cfg.lr}_{cfg.batch_size}_{cfg.wd}/'
    os.makedirs(save_root, exist_ok=True)

    all_moves = utils.get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = utils.create_elo_dict()
    move_dict = {v: k for k, v in all_moves_dict.items()}

    trained_model_path = "../weights.v2.pt"
    ckpt = torch.load(trained_model_path, map_location=device)
    
    base_model = main.MAIA2Model(len(all_moves), elo_dict, cfg)
    base_model = torch.nn.DataParallel(base_model)
    base_model.load_state_dict(ckpt['model_state_dict'])
    base_model.to(device)
    base_model.eval()  

    num_skill_levels = len(elo_dict) - 1
    
    fc_new = nn.Linear(cfg.dim_vit, len(all_moves)).to(device)
    fc_new.weight.data = base_model.module.fc_1.weight.data.clone()
    fc_new.bias.data = base_model.module.fc_1.bias.data.clone()

    policy_model = SinglePolicyHeadModel(base_model, fc_new).to(device)
    policy_params = sum(p.numel() for p in policy_model.fc_new.parameters())
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"Number of trainable parameters in policy head: {policy_params:,}")
    print(f"Number of parameters in base model (frozen): {base_params:,}")

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
        cfg
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
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False
    )

    optimizer = torch.optim.AdamW(policy_model.fc_new.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}', flush=True)
        start_time = time.time()
        
        # Training
        policy_model.fc_new.train()
        
        epoch_loss = 0
        epoch_acc = 0
        epoch_steps = 0
        
        window_loss = 0
        window_acc = 0
        window_steps = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            boards, labels, elos_self, elos_oppo, legal_moves, side_info, transition_point = [x.to(device) if torch.is_tensor(x) else x for x in batch]

            logits = policy_model(boards, elos_self, elos_oppo)

            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logits = logits * legal_moves
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).float().sum().item()
            
            # Update stats
            batch_size = len(labels)
            epoch_loss += loss.item() * batch_size
            epoch_acc += correct
            epoch_steps += batch_size
            
            window_loss += loss.item() * batch_size
            window_acc += correct
            window_steps += batch_size
            
            if (batch_idx + 1) % 10 == 0 and window_steps > 0:
                avg_window_loss = window_loss / window_steps
                avg_window_acc = window_acc / window_steps
                print(f'Batch {batch_idx + 1}/{len(train_loader)}, '
                      f'Loss: {avg_window_loss:.4f}, Accuracy: {avg_window_acc:.4f}', flush=True)
                window_loss = 0
                window_acc = 0
                window_steps = 0
        
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            avg_epoch_acc = epoch_acc / epoch_steps
            print(f'Epoch {epoch + 1} training - Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_acc:.4f}')
        
        # Validation
        val_loss = 0
        val_acc = 0
        val_steps = 0
        
        policy_model.fc_new.eval()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                boards, labels, elos_self, elos_oppo, legal_moves, side_info, transition_point = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                
                logits = policy_model(boards, elos_self, elos_oppo)
                
                loss = criterion(logits, labels)
                
                logits = logits * legal_moves
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == labels).float().sum().item()
                
                batch_size = len(labels)
                val_loss += loss.item() * batch_size
                val_acc += correct
                val_steps += batch_size
        
        if val_steps > 0:
            avg_val_loss = val_loss / val_steps
            avg_val_acc = val_acc / val_steps
            print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': policy_model.fc_new.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_acc': avg_val_acc,
                    'config': cfg_dict,
                }, f'{save_root}best_model.pt')
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': policy_model.fc_new.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss if val_steps > 0 else float('inf'),
            'val_acc': avg_val_acc if val_steps > 0 else 0,
            'config': cfg_dict,
        }, f'{save_root}epoch_{epoch + 1}.pt')
        
        print(f'Epoch {epoch + 1} completed in {utils.readable_time(time.time() - start_time)}')
    
    print("Evaluating on test set for each skill level...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False
    )
    
    policy_model.fc_new.eval()
    
    results_per_skill = {}
    
    for skill_level in range(num_skill_levels):
        results_per_skill[skill_level] = {
            'test_loss': 0,
            'test_acc': 0,
            'test_count': 0
        }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            boards, labels, elos_self, elos_oppo, legal_moves, side_info, transition_point = [x.to(device) if torch.is_tensor(x) else x for x in batch]

            logits = policy_model(boards, elos_self, elos_oppo)
            
            # Calculate loss for each skill level
            for skill_level in range(num_skill_levels):
                skill_mask = (elos_self == skill_level)
                if skill_mask.sum() == 0:
                    continue
                
                skill_indices = skill_mask.nonzero().squeeze(1)
                skill_logits = logits[skill_indices]
                skill_labels = labels[skill_indices]
                skill_legal_moves = legal_moves[skill_indices]
                
                loss = criterion(skill_logits, skill_labels)
                
                masked_logits = skill_logits * skill_legal_moves
                predictions = torch.argmax(masked_logits, dim=1)
                correct = (predictions == skill_labels).float().sum().item()
                
                results_per_skill[skill_level]['test_loss'] += loss.item() * len(skill_indices)
                results_per_skill[skill_level]['test_acc'] += correct
                results_per_skill[skill_level]['test_count'] += len(skill_indices)
    
    print("\nTest Results per Skill Level:")
    print("-" * 50)
    for skill_level in range(num_skill_levels):
        count = results_per_skill[skill_level]['test_count']
        if count > 0:
            avg_loss = results_per_skill[skill_level]['test_loss'] / count
            avg_acc = results_per_skill[skill_level]['test_acc'] / count
            print(f"Skill Level {skill_level}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
        else:
            print(f"Skill Level {skill_level}: No data")
    
    total_loss = sum(results['test_loss'] for results in results_per_skill.values())
    total_acc = sum(results['test_acc'] for results in results_per_skill.values())
    total_count = sum(results['test_count'] for results in results_per_skill.values())
    
    results_summary = {
        'per_skill_level': results_per_skill,
        'overall': {
            'test_loss': total_loss / total_count if total_count > 0 else float('inf'),
            'test_acc': total_acc / total_count if total_count > 0 else 0,
            'total_count': total_count
        }
    }
    
    with open(f'{save_root}test_results.json', 'w') as f:
        import json
        json.dump(results_summary, f, indent=4)

if __name__=="__main__":
    run_ft()