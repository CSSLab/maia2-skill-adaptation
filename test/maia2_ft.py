from maia2 import model, dataset, inference, utils, main
# from maia2.utils import board_to_tensor, get_side_info
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

class BlunderedTransitionalDataset(Dataset):
    def __init__(self, csv_path, all_moves_dict, cfg):

        df = pd.read_csv(csv_path)
        elo_dict = utils.create_elo_dict()
        
        expanded_data = []
        for _, row in df.iterrows():
            trans_point = row['transition_point']
            
            # board = chess.Board(row['fen'])
            # is_black_to_move = not board.turn

            # if is_black_to_move:
            #     board.apply_mirror()
            #     correct_move = utils.mirror_move(row['correct_move'])
            # else:
            #     correct_move = row['correct_move']

            # Add positions for all ELO levels below transition point
            # for elo_level in range(trans_point):

            for elo_level in range(len(elo_dict) - 1):
                expanded_data.append((
                    row['fen'],
                    row['correct_move'],
                    elo_level,
                    elo_level,
                    True
                ))
        
        random.seed(cfg.seed)
        random.shuffle(expanded_data)
        self.data = expanded_data
        self.all_moves_dict = all_moves_dict
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move, elo_self, elo_oppo, white_active = self.data[idx]

        board = chess.Board(fen)
        board_input = utils.board_to_tensor(board)
        move_input = self.all_moves_dict[move]
        
        legal_moves, side_info = utils.get_side_info(board, move, self.all_moves_dict)
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info

def calculate_accuracy(logits, labels, legal_moves):
    logits = logits * legal_moves
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float().sum()
    total = labels.size(0)
    return (correct / total).item()

def run_ft():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open('../maia2/finetune_config.yaml', 'r') as f:
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
    save_root = f'model/finetune_{cfg.lr}_{cfg.batch_size}_{cfg.wd}/'
    os.makedirs(save_root, exist_ok=True)

    all_moves = utils.get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = utils.create_elo_dict()
    move_dict = {v: k for k, v in all_moves_dict.items()}

    trained_model_path = "../weights.v2.pt"
    ckpt = torch.load(trained_model_path, map_location=device)
    model = main.MAIA2Model(len(all_moves), elo_dict, cfg)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # model = model.to(device)
    # model = torch.nn.DataParallel(model) 
    # model.eval()

    train_dataset = BlunderedTransitionalDataset(
        '../dataset/blundered-transitional-dataset/train_moves.csv',
        all_moves_dict,
        cfg
    )

    val_dataset = BlunderedTransitionalDataset(
        '../dataset/blundered-transitional-dataset/val_moves.csv',
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
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False
    )

    criterion_maia = nn.CrossEntropyLoss()
    criterion_side_info = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    N_params = utils.count_parameters(model)
    print(f'Trainable Parameters: {N_params}', flush=True)
    best_val_loss = float('inf')

    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}', flush=True)
        start_time = time.time()
        
        model.train()
        train_loss = 0
        train_acc = 0
        train_steps = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            boards, labels, elos_self, elos_oppo, legal_moves, side_info = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            boards = boards.to(device)
            labels = labels.to(device)
            elos_self = elos_self.to(device)
            elos_oppo = elos_oppo.to(device)
            legal_moves = legal_moves.to(device)
            side_info = side_info.to(device)
            
            logits_maia, logits_side_info, _ = model(boards, elos_self, elos_oppo)
            
            loss = criterion_maia(logits_maia, labels)
            if cfg.side_info:
                loss_side_info = criterion_side_info(logits_side_info, side_info) * cfg.side_info_coefficient
                loss += loss_side_info
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = calculate_accuracy(logits_maia, labels, legal_moves)
            train_loss += loss.item()
            train_acc += acc
            train_steps += 1
            
            # Print every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = train_loss / train_steps
                avg_acc = train_acc / train_steps
                print(f'Batch {batch_idx + 1}/{len(train_loader)}, '
                      f'Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}', flush=True)
        
        avg_train_loss = train_loss / train_steps
        avg_train_acc = train_acc / train_steps
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                boards, labels, elos_self, elos_oppo, legal_moves, side_info = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                
                logits_maia, logits_side_info, _ = model(boards, elos_self, elos_oppo)
                
                loss = criterion_maia(logits_maia, labels)
                if cfg.side_info:
                    loss_side_info = criterion_side_info(logits_side_info, side_info) * cfg.side_info_coefficient
                    loss += loss_side_info
                
                acc = calculate_accuracy(logits_maia, labels, legal_moves)
                
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
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
            }, f'{save_root}best_model.pt')
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc,
        }, f'{save_root}epoch_{epoch + 1}.pt')
        
        print(f'Epoch {epoch + 1} completed in {utils.readable_time(time.time() - start_time)}')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}\n')

if __name__=="__main__":
    run_ft()