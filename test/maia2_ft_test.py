from maia2 import model, dataset, inference, utils, main
import chess
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import collections
import random
import pdb
import matplotlib.pyplot as plt
import numpy as np

class BlunderedTransitionalDataset(Dataset):
    def __init__(self, csv_path, all_moves_dict, cfg, test_elo=None):
        df = pd.read_csv(csv_path)
        elo_dict = utils.create_elo_dict()
        
        expanded_data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing positions"):
            trans_point = row['transition_point']
            
            if test_elo is not None and trans_point <= test_elo:
                continue
                
            board = chess.Board(row['fen'])
            is_black_to_move = not board.turn

            if is_black_to_move:
                board = board.mirror()
                mirrored_fen = board.fen()
                correct_move = utils.mirror_move(row['correct_move'])
                
            if test_elo is not None:
                expanded_data.append((
                    mirrored_fen if is_black_to_move else row['fen'],
                    correct_move if is_black_to_move else row['correct_move'],
                    test_elo,
                    test_elo,
                    True
                ))
            else:
                # For training/validation, keep all ELO levels below transition point
                for elo_level in range(len(elo_dict) - 1):
                    expanded_data.append((
                        mirrored_fen if is_black_to_move else row['fen'],
                        correct_move if is_black_to_move else row['correct_move'],
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
    return (correct / total).item(), predictions == labels

def evaluate_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    with open('configs/finetune_config.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    cfg = Config(cfg_dict)
    
    # Initialize model and dictionaries
    all_moves = utils.get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = utils.create_elo_dict()
    
    # Load model
    model = main.MAIA2Model(len(all_moves), elo_dict, cfg)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    
    # Load finetuned weights
    checkpoint_path = '../maia2-ft/model/finetune_0.0001_8192_1e-05/epoch_1.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    results = {}
    elo_dict = utils.create_elo_dict()
    
    for test_elo in range(len(elo_dict) - 1):
        print(f"\nTesting ELO level {test_elo}")
        
        # Load test dataset specifically for this ELO level
        test_dataset = BlunderedTransitionalDataset(
            '../dataset/blundered-transitional-dataset/test_moves.csv',
            all_moves_dict,
            cfg,
            test_elo=test_elo
        )
        
        if len(test_dataset) == 0:
            print(f"No test cases for ELO level {test_elo}")
            continue
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                boards, labels, elos_self, elos_oppo, legal_moves, side_info = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                
                logits_maia, _, _ = model(boards, elos_self, elos_oppo)
                acc, correct_mask = calculate_accuracy(logits_maia, labels, legal_moves)
                
                correct += correct_mask.sum().item()
                total += len(labels)
        
        acc = correct / total if total > 0 else 0
        results[test_elo] = (correct, total, acc)
    
    print("\nAccuracy by ELO level:")
    print("-" * 40)
    total_correct = 0
    total_samples = 0
    
    for elo in sorted(results.keys()):
        correct, total, acc = results[elo]
        print(f"ELO level {elo}: {acc:.4f} ({correct}/{total})")
        total_correct += correct
        total_samples += total
    
    print("-" * 40)
    if total_samples > 0:
        print(f"Overall accuracy: {total_correct/total_samples:.4f} ({total_correct}/{total_samples})")

    return results

def plot_accuracy_comparison(results):
    elo_levels = list(range(10))
    finetuned_acc = [results[elo][2] for elo in elo_levels]
    baseline_acc = [0.0] * 10  # Original model accuracy (always wrong by definition)

    plt.figure(figsize=(12, 8))
    
    # Plot lines
    plt.plot(elo_levels, baseline_acc, 'go-', label='Original', linewidth=2, markersize=8)
    plt.plot(elo_levels, finetuned_acc, 'r^-', label='Fine-tuned', linewidth=2, markersize=8)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Maia 2 Strength Level', fontsize=16)
    plt.ylabel('Best Move Accuracy', fontsize=16)
    plt.title('Accuracy Comparison across Maia 2 Strength Levels', fontsize=18, pad=15)
    plt.legend(fontsize=14)

    plt.ylim(-0.05, 1.0)
    plt.xlim(-0.5, 9.5)
    plt.xticks(elo_levels)
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('../figs/maia2_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    results = evaluate_model()
    plot_accuracy_comparison(results)