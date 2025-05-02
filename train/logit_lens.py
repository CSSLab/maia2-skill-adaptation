import torch
import torch.nn as nn
import pandas as pd
import chess
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from maia2 import utils
from maia2.main import MAIA2Model
import pdb

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class LogitLensDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, all_moves_dict, cfg):
        df = pd.read_csv(csv_path)
        self.data = []
        
        for _, row in df.iterrows():
            fen = row['fen']
            correct_move = row['correct_move']
            
            for elo_level in range(11):
                self.data.append((
                    fen,
                    correct_move,
                    elo_level,
                    elo_level
                ))
        
        self.all_moves_dict = all_moves_dict
        self.cfg = cfg
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fen, move, elo_self, elo_oppo = self.data[idx]
        
        board = chess.Board(fen)
        board_input = utils.board_to_tensor(board)
        move_input = self.all_moves_dict[move]
        
        legal_moves, _ = utils.get_side_info(board, move, self.all_moves_dict)
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves, fen

class ForwardHook:
    def __init__(self, name):
        self.name = name
        self.activations = None
    
    def __call__(self, module, input, output):
        self.activations = output.detach()

def run_logit_lens():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    with open('config.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = Config(cfg_dict)
    cfg.batch_size = 8192
    
    save_dir = 'logit_lens_results'
    os.makedirs(save_dir, exist_ok=True)
    
    all_moves = utils.get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = utils.create_elo_dict()
    
    trained_model_path = "../weights.v2.pt"
    ckpt = torch.load(trained_model_path, map_location=device)

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    new_state_dict = {}
    for k, v in ckpt['model_state_dict'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to(device)
    
    # Create hooks for the transformer output at each layer
    transformer_hook = ForwardHook('transformer')
    original_forward = model.transformer.forward
    
    intermediate_outputs = {}
    
    def new_forward(x, elo_emb):
        for i, (attn, ff) in enumerate(model.transformer.elo_layers):
            x = attn(x, elo_emb) + x
            x = ff(x) + x
            
            # Store the intermediate output after each transformer block
            intermediate_outputs[f'block{i}'] = x.clone()
        
        return model.transformer.norm(x)
    
    # Replace the forward method
    model.transformer.forward = new_forward
    
    test_dataset = LogitLensDataset(
        '../blundered-transitional-dataset/test_moves.csv',
        all_moves_dict,
        cfg
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    skill_levels = 11
    predictions = {
        'model': [[] for _ in range(skill_levels)],
        'block0': [[] for _ in range(skill_levels)],
        'block1': [[] for _ in range(skill_levels)]
    }
    positions = [[] for _ in range(skill_levels)]
    correct_moves = [[] for _ in range(skill_levels)]
    
    print("Running logit lens analysis...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            boards, labels, elos_self, elos_oppo, legal_moves, fens = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            
            intermediate_outputs.clear()
            logits_maia, _, _ = model(boards, elos_self, elos_oppo)
            
            block0_output = intermediate_outputs['block0']
            block0_pooled = block0_output.mean(dim=1)
            block0_pooled = model.last_ln(block0_pooled)
            block0_logits = model.fc_1(block0_pooled)
            
            block1_output = intermediate_outputs['block1']
            block1_normed = model.transformer.norm(block1_output)
            block1_pooled = block1_normed.mean(dim=1)
            block1_pooled = model.last_ln(block1_pooled)
            block1_logits = model.fc_1(block1_pooled)
            
            # Mask outputs with legal moves
            masked_logits = logits_maia * legal_moves
            block0_masked = block0_logits * legal_moves 
            block1_masked = block1_logits * legal_moves
            
            # Get predictions
            model_preds = torch.argmax(masked_logits, dim=1)
            block0_preds = torch.argmax(block0_masked, dim=1)
            block1_preds = torch.argmax(block1_masked, dim=1)
            
            # Sanity check for hooks
            if batch_idx == 0:
                print("Validation: sum difference between block1_logits and logits_maia:", 
                      torch.sum(torch.abs(block1_logits - logits_maia)).item())
                print("Percentage of matching predictions:", 
                      (block1_preds == model_preds).float().mean().item() * 100, "%")
            
            for i in range(len(elos_self)):
                skill = elos_self[i].item()
                if skill < skill_levels:
                    predictions['model'][skill].append(model_preds[i].item())
                    predictions['block0'][skill].append(block0_preds[i].item())
                    predictions['block1'][skill].append(block1_preds[i].item())
                    positions[skill].append(fens[i])
                    correct_moves[skill].append(labels[i].item())
    
    # Restore original forward function
    model.transformer.forward = original_forward
    
    accuracy = {
        'model': np.zeros(skill_levels),
        'block0': np.zeros(skill_levels),
        'block1': np.zeros(skill_levels)
    }
    
    for source in ['model', 'block0', 'block1']:
        for skill in range(skill_levels):
            if len(predictions[source][skill]) == 0:
                accuracy[source][skill] = np.nan
                continue
                
            matches = sum(pred == correct for pred, correct in 
                        zip(predictions[source][skill], correct_moves[skill]))
            accuracy[source][skill] = matches / len(predictions[source][skill])
    
    block_model_agreement = {
        'block0': np.zeros(skill_levels),
        'block1': np.zeros(skill_levels)
    }
    
    for block in ['block0', 'block1']:
        for skill in range(skill_levels):
            if len(predictions[block][skill]) == 0 or len(predictions['model'][skill]) == 0:
                block_model_agreement[block][skill] = np.nan
                continue
                
            matches = sum(b_pred == m_pred for b_pred, m_pred in 
                        zip(predictions[block][skill], predictions['model'][skill]))
            block_model_agreement[block][skill] = matches / len(predictions[block][skill])
    
    skill_consistency = {
        'model': np.zeros((skill_levels, skill_levels)),
        'block0': np.zeros((skill_levels, skill_levels)),
        'block1': np.zeros((skill_levels, skill_levels))
    }
    
    for source in ['model', 'block0', 'block1']:
        for skill1 in range(skill_levels):
            for skill2 in range(skill_levels):
                source_by_pos_skill1 = {pos: pred for pos, pred in 
                                    zip(positions[skill1], predictions[source][skill1])}
                source_by_pos_skill2 = {pos: pred for pos, pred in 
                                    zip(positions[skill2], predictions[source][skill2])}
                
                common_positions = set(source_by_pos_skill1.keys()).intersection(
                                    set(source_by_pos_skill2.keys()))
                
                if len(common_positions) == 0:
                    skill_consistency[source][skill1, skill2] = np.nan
                    continue
                
                matches = sum(source_by_pos_skill1[pos] == source_by_pos_skill2[pos] 
                            for pos in common_positions)
                skill_consistency[source][skill1, skill2] = matches / len(common_positions)
    
    np.save(f"{save_dir}/accuracy.npy", accuracy)
    np.save(f"{save_dir}/block_model_agreement.npy", block_model_agreement)
    np.save(f"{save_dir}/skill_consistency.npy", skill_consistency)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(skill_levels), accuracy['model'], 'g-', label='Full Model')
    plt.plot(range(skill_levels), accuracy['block0'], 'b-', label='After Block 0')
    plt.plot(range(skill_levels), accuracy['block1'], 'r-', label='After Block 1')
    plt.xlabel('Skill Level')
    plt.ylabel('Move Prediction Accuracy')
    plt.title('Accuracy Across Transformer Blocks and Skill Levels')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/accuracy.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(skill_levels), block_model_agreement['block0'], 'b-', label='Block 0 vs Model')
    plt.plot(range(skill_levels), block_model_agreement['block1'], 'r-', label='Block 1 vs Model')
    plt.xlabel('Skill Level')
    plt.ylabel('Agreement with Model Predictions')
    plt.title('Transformer Block Agreement with Model')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/block_model_agreement.png")
    
    for source in ['model', 'block0', 'block1']:
        plt.figure(figsize=(10, 8))
        sns.heatmap(skill_consistency[source], annot=True, cmap='viridis', fmt='.2f')
        plt.xlabel('Skill Level 2')
        plt.ylabel('Skill Level 1')
        plt.title(f'Prediction Consistency: {source}')
        plt.savefig(f"{save_dir}/{source}_skill_consistency.png")
    
    relative_consistency = {
        'block0_vs_model': np.zeros((skill_levels, skill_levels)),
        'block1_vs_model': np.zeros((skill_levels, skill_levels))
    }
    
    for block in ['block0', 'block1']:
        for skill1 in range(skill_levels):
            for skill2 in range(skill_levels):
                if np.isnan(skill_consistency[block][skill1, skill2]) or np.isnan(skill_consistency['model'][skill1, skill2]):
                    relative_consistency[f'{block}_vs_model'][skill1, skill2] = np.nan
                    continue
                
                relative_consistency[f'{block}_vs_model'][skill1, skill2] = (
                    skill_consistency[block][skill1, skill2] - skill_consistency['model'][skill1, skill2]
                )
    
    for key, matrix in relative_consistency.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
        plt.xlabel('Skill Level 2')
        plt.ylabel('Skill Level 1')
        plt.title(f'Relative Consistency: {key}')
        plt.savefig(f"{save_dir}/{key}_relative_consistency.png")
    
    print("Logit lens analysis completed. Results saved to:", save_dir)
    
    return accuracy, block_model_agreement, skill_consistency

if __name__ == "__main__":
    accuracy, block_model_agreement, skill_consistency = run_logit_lens()