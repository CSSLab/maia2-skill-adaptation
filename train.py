import os
import maia_lib
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv

import chess.pgn
import pickle
import chess
import numpy as np
import pdb
from multiprocessing import Pool, cpu_count
import torch
import torch.nn as nn
import tqdm
import argparse
from utils import *
from main import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from tqdm.contrib.concurrent import process_map
import math
import os
import argparse
from functools import partial
from collections import defaultdict
import concurrent.futures
import einops
import time
import threading

_thread_local = threading.local()

class ActivationBuffer:
    def __init__(self, buffer_size, activation_dim):
        self.buffer_size = buffer_size
        self.activation_dim = activation_dim
        self.buffer = torch.zeros((buffer_size, activation_dim))
        self.current_index = 0

    def add(self, activations):
        batch_size = activations.size(0)
        if self.current_index + batch_size > self.buffer_size:
            return True
        self.buffer[self.current_index:self.current_index+batch_size] = activations
        self.current_index += batch_size
        return False

    def get_data(self):
        return self.buffer[:self.current_index]

    def clear(self):
        self.current_index = 0

# D = d_model, F = dictionary_size
# e.g. if d_model = 12288 and dictionary_size = 49152
# then model_activations_D.shape = (12288,) and encoder_DF.weight.shape = (12288, 49152)

class SparseAutoEncoder(nn.Module):
    """
    A one-layer autoencoder.
    """
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        self.encoder_DF = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder_FD = nn.Linear(dict_size, activation_dim, bias=True)

    def encode(self, model_activations_D: torch.Tensor) -> torch.Tensor:
        return nn.ReLU()(self.encoder_DF(model_activations_D))
    
    def decode(self, encoded_representation_F: torch.Tensor) -> torch.Tensor:
        return self.decoder_FD(encoded_representation_F)
    
    def forward_pass(self, model_activations_D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_representation_F = self.encode(model_activations_D)
        reconstructed_model_activations_D = self.decode(encoded_representation_F)
        return reconstructed_model_activations_D, encoded_representation_F

def train_sae(sae, optimizer, activations, l1_coefficient):
    optimizer.zero_grad()
    total_loss, l2_loss, l1_loss = calculate_loss(sae, activations, l1_coefficient)
    total_loss.backward()
    optimizer.step()
    return total_loss, l2_loss, l1_loss

# B = batch size, D = d_model, F = dictionary_size

def calculate_loss(autoencoder: SparseAutoEncoder, model_activations_BD: torch.Tensor, l1_coefficient: float):
    reconstructed_model_activations_BD, encoded_representation_BF = autoencoder.forward_pass(model_activations_BD)
    reconstruction_error_BD = (reconstructed_model_activations_BD - model_activations_BD).pow(2)
    reconstruction_error_B = einops.reduce(reconstruction_error_BD, 'B D -> B', 'sum')
    l2_loss = reconstruction_error_B.mean()

    l1_loss = l1_coefficient * encoded_representation_BF.abs().sum()
    total_loss = l2_loss + l1_loss
    return total_loss, l2_loss, l1_loss

# Hooks to extract model internals

def _enable_activation_hook(model, cfg):
    def get_activation(name):
        def hook(model, input, output):
            if not hasattr(_thread_local, 'activations'):
                _thread_local.activations = {}
            _thread_local.activations[name] = output.detach()
        return hook
    
    for i in range(cfg.num_blocks_vit):
        feedforward_module = model.module.transformer.elo_layers[i][1]
        feedforward_module.register_forward_hook(get_activation(f'transformer block {i} hidden states'))

def train_sae_pipeline(model, cfg, pgn_chunks, all_moves_dict, elo_dict, num_epochs, buffer_size=8192, l1_coefficient=0.00001):
    _enable_activation_hook(model, cfg)
    
    target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states']
    
    saes = {key: SparseAutoEncoder(activation_dim=cfg.dim_vit, dict_size=cfg.dim_vit * 4) for key in target_key_list}
    optimizers = {key: optim.Adam(saes[key].parameters(), lr=3e-4) for key in target_key_list}
    
    buffers = {key: ActivationBuffer(buffer_size * cfg.vit_length, cfg.dim_vit) for key in target_key_list}
    pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{cfg.test_year}-{formatted_month}.pgn"

    # Initialize monitoring variables
    start_time = time.time()
    total_sae_updates = {key: 0 for key in target_key_list}
    total_losses = {key: 0.0 for key in target_key_list}
    total_l2_losses = {key: 0.0 for key in target_key_list}
    total_l1_losses = {key: 0.0 for key in target_key_list}
    
    print(f"Starting SAE training pipeline for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_batches = 0
        
        for i in range(0, len(pgn_chunks), cfg.num_workers):
            pgn_chunks_sublist = pgn_chunks[i:i + cfg.num_workers]
            data, game_count, chunk_count = process_chunks(cfg, pgn_path, pgn_chunks_sublist, elo_dict)
            dataset = MAIA2Dataset(data, all_moves_dict, cfg)
            dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=cfg.batch_size, 
                                                shuffle=False, 
                                                drop_last=False,
                                                num_workers=cfg.num_workers)
            
            total_batches += len(dataloader)
            
            for batch_idx, (boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win, board_input) in enumerate(dataloader):
                logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)
                activations = getattr(_thread_local, 'activations', {})

                for key in target_key_list:
                    if key in activations:
                        split_activations = activations[key].view(-1, cfg.dim_vit)
                        is_buffer_full = buffers[key].add(split_activations)
                        
                        if is_buffer_full:
                            total_loss, l2_loss, l1_loss = train_sae(saes[key], optimizers[key], buffers[key].get_data(), l1_coefficient)
                            buffers[key].clear()
                            
                            total_sae_updates[key] += 1
                            total_losses[key] += total_loss.item()
                            total_l2_losses[key] += l2_loss.item()
                            total_l1_losses[key] += l1_loss.item()
                            
                            if total_sae_updates[key] % 100 == 0:
                                avg_total_loss = total_losses[key] / total_sae_updates[key]
                                avg_l2_loss = total_l2_losses[key] / total_sae_updates[key]
                                avg_l1_loss = total_l1_losses[key] / total_sae_updates[key]
                                print(f"Epoch {epoch+1}/{num_epochs}, Chunk {i//cfg.num_workers + 1}/{len(pgn_chunks)//cfg.num_workers}, "
                                    f"Batch {batch_idx+1}/{len(dataloader)}, "
                                    f"Key: {key}, Updates: {total_sae_updates[key]}, "
                                    f"Avg Total Loss: {avg_total_loss:.4f}, "
                                    f"Avg L2 Loss: {avg_l2_loss:.4f}, "
                                    f"Avg L1 Loss: {avg_l1_loss:.4f}")
                
                if hasattr(_thread_local, 'activations'):
                    _thread_local.activations.clear()
                
                # Print overall progress every 1000 batches
                if (batch_idx + 1) % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Epoch {epoch+1}/{num_epochs}, Chunk {i//cfg.num_workers + 1}/{len(pgn_chunks)//cfg.num_workers}, "
                          f"Batch {batch_idx+1}/{len(dataloader)}, Elapsed Time: {elapsed_time:.2f}s")
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Total batches processed: {total_batches}")
        for key in target_key_list:
            avg_loss = total_losses[key] / max(total_sae_updates[key], 1)
            print(f"  {key}: Total Updates: {total_sae_updates[key]}, Avg Loss: {avg_loss:.4f}")
        print()
    
    total_time = time.time() - start_time
    print(f"\nSAE training completed in {total_time:.2f}s")
    for key in target_key_list:
        avg_loss = total_losses[key] / max(total_sae_updates[key], 1)
        print(f"  {key}: Total Updates: {total_sae_updates[key]}, Final Avg Loss: {avg_loss:.4f}")
    
    return saes

def perform_intervention(model_activations_D: torch.Tensor, decoder_FD: torch.Tensor, scale: float) -> torch.Tensor:
    intervention_vector_D = decoder_FD[317, :]
    scaled_intervention_vector_D = intervention_vector_D * scale
    modified_model_activations_D = model_activations_D + scaled_intervention_vector_D
    return modified_model_activations_D

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # Supporting Arguments
    parser.add_argument('--data_root', default='pgn', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--max_ply', default=300, type=int)
    parser.add_argument('--clock_threshold', default=30, type=int)
    parser.add_argument('--chunk_size', default=20000, type=int)
    parser.add_argument('--start_year', default=2013, type=int)
    parser.add_argument('--start_month', default=1, type=int)
    parser.add_argument('--end_year', default=2019, type=int)
    parser.add_argument('--end_month', default=12, type=int)
    parser.add_argument('--from_checkpoint', default=False, type=bool)
    parser.add_argument('--checkpoint_year', default=2018, type=int)
    parser.add_argument('--checkpoint_month', default=12, type=int)
    parser.add_argument('--test_year', default=2024, type=int)
    parser.add_argument('--test_month', default=1, type=int)
    parser.add_argument('--num_cpu_left', default=4, type=int)
    parser.add_argument('--model', default='ViT', type=str)
    parser.add_argument('--max_games_per_elo_range', default=20, type=int)

    # Tunable Arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8192, type=int)
    parser.add_argument('--first_n_moves', default=10, type=int)
    parser.add_argument('--last_n_moves', default=10, type=int)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=2, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=8, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    parser.add_argument('--side_info', default=True, type=bool)
    parser.add_argument('--side_info_coefficient', default=1, type=float)
    parser.add_argument('--value', default=True, type=bool)
    parser.add_argument('--value_coefficient', default=1, type=float)
    parser.add_argument('--num_sae_epochs', default=1, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    num_processes = cpu_count() - cfg.num_cpu_left

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    move_dict = {v: k for k, v in all_moves_dict.items()}

    trained_model_path = "weights.v2.pt"
    ckpt = torch.load(trained_model_path, map_location=torch.device('cuda:0'))
    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    formatted_month = f"{cfg.test_month:02d}"
    pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{cfg.test_year}-{formatted_month}.pgn"
    pgn_chunks = get_chunks(pgn_path, cfg.chunk_size)
    print(f'Testing Mixed Elo with {len(pgn_chunks)} chunks from {pgn_path}: ', flush=True)

    trained_saes = train_sae_pipeline(model, cfg, pgn_chunks, all_moves_dict, elo_dict, num_epochs=cfg.num_sae_epochs)
    sae_state_dicts = {}
    for key, sae in trained_saes.items():
        sae_state_dicts[key] = sae.state_dict()
    save_path = f'sae/trained_saes_{cfg.test_year}-{formatted_month}-new.pt'
    torch.save(sae_state_dicts, save_path)
    print(f"Trained SAEs saved to {save_path}. Finished")
    
    