import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
import torch
import torch.nn as nn
import tqdm
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.contrib.concurrent import process_map
import argparse
from functools import partial
from collections import defaultdict
import einops
import time
from itertools import product

from ..maia2.main import *
from ..maia2.utils import *
from ..test.evaluate_sae_features_on_strategic import *
from ..test.get_self_implemented_concepts import *
from ..test.evaluate_sae_features_on_board_reconstruction import cache_examples_and_activations_board_states
import threading
# import pdb

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
# JumpRelu SAE

class JumpReLU(torch.autograd.Function):
    """
    JumpReLU activation with correct gradient estimation as per DM paper:
    - Forward: Hard threshold at theta
    - Backward: Uses kernel density estimation for gradients
    """
    @staticmethod
    def forward(ctx, input, threshold, epsilon=0.001):
        ctx.save_for_backward(input, threshold)
        ctx.epsilon = epsilon
        return input * (input >= threshold)  # z * H(z - theta)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        epsilon = ctx.epsilon
        grad_input = grad_output.clone()
        
        # Standard gradient for input when above threshold
        grad_input = torch.where(input >= threshold, grad_output, torch.zeros_like(grad_input))
        
        # Gradient for threshold from reconstruction loss (JumpReLU part)
        # ∂JumpReLU(z)/∂θ := -(θ/ε)K((z-θ)/ε)
        z_centered = (input - threshold) / epsilon
        kernel_vals = (torch.abs(z_centered) <= 0.5).float()  # rectangle kernel
        grad_threshold_recon = -(threshold / epsilon) * kernel_vals * grad_output
        
        # final gradient for threshold parameter
        grad_threshold = grad_threshold_recon.sum(dim=0)
        
        return grad_input, grad_threshold, None
    
class SparseAutoEncoder(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        # Initialize encoder and decoder
        self.encoder_DF = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder_FD = nn.Linear(dict_size, activation_dim, bias=True)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.encoder_DF.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.decoder_FD.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder_DF.bias)
        nn.init.zeros_(self.decoder_FD.bias)

        self.threshold = nn.Parameter(torch.ones(dict_size) * 0.5)
        self.epsilon = 0.001  # KDE bandwidth parameter
        
    def encode(self, model_activations: torch.Tensor) -> torch.Tensor:
        return JumpReLU.apply(self.encoder_DF(model_activations), self.threshold, self.epsilon)
    
    def decode(self, encoded_activations: torch.Tensor) -> torch.Tensor:
        return self.decoder_FD(encoded_activations)
    
    def forward_pass(self, model_activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(model_activations)
        reconstructed = self.decode(encoded)
        return reconstructed, encoded

def calculate_loss(sae: nn.Module, model_activations: torch.Tensor, l0_coefficient: float, epsilon=0.001):
    """
    Compute total loss including:
    1. Reconstruction loss using JumpReLU activation
    2. L0 sparsity loss using Heaviside with gradient estimator illustrated in the papere
    """

    reconstructed_activations, encoded = sae.forward_pass(model_activations)
    reconstruction_loss = (reconstructed_activations - model_activations).pow(2).mean()
    
    # L0 sparsity loss with gradient estimator for threshold
    pre_activations = sae.encoder_DF(model_activations)
    threshold = sae.threshold

    z_centered = (pre_activations - threshold.unsqueeze(0)) / epsilon
    kernel_vals = (torch.abs(z_centered) <= 0.5).float() 

    def backward_hook(grad):
        # ∂H(z-θ)/∂θ := -(1/ε)K((z-θ)/ε)
        return -(1.0 / epsilon) * kernel_vals * grad
    
    heaviside = (pre_activations >= threshold.unsqueeze(0)).float()
    heaviside = heaviside.detach().requires_grad_()
    heaviside.register_hook(backward_hook)
    l0_loss = l0_coefficient * heaviside.mean()
    
    total_loss = reconstruction_loss + l0_loss
    return total_loss, reconstruction_loss, l0_loss

def train_sae(sae, optimizer, activations, l0_coefficient):
    optimizer.zero_grad()
    total_loss, l2_loss, l0_loss = calculate_loss(sae, activations, l0_coefficient)
    total_loss.backward()

    # Ensure threshold stays positive
    with torch.no_grad():
        sae.threshold.clamp_(min=1e-6)

    optimizer.step()
    return total_loss, l2_loss, l0_loss

# Hooks to extract model internals

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


# def _enable_activation_hook(model, cfg):
#     def get_activation(name):
#         def hook(model, input, output):
#             if not hasattr(_thread_local, 'activations'):
#                 _thread_local.activations = {}
#             _thread_local.activations[name] = output.detach()
#         return hook
    
#     for i in range(cfg.num_blocks_vit):
#         feedforward_module = model.module.transformer.elo_layers[i][1]
#         feedforward_module.register_forward_hook(get_activation(f'transformer block {i} hidden states'))

def train_sae_pipeline(model, saes, cfg, pgn_chunks, all_moves_dict, elo_dict, num_epochs, buffer_size=8192, l0_coefficient=0.00001):
    
    concept_functions = {
        "in_check": in_check,
        "has_mate_threat": has_mate_threat,
        "has_connected_rooks_mine": has_connected_rooks_mine,
        "has_connected_rooks_opponent": has_connected_rooks_opponent,
        "has_bishop_pair_mine": has_bishop_pair_mine,
        "has_bishop_pair_opponent": has_bishop_pair_opponent,
        "has_control_of_open_file_mine": has_control_of_open_file_mine,
        "has_control_of_open_file_opponent": has_control_of_open_file_opponent,
        "can_capture_queen_mine": can_capture_queen_mine,
        "can_capture_queen_opponent": can_capture_queen_opponent,
        "has_contested_open_file": has_contested_open_file,
        "has_right_bc_ha_promotion_mine": has_right_bc_ha_promotion_mine,
        "has_right_bc_ha_promotion_opponent": has_right_bc_ha_promotion_opponent,
        "capture_possible_on_d1_mine": capture_possible_on_d1_mine,
        "capture_possible_on_d2_mine": capture_possible_on_d2_mine,
        "capture_possible_on_d3_mine": capture_possible_on_d3_mine,
        "capture_possible_on_e1_mine": capture_possible_on_e1_mine,
        "capture_possible_on_e2_mine": capture_possible_on_e2_mine,
        "capture_possible_on_e3_mine": capture_possible_on_e3_mine,
        "capture_possible_on_g5_mine": capture_possible_on_g5_mine,
        "capture_possible_on_b5_mine": capture_possible_on_b5_mine,
        "capture_possible_on_d1_opponent": capture_possible_on_d1_opponent,
        "capture_possible_on_d2_opponent": capture_possible_on_d2_opponent,
        "capture_possible_on_d3_opponent": capture_possible_on_d3_opponent,
        "capture_possible_on_e1_opponent": capture_possible_on_e1_opponent,
        "capture_possible_on_e2_opponent": capture_possible_on_e2_opponent,
        "capture_possible_on_e3_opponent": capture_possible_on_e3_opponent,
        "capture_possible_on_g5_opponent": capture_possible_on_g5_opponent,
        "capture_possible_on_b5_opponent": capture_possible_on_b5_opponent,
    }
    
    _enable_activation_hook(model, cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    assert cfg.sae_attention_heads + cfg.sae_mlp_outputs + cfg.sae_residual_streams == 1
    if cfg.sae_attention_heads:
        target_key_list = ['transformer block 0 attention heads', 'transformer block 1 attention heads']
    if cfg.sae_mlp_outputs:
        target_key_list = ['transformer block 0 MLP outputs', 'transformer block 1 MLP outputs']
    if cfg.sae_residual_streams:
        target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states']

    optimizers = {key: optim.Adam(saes[key].parameters(), lr=3e-4) for key in target_key_list}
    
    buffers = {key: ActivationBuffer(buffer_size, cfg.dim_vit) for key in target_key_list}
    pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{cfg.test_year}-{formatted_month}.pgn"

    # Initialize monitoring variables
    start_time = time.time()
    total_sae_updates = {key: 0 for key in target_key_list}
    total_losses = {key: 0.0 for key in target_key_list}
    total_l2_losses = {key: 0.0 for key in target_key_list}
    total_l0_losses = {key: 0.0 for key in target_key_list}

    loss_history = {key: {
        'total_loss': [],
        'l2_loss': [],
        'l0_loss': [],
        'step': []
    } for key in target_key_list}

    auc_history = {key: {
        'str_auc': [],
        'brd_auc': [],
        'step': []
    } for key in target_key_list}

    # Initialize variables for calibration and early stopping
    best_auc_sum = {key: 0 for key in target_key_list}
    best_model_state_dicts = {key: None for key in target_key_list}
    best_model_metrics = {
        key: {
            'str_auc': 0,
            'brd_auc': 0,
            'total_auc': 0,
            'step': 0,
            'epoch': 0
        } for key in target_key_list
    }
    patience = 3
    patience_counter = {key: 0 for key in target_key_list}
    early_stop = False
    
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
                # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
                moves = moves.to(device)
                boards = boards.to(device)
                elos_self = elos_self.to(device)
                elos_oppo = elos_oppo.to(device)
                legal_moves = legal_moves.to(device)
                
                logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)
                
                if cfg.sae_attention_heads:

                    # Concat all attention heads into dim_vit
                    tmp_activations = getattr(_thread_local, 'attention_heads', {})
                    activations = {}

                    for key in target_key_list:
                        all_heads = []
                        for i in range(16):
                            heads_name = key + f"_head_{i}"
                            tmp_att_head = tmp_activations[heads_name]
                            all_heads.append(tmp_att_head)

                        concatenated_att = torch.cat(all_heads, dim=-1)
                        assert concatenated_att.shape[-1] == cfg.dim_vit
                        activations[key] = concatenated_att

                if cfg.sae_mlp_outputs:
                    activations = getattr(_thread_local, 'mlp_outputs', {})
                if cfg.sae_residual_streams:
                    activations = getattr(_thread_local, 'residual_streams', {})

                # print(activations[target_key_list[0]].shape)

                for key in target_key_list:
                    if key in activations:
                        split_activations = torch.mean(activations[key], dim=1).view(-1, cfg.dim_vit)
                        is_buffer_full = buffers[key].add(split_activations)
                        
                        if is_buffer_full:
                            total_loss, l2_loss, l0_loss = train_sae(saes[key], optimizers[key], buffers[key].get_data(), l0_coefficient)
                            buffers[key].clear()
                            
                            total_sae_updates[key] += 1
                            total_losses[key] += total_loss.item()
                            total_l2_losses[key] += l2_loss.item()
                            total_l0_losses[key] += l0_loss.item()

                            loss_history[key]['total_loss'].append(total_loss.item())
                            loss_history[key]['l2_loss'].append(l2_loss.item())
                            loss_history[key]['l0_loss'].append(l0_loss.item())
                            loss_history[key]['step'].append(total_sae_updates[key])
                            
                            if total_sae_updates[key] % 100 == 0:
                                sae_state_dicts = {}
                                for temp_key, sae in saes.items():
                                    sae_state_dicts[temp_key] = sae.state_dict()
                                print(f"Feature Calibration for SAE dim = {cfg.sae_dim}, l0 coefficient = {l0_coefficient} on Layer {key}:")
                                str_res, str_auc = evaluate_sae_features_in_train_strategic(split_activations, sae_state_dicts, key, board_input, concept_functions)
                                brd_res, brd_auc = evaluate_sae_features_in_train_board_state(split_activations, sae_state_dicts, key, board_input)
                                
                                auc_history[key]['str_auc'].append(str_auc)
                                auc_history[key]['brd_auc'].append(brd_auc)
                                auc_history[key]['step'].append(total_sae_updates[key])

                                avg_total_loss = total_losses[key] / total_sae_updates[key]
                                avg_l2_loss = total_l2_losses[key] / total_sae_updates[key]
                                avg_l0_loss = total_l0_losses[key] / total_sae_updates[key]
                                print(f"Epoch {epoch+1}/{num_epochs}, Chunk {i//cfg.num_workers + 1}/{len(pgn_chunks)//cfg.num_workers}, "
                                    f"Batch {batch_idx+1}/{len(dataloader)}, "
                                    f"Key: {key}, Updates: {total_sae_updates[key]}, "
                                    f"Avg Total Loss: {avg_total_loss:.4f}, "
                                    f"Avg L2 Loss: {avg_l2_loss:.4f}, "
                                    f"Avg L0 Loss: {avg_l0_loss:.4f}")

                                checkpoint = {
                                    'sae_state_dicts': sae_state_dicts,
                                    'loss_history': {
                                        k: {
                                            'total_loss': loss_history[k]['total_loss'],
                                            'l2_loss': loss_history[k]['l2_loss'],
                                            'l0_loss': loss_history[k]['l0_loss'],
                                            'step': loss_history[k]['step']
                                        } for k in target_key_list
                                    },
                                    'auc_history': {
                                        k: {
                                            'str_auc': auc_history[k]['str_auc'],
                                            'brd_auc': auc_history[k]['brd_auc'],
                                            'step': auc_history[key]['step']
                                        } for k in target_key_list
                                    }
                                }

                                # sae_state_dicts = {}
                                # for tmp_key, sae in saes.items():
                                #     sae_state_dicts[tmp_key] = sae.state_dict()

                                if cfg.sae_attention_heads:    
                                    save_path = f'maia2-sae/sae/trained_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-att-temp.pt'
                                if cfg.sae_residual_streams:
                                    save_path = f'maia2-sae/sae/trained_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-res-temp.pt'
                                if cfg.sae_mlp_outputs:
                                    save_path = f'maia2-sae/sae/trained_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-mlp-temp.pt'
                                torch.save(checkpoint, save_path)

                                current_auc_sum = str_auc + brd_auc
                                if current_auc_sum > best_auc_sum[key]:
                                    best_auc_sum[key] = current_auc_sum
                                    best_model_state_dicts = {k: v.state_dict() for k, v in saes.items()}  # Save all SAEs state
                                    best_model_metrics[key] = {
                                        'str_auc': str_auc,
                                        'brd_auc': brd_auc,
                                        'total_auc': current_auc_sum,
                                        'step': total_sae_updates[key],
                                        'epoch': epoch
                                    }
                                    
                                    best_checkpoint = {
                                        'sae_state_dicts': best_model_state_dicts,  # Consistent with regular save format
                                        'loss_history': {
                                            k: {
                                                'total_loss': loss_history[k]['total_loss'],
                                                'l2_loss': loss_history[k]['l2_loss'],
                                                'l0_loss': loss_history[k]['l0_loss'],
                                                'step': loss_history[k]['step']
                                            } for k in target_key_list
                                        },
                                        'auc_history': {
                                            k: {
                                                'str_auc': auc_history[k]['str_auc'],
                                                'brd_auc': auc_history[k]['brd_auc'],
                                                'step': auc_history[k]['step']
                                            } for k in target_key_list
                                        },
                                        'best_metrics': best_model_metrics
                                    }
                                    
                                    if cfg.sae_attention_heads:    
                                        best_save_path = f'maia2-sae/sae/best_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-att.pt'
                                    if cfg.sae_residual_streams:
                                        best_save_path = f'maia2-sae/sae/threat_finetuned_best_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-res.pt'
                                    if cfg.sae_mlp_outputs:
                                        best_save_path = f'maia2-sae/sae/best_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-mlp.pt'
                                    
                                    torch.save(best_checkpoint, best_save_path)
                                    print(f"New best model saved with AUC sum: {current_auc_sum:.4f} (STR: {str_auc:.4f}, BRD: {brd_auc:.4f})")
                                    
                                    patience_counter[key] = 0
                                else:
                                    patience_counter[key] += 1

                                if patience_counter[key] >= patience:
                                    print(f"Early stopping condition met for layer: {key}")
                                    if all(patience_counter[k] >= patience for k in target_key_list):
                                        print("Early stopping condition met for all layers. Stopping training.")
                                        early_stop = True
                                        break
                
                for site in ['attention_heads', 'mlp_outputs', 'residual_streams']:
                    if hasattr(_thread_local, site):
                        delattr(_thread_local, site)
                
                if early_stop:
                    break

                # Print overall progress every 1000 batches
                if (batch_idx + 1) % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx + 1}, Chunk {i//cfg.num_workers + 1}/{len(pgn_chunks)//cfg.num_workers}, "
                          f"Batch {batch_idx+1}/{len(dataloader)}, Elapsed Time: {elapsed_time:.2f}s")
        
            if early_stop:
                break

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Total batches processed: {total_batches}")
        for key in target_key_list:
            avg_loss = total_losses[key] / max(total_sae_updates[key], 1)
            print(f"  {key}: Total Updates: {total_sae_updates[key]}, Avg Loss: {avg_loss:.4f}")
        print()

        if early_stop:
            break
    
    final_checkpoint = {
        'sae_state_dicts': {key: sae.state_dict() for key, sae in saes.items()},
        'loss_history': loss_history,
        'total_sae_updates': total_sae_updates,
        'training_time': time.time() - start_time,
        'epochs_completed': epoch + 1
    }

    prefix = 'att' if cfg.sae_attention_heads else ('res' if cfg.sae_residual_streams else 'mlp')
    final_save_path = f'maia2-sae/sae/trained_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-{prefix}.pt'
    torch.save(final_checkpoint, final_save_path)

    total_time = time.time() - start_time
    print(f"\nSAE training completed in {total_time:.2f}s")
    for key in target_key_list:
        avg_loss = total_losses[key] / max(total_sae_updates[key], 1)
        print(f"  {key}: Total Updates: {total_sae_updates[key]}, Final Avg Loss: {avg_loss:.4f}")
    
    return saes

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # Supporting Arguments
    parser.add_argument('--data_root', default='maia2_sae/pgn', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--max_ply', default=300, type=int)
    parser.add_argument('--clock_threshold', default=30, type=int)
    parser.add_argument('--chunk_size', default=20000, type=int)
    parser.add_argument('--start_year', default=2023, type=int)
    parser.add_argument('--start_month', default=11, type=int)
    parser.add_argument('--end_year', default=2023, type=int)
    parser.add_argument('--end_month', default=11, type=int)
    parser.add_argument('--from_checkpoint', default=False, type=bool)
    parser.add_argument('--checkpoint_year', default=2018, type=int)
    parser.add_argument('--checkpoint_month', default=12, type=int)
    parser.add_argument('--test_year', default=2023, type=int)
    parser.add_argument('--test_month', default=12, type=int)
    parser.add_argument('--num_cpu_left', default=4, type=int)
    parser.add_argument('--model', default='ViT', type=str)
    parser.add_argument('--max_games_per_elo_range', default=20, type=int)

    # Tunable Arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4096, type=int)
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
    parser.add_argument('--sae_dim', default=16384, type=int)
    parser.add_argument('--num_sae_epochs', default=1, type=int)
    parser.add_argument('--l0_coefficient', default=1, type=float)
    # parser.add_argument('--l1_coefficient', default=0.00005, type=float)
    parser.add_argument('--sae_attention_heads', default=False, type=bool)
    parser.add_argument('--sae_residual_streams', default=True, type=bool)
    parser.add_argument('--sae_mlp_outputs', default=False, type=bool)

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

    trained_model_path = "maia2-ft/model/threats_finetune_0.0001_8192_1e-05/best_model.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(trained_model_path, map_location=torch.device(device))
    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    assert cfg.sae_attention_heads + cfg.sae_mlp_outputs + cfg.sae_residual_streams == 1
    if cfg.sae_attention_heads:
        target_key_list = ['transformer block 0 attention heads', 'transformer block 1 attention heads']
    if cfg.sae_mlp_outputs:
        target_key_list = ['transformer block 0 MLP outputs', 'transformer block 1 MLP outputs']
    if cfg.sae_residual_streams:
        target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states']

    saes = {key: SparseAutoEncoder(activation_dim=cfg.dim_vit, dict_size=cfg.sae_dim) for key in target_key_list}

    current_year = cfg.start_year
    current_month = cfg.start_month

    sae_save_dir = f'maia2-sae/sae'
    if not os.path.exists(sae_save_dir):
        try:
            os.makedirs(sae_save_dir)
            print(f"Directory '{sae_save_dir}' for saving trained saes created successfully.")
        except OSError as e:
            print(f"Error creating directory '{sae_save_dir}': {e}")
    else:
        print(f"Directory '{sae_save_dir}' already exists.")

    while (current_year < cfg.end_year) or (current_year == cfg.end_year and current_month <= cfg.end_month):
        formatted_month = f"{current_month:02d}"
        pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{current_year}-{formatted_month}.pgn"
        
        try:
            pgn_chunks = get_chunks(pgn_path, cfg.chunk_size)
            print(f'Processing data for {current_year}-{formatted_month}')
            print(f'Testing Mixed Elo with {len(pgn_chunks)} chunks from {pgn_path}: ', flush=True)
            trained_saes = train_sae_pipeline(model, saes, cfg, pgn_chunks, all_moves_dict, elo_dict, num_epochs=cfg.num_sae_epochs, buffer_size=8192, l0_coefficient=cfg.l0_coefficient)

        except FileNotFoundError:
            print(f"File not found: {pgn_path}. Skipping this month.")

        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1
    
    sae_state_dicts = {}
    for key, sae in trained_saes.items():
        sae_state_dicts[key] = sae.state_dict()

    if cfg.sae_attention_heads:    
        save_path = f'maia2-sae/sae/trained_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-att.pt'
    if cfg.sae_residual_streams:
        save_path = f'maia2-sae/sae/threat_finetuned_trained_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-res.pt'
    if cfg.sae_mlp_outputs:
        save_path = f'maia2-sae/sae/trained_jrsaes_{cfg.test_year}-{formatted_month}-{cfg.sae_dim}-{cfg.l0_coefficient}-mlp.pt'
    torch.save(sae_state_dicts, save_path)
    print(f"Trained SAEs saved to {save_path}. Finished")
    
    