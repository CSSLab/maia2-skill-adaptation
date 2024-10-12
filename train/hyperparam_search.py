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

# _thread_local = threading.local()

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

def evaluate_sae_features_in_train_strategic(split_activations, sae, key, board_fen, function_map):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split_activations = split_activations.to(device)
    sae_weight = sae[key]['encoder_DF.weight'].to(device)
    sae_bias = sae[key]['encoder_DF.bias'].to(device)

    encoded = nn.functional.linear(split_activations, sae_weight, sae_bias)
    encoded = nn.functional.relu(encoded)
    activations = {}
    activations[key] = encoded
    num_cores = min(72, cpu_count())
    n_features = encoded.shape[1]
    results = {}
    total_auc = 0
    num_concepts = len(function_map)

    samples_and_activations = cache_examples_and_activations(list(function_map.values()), list(function_map.keys()), activations, board_fen)
    for concept_name, concept_func in function_map.items():
        sampled_sae_activations = samples_and_activations[concept_name][key]
        ground_truth = torch.cat((samples_and_activations[concept_name]["positives"], samples_and_activations[concept_name]["negatives"]))
        sampled_sae_activations = sampled_sae_activations.cpu()
        ground_truth = ground_truth.cpu()
        args_list = [(i, sampled_sae_activations[:, i], ground_truth) for i in range(n_features)]
            
        with Pool(num_cores) as pool:
            concept_results = list(tqdm.tqdm(pool.imap(process_sae_feature, args_list), total=n_features))

        concept_results.sort(key=lambda x: x[1], reverse=True)
        results[concept_name] = concept_results
        top_feature = concept_results[0]
        print(f"Best feature for {concept_name}:")
        print(f"Feature index: {top_feature[0]}")
        print(f"AUC: {top_feature[1]:.4f}")
        
        total_auc += top_feature[1]

    average_auc = total_auc / num_concepts
    print(f"\nAverage AUC across all strategic concepts: {average_auc:.4f}")
    return results, average_auc

def evaluate_sae_features_in_train_board_state(split_activations, sae, key, board_fen):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split_activations = split_activations.to(device)
    sae_weight = sae[key]['encoder_DF.weight'].to(device)
    sae_bias = sae[key]['encoder_DF.bias'].to(device)

    encoded = nn.functional.linear(split_activations, sae_weight, sae_bias)
    encoded = nn.functional.relu(encoded)
    activations = {}
    activations[key] = encoded
    num_cores = min(72, cpu_count())
    n_features = encoded.shape[1]
    results = {}
    total_auc = 0
    
    # Define the initial positions for each piece as a calibration during training
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    initial_positions = {
        'p': [8, 9, 10, 11, 12, 13, 14, 15],
        'n': [1, 6],
        'b': [2, 5],
        'r': [0, 7],
        'q': [3],
        'k': [4],
        'P': [48, 49, 50, 51, 52, 53, 54, 55],
        'N': [57, 62],
        'B': [58, 61],
        'R': [56, 63],
        'Q': [59],
        'K': [60]
    }

    num_concepts = sum(len(positions) for positions in initial_positions.values())

    samples_and_activations = cache_examples_and_activations_board_states(pieces, activations, board_fen)

    for piece_index, piece in enumerate(pieces):
        for square in initial_positions[piece]:
            concept_name = f"{piece}_is_on_{square_name(square)}"
            print(f"Evaluating concept: {concept_name}")

            sampled_sae_activations = samples_and_activations[piece_index][square][key]
            ground_truth = torch.cat((samples_and_activations[piece_index][square]['positives'], samples_and_activations[piece_index][square]['negatives']))
            sampled_sae_activations = sampled_sae_activations.cpu()
            ground_truth = ground_truth.cpu()
            
            args_list = [(i, sampled_sae_activations[:, i], ground_truth) for i in range(n_features)]
            
            with Pool(num_cores) as pool:
                concept_results = list(tqdm.tqdm(pool.imap(process_sae_feature, args_list), total=n_features))

            concept_results.sort(key=lambda x: x[1], reverse=True)  # Sort by AUC
            results[concept_name] = concept_results
            top_feature = concept_results[0]
            print(f"Best feature for {concept_name}:")
            print(f"Feature index: {top_feature[0]}")
            print(f"AUC: {top_feature[1]:.4f}")
            
            total_auc += top_feature[1]

    average_auc = total_auc / num_concepts
    print(f"\nAverage AUC across all concepts: {average_auc:.4f}")
    return results, average_auc

def train_sae_pipeline(model, cfg, pgn_chunks, all_moves_dict, elo_dict, num_epochs, buffer_size=8192, l1_coefficient=0.00001):
    _enable_activation_hook(model, cfg)

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
    target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states']
    pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{cfg.test_year}-{formatted_month}.pgn"

    # Initialize monitoring variables
    start_time = time.time()
    total_sae_updates = {key: 0 for key in target_key_list}
    total_losses = {key: 0.0 for key in target_key_list}
    total_l2_losses = {key: 0.0 for key in target_key_list}
    total_l1_losses = {key: 0.0 for key in target_key_list}
    
    print(f"Starting SAE hyperparameter tuning pipeline for {num_epochs} epochs")
    
    hidden_dims = [1024, 2048, 4096]
    l1_coefficients = [1e-5, 5e-5, 1e-4]

    for hidden_dim, l1_coefficient in product(hidden_dims, l1_coefficients):
        total_sae_updates = {key: 0 for key in target_key_list}
        print(f"HP Setting: SAE hidden dimension = {hidden_dim}; l1 coefficient = {l1_coefficient}")
        saes = {key: SparseAutoEncoder(activation_dim=cfg.dim_vit, dict_size=hidden_dim) for key in target_key_list}
        optimizers = {key: optim.Adam(saes[key].parameters(), lr=3e-4) for key in target_key_list}
        buffers = {key: ActivationBuffer(buffer_size * cfg.vit_length, cfg.dim_vit) for key in target_key_list}
        
        for epoch in range(num_epochs):
            # epoch_start_time = time.time()
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
                
                for batch_idx, (boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win, board_fen) in enumerate(dataloader):
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    # print(device)
                    boards = boards.to(device)
                    elos_self = elos_self.to(device)
                    elos_oppo = elos_oppo.to(device)
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
                                
                                if total_sae_updates[key] % 1 == 0:
                                    sae_state_dicts = {}
                                    for key, sae in saes.items():
                                        sae_state_dicts[key] = sae.state_dict()
                                    print(f"Feature Calibration for SAE dim = {hidden_dim}, l1 coefficient = {l1_coefficient} on Layer {key}:")
                                    str_res, str_auc = evaluate_sae_features_in_train_strategic(split_activations, sae_state_dicts, key, board_fen, concept_functions)
                                    brd_res, brd_auc = evaluate_sae_features_in_train_board_state(split_activations, sae_state_dicts, key, board_fen)
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
                        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx + 1}, Chunk {i//cfg.num_workers + 1}/{len(pgn_chunks)//cfg.num_workers}, "
                            f"Batch {batch_idx+1}/{len(dataloader)}, Elapsed Time: {elapsed_time:.2f}s")

                    if total_sae_updates[key] >= 1000:
                        break

                if total_sae_updates[key] >= 1000:
                    break

        # Print epoch summary
    #     epoch_time = time.time() - epoch_start_time
    #     print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
    #     print(f"Total batches processed: {total_batches}")
    #     for key in target_key_list:
    #         avg_loss = total_losses[key] / max(total_sae_updates[key], 1)
    #         print(f"  {key}: Total Updates: {total_sae_updates[key]}, Avg Loss: {avg_loss:.4f}")
    #     print()
    
    # total_time = time.time() - start_time
    # print(f"\nSAE training completed in {total_time:.2f}s")
    # for key in target_key_list:
    #     avg_loss = total_losses[key] / max(total_sae_updates[key], 1)
    #     print(f"  {key}: Total Updates: {total_sae_updates[key]}, Final Avg Loss: {avg_loss:.4f}")
    
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
    parser.add_argument('--test_year', default=2019, type=int)
    parser.add_argument('--test_month', default=12, type=int)
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
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model is on device: {next(model.parameters()).device}")

    formatted_month = f"{cfg.test_month:02d}"
    pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{cfg.test_year}-{formatted_month}.pgn"
    pgn_chunks = get_chunks(pgn_path, cfg.chunk_size)
    print(f'Testing Mixed Elo with {len(pgn_chunks)} chunks from {pgn_path}: ', flush=True)

    trained_saes = train_sae_pipeline(model, cfg, pgn_chunks, all_moves_dict, elo_dict, num_epochs=cfg.num_sae_epochs)
    # sae_state_dicts = {}
    # for key, sae in trained_saes.items():
    #     sae_state_dicts[key] = sae.state_dict()
    # save_path = f'sae/trained_saes_{cfg.test_year}-{formatted_month}-new.pt'
    # torch.save(sae_state_dicts, save_path)
    # print(f"Trained SAEs saved to {save_path}. Finished")
    
    