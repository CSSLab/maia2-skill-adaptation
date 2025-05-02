import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import yaml
import time
import random
import pdb
from multiprocessing import Process, Queue, cpu_count
import threading

from maia2.utils import (
    seed_everything, 
    create_elo_dict,
    get_all_possible_moves,
    read_or_create_chunks,
    decompress_zst,
    readable_time,
    Config,
    parse_args,
)
from maia2.main import MAIA2Model, MAIA2Dataset, process_chunks

class TopKActivation(nn.Module):
    def __init__(self, k=256):
        super().__init__()
        self.k = k
        
    def forward(self, x):
        batch_size = x.size(0)
        feature_size = x.size(1)
        k = min(feature_size, self.k)
        x_relu = F.relu(x)

        threshold_values, _ = torch.topk(x_relu, k, dim=1)
        thresholds = threshold_values[:, -1].unsqueeze(1)
        
        mask = (x_relu >= thresholds).float()
        result = x_relu * mask
        
        return result


class Transcoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, k=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.activation = TopKActivation(k=k) if k > 0 else nn.ReLU()
        
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, x):
        x = x.detach().clone().requires_grad_(True)
        
        hidden_pre = self.encoder(x)
        hidden_post = self.activation(hidden_pre)
        output = self.decoder(hidden_post)
        
        return output, hidden_post
    
    def set_decoder_norm_to_unit_norm(self):
        with torch.no_grad():
            norm = torch.norm(self.decoder.weight.data, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=1e-8)
            self.decoder.weight.data /= norm


class ActivationStore:
    def __init__(self, input_dim, output_dim, buffer_size, device):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.buffer_size = buffer_size
        self.device = device
        
        self.inputs = torch.zeros((buffer_size, input_dim), device=device)
        self.outputs = torch.zeros((buffer_size, output_dim), device=device)
        self.current_index = 0
        self.is_full = False
        self.last_training_index = 0
    
    def add(self, inputs, outputs):
        batch_size = inputs.size(0)
        
        if self.current_index + batch_size > self.buffer_size:
            self.is_full = True
            return True
            
        self.inputs[self.current_index:self.current_index+batch_size] = inputs.detach().clone()
        self.outputs[self.current_index:self.current_index+batch_size] = outputs.detach().clone()
        self.current_index += batch_size
        
        if self.current_index >= self.buffer_size:
            self.is_full = True
        
        return self.is_full
    
    def reset(self):
        self.current_index = 0
        self.is_full = False
        self.last_training_index = 0
    
    def get_sequential_batch(self, batch_size):
        total_samples = self.buffer_size if self.is_full else self.current_index
        
        if self.last_training_index >= total_samples:
            self.last_training_index = 0
            return None, None
        
        end_idx = min(self.last_training_index + batch_size, total_samples)
        
        inputs = self.inputs[self.last_training_index:end_idx].detach().clone()
        outputs = self.outputs[self.last_training_index:end_idx].detach().clone()
        
        self.last_training_index = end_idx
        
        return inputs, outputs


class TranscoderHook:
    def __init__(self, activation_store, transformer_block_idx, device):
        self.activation_store = activation_store
        self.block_idx = transformer_block_idx
        self.device = device
        self.inputs = None
        self.outputs = None
        self.batch_counter = 0
        
    def hook_attn_input(self, module, input, output):
        # Capture output of LayerNorm
        self.inputs = output.detach().clone().to(self.device)
        
    def hook_ff_output(self, module, input, output):
        # Capture output of final Dropout
        self.outputs = output.detach().clone().to(self.device)
        
        if self.inputs is not None:
            self._process_activations()
            
    def _process_activations(self):
        try:
            if len(self.inputs.shape) == 3:
                batch_size, seq_len, input_dim = self.inputs.shape
                
                inputs_reshaped = self.inputs.reshape(-1, input_dim)
                outputs_reshaped = self.outputs.reshape(-1, input_dim)
            else:
                inputs_reshaped = self.inputs
                outputs_reshaped = self.outputs
            
            self.activation_store.add(inputs_reshaped, outputs_reshaped)
            
            self.inputs = None
            self.outputs = None
            self.batch_counter += 1
        except Exception as e:
            print(f"Error in processing activations (batch {self.batch_counter}): {e}")
            self.inputs = None
            self.outputs = None


def train_transcoder(transcoder, activation_store, optimizer, batch_size):
    try:
        inputs, targets = activation_store.get_sequential_batch(batch_size)
        
        if inputs is None or targets is None:
            return 0.0, 0.0, 0.0
        
        optimizer.zero_grad()
        
        with torch.enable_grad():
            outputs, hidden_activations = transcoder(inputs)
            
            mse_loss = F.mse_loss(outputs, targets)
            mse_loss.backward()

        optimizer.step()
        
        transcoder.set_decoder_norm_to_unit_norm()
        
        return mse_loss.item()
    except Exception as e:
        print(f"Error in train_transcoder: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def evaluate_transcoder(transcoder, model, test_dataloader, layer_idx, device):
    original_correct = 0
    transcoder_correct = 0
    total = 0
    
    thread_local = threading.local()
    
    def capture_input_hook(name):
        def hook(module, input, output):
            # Capture output of LayerNorm
            thread_local.input_activations = output.detach().clone()
            return None
        return hook

    def replace_output_hook(name):
        def hook(module, input, output):
            # Replace output of final Dropout
            if hasattr(thread_local, 'modified_activations'):
                return thread_local.modified_activations
            return output
        return hook
    
    if hasattr(model, 'module'):
        transformer_layers = model.module.transformer.elo_layers
    else:
        transformer_layers = model.transformer.elo_layers
    
    model.eval()
    transcoder.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating original model"):
            try:
                if len(batch) == 7:
                    boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win = batch
                elif len(batch) == 8:
                    boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win, _ = batch
                else:
                    print(f"Unexpected batch size: {len(batch)}")
                    continue
                
                boards = boards.to(device)
                moves = moves.to(device)
                elos_self = elos_self.to(device)
                elos_oppo = elos_oppo.to(device)
                legal_moves = legal_moves.to(device)
                
                logits, _, _ = model(boards, elos_self, elos_oppo)
                original_preds = torch.argmax(logits * legal_moves, dim=1)
                
                original_correct += (original_preds == moves).sum().item()
                total += moves.size(0)
            except Exception as e:
                print(f"Error in original evaluation: {e}")
                continue
    
    # Register hooks for capturing and replacing activations
    input_hook = transformer_layers[layer_idx][1].net[0].register_forward_hook(
        capture_input_hook(f'layer{layer_idx}_ff_input')
    )

    output_hook = transformer_layers[layer_idx][1].net[5].register_forward_hook(
        replace_output_hook(f'layer{layer_idx}_ff_output')
    )
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating transcoder model"):
            try:
                if len(batch) == 7:
                    boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win = batch
                elif len(batch) == 8:
                    boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win, _ = batch
                else:
                    print(f"Unexpected batch size: {len(batch)}")
                    continue
                
                boards = boards.to(device)
                moves = moves.to(device)
                elos_self = elos_self.to(device)
                elos_oppo = elos_oppo.to(device)
                legal_moves = legal_moves.to(device)
                
                # First forward pass to get original activations
                _ = model(boards, elos_self, elos_oppo)
                
                # Process the activations with the transcoder
                input_activations = thread_local.input_activations
                
                original_shape = input_activations.shape
                if len(original_shape) == 3:
                    batch_size, seq_len, dim = original_shape
                    flattened_input = input_activations.reshape(-1, dim)
                    
                    new_output, _ = transcoder(flattened_input)
                    
                    thread_local.modified_activations = new_output.reshape(original_shape)
                else:
                    thread_local.modified_activations = transcoder(input_activations)[0]
                
                # Second forward pass with modified activations
                transcoder_logits, _, _ = model(boards, elos_self, elos_oppo)
                transcoder_preds = torch.argmax(transcoder_logits * legal_moves, dim=1)
                
                transcoder_correct += (transcoder_preds == moves).sum().item()
                
                # Clean up thread_local to prevent memory leaks
                del thread_local.input_activations
                del thread_local.modified_activations
                
            except Exception as e:
                print(f"Error in transcoder evaluation: {e}")
                continue
    
    input_hook.remove()
    output_hook.remove()
    
    original_accuracy = original_correct / total if total > 0 else 0
    transcoder_accuracy = transcoder_correct / total if total > 0 else 0
    
    print(f"Original model: {original_correct}/{total} correct predictions ({original_accuracy:.4f})")
    print(f"Transcoder model: {transcoder_correct}/{total} correct predictions ({transcoder_accuracy:.4f})")
    print(f"Delta: {transcoder_accuracy - original_accuracy:.4f}")
    
    return original_accuracy, transcoder_accuracy


def transcoder_preprocess_thread(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict):
    try:
        data, game_count, chunk_count = process_chunks(cfg, pgn_path, pgn_chunks_sublist, elo_dict)
        queue.put([data, game_count, chunk_count])
    except Exception as e:
        print(f"Error in preprocessing thread: {e}")
        queue.put([[], 0, len(pgn_chunks_sublist)])


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to Maia2 config file')
    parser.add_argument('--model_path', type=str, default='../weights.v2.pt', help='Path to pre-trained Maia2 model')
    parser.add_argument('--data_root', type=str, default='/datadrive2/lichess_data', help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='transcoder_models', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=65536, help='Size of activation buffer')
    parser.add_argument('--chunk_size', type=int, default=8000, help='Chunk size for PGN processing')
    parser.add_argument('--max_steps', type=int, default=500000, help='Maximum steps for transcoder training')
    
    parser.add_argument('--hidden_dim', type=int, default=16384, help='Hidden dimension of transcoder')
    parser.add_argument('--k', type=int, default=512, help='Number of active features in TopK ReLU')
    parser.add_argument('--layer_idx', type=int, default=1, help='Transformer layer index to train on (0 or 1)')
    # parser.add_argument('--l1_coefficient', type=float, default=0.1, help='Coefficient for L1 regularization')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval (steps)')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=10000, help='Save interval (steps)')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--queue_length', type=int, default=1, help='Queue length for preprocessing')
    parser.add_argument('--max_games_per_elo_range', type=int, default=100, help='Max games per ELO range')
    parser.add_argument('--eval_samples', type=int, default=100000, help='Number of positions for evaluation step')
    
    parser.add_argument('--clock_threshold', type=int, default=30, help='Clock threshold for move filtering')
    parser.add_argument('--first_n_moves', type=int, default=10, help='Skip first n moves of each game')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        with open(args.config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        cfg = Config(cfg_dict)
    except (FileNotFoundError, yaml.YAMLError):
        print(f"Config file {args.config_path} not found or invalid, using default config")
        cfg = parse_args(args=None)
    
    cfg.data_root = args.data_root
    cfg.chunk_size = args.chunk_size
    cfg.num_workers = args.num_workers
    cfg.verbose = args.verbose
    cfg.clock_threshold = args.clock_threshold
    cfg.first_n_moves = args.first_n_moves
    cfg.max_games_per_elo_range = args.max_games_per_elo_range
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()
    
    print("Loading pre-trained model...")
    trained_model_path = args.model_path
    try:
        ckpt = torch.load(trained_model_path, map_location=device)
        
        model = MAIA2Model(len(all_moves), elo_dict, cfg).to(device)
        
        if all(k.startswith('module.') for k in ckpt['model_state_dict'].keys()):
            model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()})
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        input_dim = cfg.dim_vit
        output_dim = cfg.dim_vit
        print(f"Creating transcoder with input_dim={input_dim}, output_dim={output_dim}, hidden_dim={args.hidden_dim}")
        
        transcoder = Transcoder(
            input_dim=input_dim, 
            output_dim=output_dim, 
            hidden_dim=args.hidden_dim,
            k=args.k
        ).to(device)

        transcoder.train()
        
        for param in transcoder.parameters():
            if param.device != device:
                param.data = param.data.to(device)
                
        optimizer = optim.Adam(transcoder.parameters(), lr=args.lr)
        print("Transcoder created successfully")
    except Exception as e:
        print(f"Error creating transcoder: {e}")
        return
    
    activation_store = ActivationStore(
        input_dim=input_dim,
        output_dim=output_dim,
        buffer_size=args.buffer_size,
        device=device
    )
    
    hooks = []
    transcoder_hook = TranscoderHook(activation_store, args.layer_idx, device)
    
    transformer_layers = model.transformer.elo_layers
    
    attn_hook = transformer_layers[args.layer_idx][1].net[0].register_forward_hook(
        transcoder_hook.hook_attn_input
    )
    hooks.append(attn_hook)

    ff_hook = transformer_layers[args.layer_idx][1].net[5].register_forward_hook(
        transcoder_hook.hook_ff_output
    )
    hooks.append(ff_hook)
    
    num_processes = args.num_workers

    # pdb.set_trace()
    
    total_steps = 0
    best_accuracy = 0.0
    accumulated_samples = 0
    eval_count = 1
    log_count = 1
    save_count = 1
    
    year, month = 2023, 11
    formatted_month = f"{month:02d}"
    pgn_path = f"{args.data_root}/lichess_db_standard_rated_{year}-{formatted_month}.pgn"
    
    if not os.path.exists(pgn_path):
        compressed_path = f"{pgn_path}.zst"
        if os.path.exists(compressed_path):
            print(f"Decompressing {compressed_path}...")
            start_time = time.time()
            decompress_zst(compressed_path, pgn_path)
            print(f"Decompression took {readable_time(time.time() - start_time)}")
        else:
            print(f"Error: Neither {pgn_path} nor {compressed_path} found.")
            return
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        try:
            pgn_chunks = read_or_create_chunks(pgn_path, cfg)
            print(f"Training with {len(pgn_chunks)} chunks from {pgn_path}")
        except Exception as e:
            print(f"Error reading chunks: {e}")
            return
        
        queue = Queue(maxsize=args.queue_length)
        
        pgn_chunks_sublists = []
        for i in range(0, len(pgn_chunks), num_processes):
            pgn_chunks_sublists.append(pgn_chunks[i:i + num_processes])
        
        if len(pgn_chunks_sublists) > 0:
            pgn_chunks_sublist = pgn_chunks_sublists[0]
            worker = Process(target=transcoder_preprocess_thread, args=(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict))
            worker.start()
        else:
            print("No chunks to process")
            continue
        
        num_chunk = 0
        offset = 0
        workers = [worker]
        
        while True:
            try:
                if not queue.empty():
                    if offset + 1 < len(pgn_chunks_sublists):
                        pgn_chunks_sublist = pgn_chunks_sublists[offset + 1]
                        worker = Process(target=transcoder_preprocess_thread, args=(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict))
                        worker.start()
                        workers.append(worker)
                        offset += 1
                    
                    data, game_count, chunk_count = queue.get()
                    num_chunk += chunk_count
                    accumulated_samples += len(data)
                    
                    print(f"Processed chunk {num_chunk}/{len(pgn_chunks)}: {len(data)} positions from {game_count} games")
                    
                    if len(data) > 0:
                        dataset = MAIA2Dataset(data, all_moves_dict, cfg)
                        dataloader = DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=False,
                            num_workers=args.num_workers
                        )
                        
                        print(f"Collecting activations from chunk {num_chunk}/{len(pgn_chunks)}")
                        model.eval()
                        with torch.no_grad():
                            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                                try:
                                    if len(batch) == 7:
                                        boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win = batch
                                    elif len(batch) == 8:
                                        boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win, _ = batch
                                    else:
                                        print(f"Unexpected batch size: {len(batch)}")
                                        continue
                                    
                                    boards = boards.to(device)
                                    elos_self = elos_self.to(device)
                                    elos_oppo = elos_oppo.to(device)
                                    
                                    _ = model(boards, elos_self, elos_oppo)
                                    
                                    if activation_store.is_full:
                                        transcoder.train()
                                        
                                        num_buffer_batches = args.buffer_size // args.batch_size + (1 if args.buffer_size % args.batch_size != 0 else 0)
                                        
                                        for train_step in range(num_buffer_batches):
                                            try:
                                                mse_loss = train_transcoder(
                                                    transcoder,
                                                    activation_store,
                                                    optimizer,
                                                    args.batch_size,
                                                )
                                                
                                                if mse_loss > 0:
                                                    total_steps += 1
                                                    
                                                    if total_steps % args.log_interval == 0:
                                                        print(f"Step {total_steps}, MSE Loss: {mse_loss:.6f}")
                                            except Exception as e:
                                                print(f"Error in training step: {e}")
                                                continue
                                        
                                        activation_store.reset()
                                        
                                        if total_steps > (args.eval_interval * eval_count):
                                            try:
                                                eval_count += 1
                                                test_dataset = MAIA2Dataset(data[0:min(args.eval_samples, len(data))], all_moves_dict, cfg)
                                                test_dataloader = DataLoader(
                                                    test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    num_workers=args.num_workers
                                                )
                                                
                                                original_accuracy, transcoder_accuracy = evaluate_transcoder(
                                                    transcoder, model, test_dataloader, args.layer_idx, device
                                                )
                                                
                                                print(f"Training Step {total_steps}: Original accuracy: {original_accuracy:.4f}, "
                                                      f"Transcoder accuracy: {transcoder_accuracy:.4f}, "
                                                      f"Delta: {transcoder_accuracy - original_accuracy:.4f}")
                                                
                                                if transcoder_accuracy > best_accuracy:
                                                    best_accuracy = transcoder_accuracy
                                                    torch.save({
                                                        'model_state_dict': transcoder.state_dict(),
                                                        'optimizer_state_dict': optimizer.state_dict(),
                                                        'step': total_steps,
                                                        'accuracy': transcoder_accuracy,
                                                        'config': {
                                                            'hidden_dim': args.hidden_dim,
                                                            'top_k': args.k,
                                                            'layer_idx': args.layer_idx
                                                        }
                                                    }, f"{args.save_dir}/best_transcoder_layer{args.layer_idx}_{args.hidden_dim}_{args.k}.pt")
                                            except Exception as e:
                                                print(f"Error in evaluation: {e}")
                                        
                                        if total_steps > (args.save_interval * save_count):
                                            try:
                                                save_count += 1
                                                torch.save({
                                                    'model_state_dict': transcoder.state_dict(),
                                                    'optimizer_state_dict': optimizer.state_dict(),
                                                    'step': total_steps,
                                                    'config': {
                                                        'hidden_dim': args.hidden_dim,
                                                        'top_k': args.k,
                                                        'layer_idx': args.layer_idx
                                                    }
                                                }, f"{args.save_dir}/transcoder_layer{args.layer_idx}_step{total_steps}_{args.hidden_dim}_{args.k}.pt")
                                            except Exception as e:
                                                print(f"Error saving checkpoint: {e}")
                                except Exception as e:
                                    print(f"Error processing batch {batch_idx}: {e}")
                                    continue
                                    
                    if total_steps >= args.max_steps:
                        break
                    
                    if num_chunk >= len(pgn_chunks):
                        break
                
                time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("Interrupted by user. Cleaning up...")
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                break
    
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    
    try:
        torch.save({
            'model_state_dict': transcoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': total_steps,
            'config': {
                'hidden_dim': args.hidden_dim,
                'top_k': args.k,
                'layer_idx': args.layer_idx
            }
        }, f"{args.save_dir}/final_transcoder_layer{args.layer_idx}_{args.hidden_dim}_{args.k}.pt")
    except Exception as e:
        print(f"Error saving final model: {e}")
    
    for hook in hooks:
        hook.remove()
    
    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()