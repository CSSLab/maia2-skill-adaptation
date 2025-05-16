import os
import pickle
import torch
import chess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import json
import time
import threading
from multiprocessing import Process, Queue
import argparse
import heapq
import yaml

from transcoder_train import TopKActivation, Transcoder
from maia2.utils import (
    board_to_tensor, 
    create_elo_dict, 
    get_all_possible_moves, 
    seed_everything, 
    Config, 
    parse_args,
    read_or_create_chunks,
    decompress_zst,
    readable_time,
    get_side_info
)
from maia2.main import MAIA2Model, process_chunks
from torch.utils.data import DataLoader


class MAIA2Dataset(torch.utils.data.Dataset):
    def __init__(self, data, all_moves_dict, cfg):
        self.all_moves_dict = all_moves_dict
        self.data = data
        self.cfg = cfg
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board_fen, move_uci, elo_self, elo_oppo, active_win = self.data[idx]
        
        board = chess.Board(board_fen)
        board_input = board_to_tensor(board)
        
        legal_moves, side_info = get_side_info(board, move_uci, self.all_moves_dict)
        
        move_input = self.all_moves_dict[move_uci]
        
        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info, active_win, board_fen


class FeatureActivationCollector:
    def __init__(self, 
                 cfg, 
                 model_path,
                 transcoder_paths,
                 circuits_path,
                 data_root,
                 output_dir="feature_activation_examples",
                 device="cuda:0",
                 top_k=50,
                 num_positions=100000,
                 batch_size=128,
                 num_workers=16,
                 queue_length=1,
                 seed=42):
        
        seed_everything(seed)
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.top_k = top_k
        self.num_positions = num_positions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue_length = queue_length
        self.data_root = data_root
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.elo_dict = create_elo_dict()
        self.all_moves = get_all_possible_moves()
        self.moves_dict = {move: i for i, move in enumerate(self.all_moves)}
        self.reverse_moves_dict = {i: move for move, i in self.moves_dict.items()}
        
        print("Loading model...")
        self.model = self._load_model(model_path)
        
        print("Loading transcoders...")
        self.transcoders = self._load_transcoders(transcoder_paths)
        
        print("Loading skill-relevant circuits...")
        self.circuits = self._load_circuits(circuits_path)
        
        self._setup_hooks()
        
        self.target_features = self._get_target_features()
        self.feature_top_boards = self._initialize_top_boards_trackers()

        self.positions_processed = 0
        
        print(f"Initialized feature collector for {len(self.target_features)} features")

    def _load_model(self, model_path):
        model = MAIA2Model(len(self.all_moves), self.elo_dict, self.cfg)
        ckpt = torch.load(model_path, map_location=self.device)
        
        if all(k.startswith('module.') for k in ckpt['model_state_dict'].keys()):
            model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()})
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            
        return model.eval().to(self.device)

    def _load_transcoders(self, transcoder_paths):
        transcoders = {}
        
        for layer in [0, 1]:
            model_path = transcoder_paths[layer]
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Transcoder file not found: {model_path}")
            
            transcoder_state = torch.load(model_path, map_location=self.device)
            transcoders[layer] = self._create_transcoder_from_state(transcoder_state)
            print(f"Loaded transcoder for layer {layer} from {model_path}")
        
        return transcoders

    def _create_transcoder_from_state(self, state_dict):
        config = state_dict['config']
        transcoder = Transcoder(
            input_dim=self.cfg.dim_vit,
            output_dim=self.cfg.dim_vit,
            hidden_dim=config['hidden_dim'],
            k=config['top_k']
        ).to(self.device)
        
        transcoder.load_state_dict(state_dict['model_state_dict'])
        transcoder.eval()
        return transcoder

    def _load_circuits(self, circuits_path):
        with open(circuits_path, 'rb') as f:
            circuits = pickle.load(f)
        return circuits

    def _setup_hooks(self):
        self.activation_hooks = []
        self.activations = {}
        
        for layer_idx in self.transcoders.keys():
            def get_hook(layer_i):
                def hook(module, input, output):
                    self.activations[f'layer{layer_i}_pre_mlp'] = output.detach()
                    return output
                return hook
            
            hook = self.model.transformer.elo_layers[layer_idx][1].net[0].register_forward_hook(
                get_hook(layer_idx)
            )
            self.activation_hooks.append(hook)

    def _cleanup_hooks(self):
        for hook in self.activation_hooks:
            hook.remove()

    def _get_target_features(self):
        target_features = set()
        
        for circuit in self.circuits:
            target_features.add(f"layer1_{circuit['layer1_feature']}")
            
            for connection in circuit['connections']:
                target_features.add(f"layer0_{connection['layer0_feature']}")
        
        return list(target_features)

    def _initialize_top_boards_trackers(self):
        feature_top_boards = {}
        
        for feature_key in self.target_features:
            feature_top_boards[feature_key] = []
        
        return feature_top_boards

    def preprocess_thread(self, queue, pgn_chunks_sublist, pgn_path):
        try:
            data, game_count, chunk_count = process_chunks(self.cfg, pgn_path, pgn_chunks_sublist, self.elo_dict)
            queue.put([data, game_count, chunk_count])
        except Exception as e:
            print(f"Error in preprocessing thread: {e}")
            queue.put([[], 0, len(pgn_chunks_sublist)])

    def process_batch(self, batch):
        try:
            if len(batch) == 8:
                boards, moves, elos_self, elos_oppo, legal_moves, side_info, active_win, fens = batch
            else:
                print(f"Unexpected batch size: {len(batch)}")
                return False

            boards = boards.to(self.device)
            elos_self = elos_self.to(self.device)
            elos_oppo = elos_oppo.to(self.device)
            moves = moves.to(self.device)
            legal_moves = legal_moves.to(self.device)

            with torch.no_grad():
                logits, _, _ = self.model(boards, elos_self, elos_oppo)
                
                pred_move_indices = torch.argmax(logits * legal_moves, dim=1)
                
                pred_move_ucis = [self.reverse_moves_dict.get(idx.item(), "unknown") for idx in pred_move_indices]
                
                position_data = []
                for i in range(len(boards)):
                    fen = fens[i]
                    position_data.append({
                        'fen': fen,
                        'fen_key': fen.split()[0],  # First part of FEN to check for duplicates
                        'elo_self': int(elos_self[i].item()),
                        'elo_oppo': int(elos_oppo[i].item()),
                        'label': self.reverse_moves_dict.get(moves[i].item(), "unknown"),
                        'move': pred_move_ucis[i]  # Use model's predicted move
                    })
                
                for layer_idx in self.transcoders.keys():
                    transcoder = self.transcoders[layer_idx]
                    
                    pre_mlp_acts = self.activations[f'layer{layer_idx}_pre_mlp']
                    
                    for token_idx in range(pre_mlp_acts.shape[1]):
                        token_acts = pre_mlp_acts[:, token_idx]
                        _, feature_acts = transcoder(token_acts)
                        
                        # For each tracked feature, update top_boards
                        for feature_idx in range(feature_acts.shape[1]):
                            feature_key = f"layer{layer_idx}_{feature_idx}"
                            
                            if feature_key in self.target_features:
                                if not hasattr(self, 'feature_unique_fens'):
                                    self.feature_unique_fens = {key: set() for key in self.target_features}
                                
                                for i, activation in enumerate(feature_acts[:, feature_idx].cpu().numpy()):
                                    activation_value = float(activation)
                                    position_info = position_data[i]
                                    fen_key = position_info['fen_key']

                                    if fen_key in self.feature_unique_fens[feature_key]:
                                        continue
                                    
                                    if len(self.feature_top_boards[feature_key]) < self.top_k:
                                        # Add to set of seen FENs
                                        self.feature_unique_fens[feature_key].add(fen_key)
                                        heapq.heappush(self.feature_top_boards[feature_key], 
                                                    (activation_value, i, self.positions_processed + i))
                                    else:
                                        if activation_value > self.feature_top_boards[feature_key][0][0]:
                                            _, _, old_pos_idx = self.feature_top_boards[feature_key][0]
                                            if hasattr(self, 'all_positions') and old_pos_idx < len(self.all_positions):
                                                old_fen_key = self.all_positions[old_pos_idx].get('fen_key', '')
                                                self.feature_unique_fens[feature_key].discard(old_fen_key)
                                            
                                            self.feature_unique_fens[feature_key].add(fen_key)
                                            heapq.heappushpop(self.feature_top_boards[feature_key], 
                                                            (activation_value, i, self.positions_processed + i))

            if not hasattr(self, 'all_positions'):
                self.all_positions = []
            self.all_positions.extend(position_data)
                
            self.positions_processed += len(boards)
            if self.positions_processed >= self.num_positions:
                return True  
            
            return False
        except Exception as e:
            print(f"Error in process_batch: {e}")
            import traceback
            traceback.print_exc()
            return False

    def collect_feature_activations_from_lichess(self):
        year, month = 2023, 11
        formatted_month = f"{month:02d}"
        pgn_path = f"{self.data_root}/lichess_db_standard_rated_{year}-{formatted_month}.pgn"
        
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
        
        try:
            pgn_chunks = read_or_create_chunks(pgn_path, self.cfg)
            print(f"Processing with {len(pgn_chunks)} chunks from {pgn_path}")
        except Exception as e:
            print(f"Error reading chunks: {e}")
            return
        
        queue = Queue(maxsize=self.queue_length)
        
        pgn_chunks_sublists = []
        for i in range(0, len(pgn_chunks), self.num_workers):
            pgn_chunks_sublists.append(pgn_chunks[i:i + self.num_workers])
        
        if len(pgn_chunks_sublists) > 0:
            pgn_chunks_sublist = pgn_chunks_sublists[0]
            worker = Process(target=self.preprocess_thread, args=(queue, pgn_chunks_sublist, pgn_path))
            worker.start()
        else:
            print("No chunks to process")
            return
        
        num_chunk = 0
        offset = 0
        workers = [worker]
        
        stop_processing = False
        
        while not stop_processing:
            try:
                if not queue.empty():
                    if offset + 1 < len(pgn_chunks_sublists):
                        pgn_chunks_sublist = pgn_chunks_sublists[offset + 1]
                        worker = Process(target=self.preprocess_thread, args=(queue, pgn_chunks_sublist, pgn_path))
                        worker.start()
                        workers.append(worker)
                        offset += 1
                    
                    data, game_count, chunk_count = queue.get()
                    num_chunk += chunk_count
                    
                    print(f"Processed chunk {num_chunk}/{len(pgn_chunks)}: {len(data)} positions from {game_count} games")
                    
                    if len(data) > 0:
                        dataset = MAIA2Dataset(data, self.moves_dict, self.cfg)
                        dataloader = DataLoader(
                            dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=False,
                            num_workers=0
                        )
                        
                        print(f"Collecting activations from chunk {num_chunk}/{len(pgn_chunks)}")
                        self.model.eval()
                        
                        with torch.no_grad():
                            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                                try:
                                    stop_batch = self.process_batch(batch)
                                    if stop_batch or self.positions_processed >= self.num_positions:
                                        stop_processing = True
                                        break
                                except Exception as e:
                                    print(f"Error processing batch {batch_idx}: {e}")
                                    continue
                    
                    if num_chunk >= len(pgn_chunks):
                        stop_processing = True
                
                time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("Interrupted by user. Cleaning up...")
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Clean up
        for worker in workers:
            if worker.is_alive():
                worker.terminate()

    def save_results(self):
        os.makedirs(os.path.join(self.output_dir, "circuit_features"), exist_ok=True)

        feature_results = {}
        for feature_key, heap in self.feature_top_boards.items():
            sorted_heap = sorted(heap, key=lambda x: x[0], reverse=True)
            
            sorted_results = []
            for activation, _, position_idx in sorted_heap:
                if position_idx < len(self.all_positions):
                    position_info = self.all_positions[position_idx]
                    sorted_results.append((activation, position_info))
            
            feature_results[feature_key] = {
                'activations': [float(act) for act, _ in sorted_results],
                'positions': [pos for _, pos in sorted_results]
            }
        
        with open(os.path.join(self.output_dir, "feature_top_boards.json"), 'w') as f:
            json.dump(feature_results, f, indent=2)

        circuit_features = {}
        
        for circuit_idx, circuit in enumerate(self.circuits):
            circuit_dir = os.path.join(self.output_dir, f"circuit_{circuit_idx}")
            os.makedirs(circuit_dir, exist_ok=True)
            
            layer1_key = f"layer1_{circuit['layer1_feature']}"
            circuit_data = {
                'layer1_feature': circuit['layer1_feature'],
                'layer1_top_boards': feature_results.get(layer1_key, {'activations': [], 'positions': []}),
                'connections': []
            }
            
            for connection in circuit['connections']:
                layer0_feature = connection['layer0_feature']
                layer0_key = f"layer0_{layer0_feature}"
                connection_data = {
                    'layer0_feature': layer0_feature,
                    'input_invariant_connection': connection['input_invariant_connection'],
                    'is_skill_relevant': connection['is_skill_relevant'],
                    'top_boards': feature_results.get(layer0_key, {'activations': [], 'positions': []})
                }
                circuit_data['connections'].append(connection_data)
            
            with open(os.path.join(circuit_dir, "feature_activations.json"), 'w') as f:
                json.dump(circuit_data, f, indent=2)
            
            circuit_features[f"circuit_{circuit_idx}"] = circuit_data
        
        with open(os.path.join(self.output_dir, "circuit_features.json"), 'w') as f:
            json.dump(circuit_features, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")

    def run(self):
        try:
            print("Collecting feature activations from Lichess data...")
            self.collect_feature_activations_from_lichess()
            
            print("Saving results...")
            self.save_results()
            
            print(f"Done! Processed {self.positions_processed} positions.")
            
        finally:
            self._cleanup_hooks()


def main():
    parser = argparse.ArgumentParser(description="Collect top activating positions for features in skill-relevant circuits")
    
    parser.add_argument('--config_path', type=str, default='config.yaml', 
                        help='Path to config YAML file')
    parser.add_argument('--model_path', type=str, default='../weights.v2.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='/datadrive2/lichess_data',
                        help='Path to data directory containing Lichess PGN files')
    parser.add_argument('--circuits_path', type=str, default='skill_relevant_circuits_e5_threat/skill_relevant_circuits.pkl',
                        help='Path to saved circuits')
    parser.add_argument('--output_dir', type=str, default='feature_activation_examples',
                        help='Directory to save results')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Number of top activating positions to save per feature')
    parser.add_argument('--num_positions', type=int, default=100000,
                        help='Number of positions to process')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of preprocessing workers')
    parser.add_argument('--queue_length', type=int, default=1,
                        help='Queue length for preprocessing')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    try:
        with open(args.config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        cfg = Config(cfg_dict)
    except (FileNotFoundError, yaml.YAMLError):
        print(f"Config file {args.config_path} not found or invalid, using default config")
        cfg = parse_args(args=None)
    
    cfg.data_root = args.data_root
    cfg.chunk_size = 8000
    cfg.max_ply = 1000  
    cfg.clock_threshold = 30
    cfg.first_n_moves = 10
    cfg.max_games_per_elo_range = 100
    cfg.verbose = True
    cfg.num_workers = args.num_workers
    
    transcoder_paths = {
        0: "best_transcoder_models/best_transcoder_layer0_32768_512.pt",
        1: "best_transcoder_models/best_transcoder_layer1_32768_512.pt"
    }
    
    collector = FeatureActivationCollector(
        cfg=cfg,
        model_path=args.model_path,
        transcoder_paths=transcoder_paths,
        circuits_path=args.circuits_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        top_k=args.top_k,
        num_positions=args.num_positions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        queue_length=args.queue_length,
        device=args.device,
        seed=args.seed
    )
    
    collector.run()


if __name__ == "__main__":
    main()