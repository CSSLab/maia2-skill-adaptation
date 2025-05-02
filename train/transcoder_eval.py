import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import yaml
import threading
import chess

from maia2.utils import (
    seed_everything,
    create_elo_dict,
    get_all_possible_moves,
    read_or_create_chunks,
    board_to_tensor,
    Config,
    parse_args,
)
from maia2.main import MAIA2Model, MAIA2Dataset, process_chunks
from transcoder_train import Transcoder, TopKActivation


class TranscoderEvaluator:
    def __init__(self, model_path, transcoder_path, layer_idx, cfg, device='cuda:0'):
        """
        Initialize the evaluator with model and transcoder paths
        
        Args:
            model_path: Path to the pre-trained Maia2 model
            transcoder_path: Path to the trained transcoder model
            layer_idx: Index of the layer where transcoder is applied
            cfg: Configuration object
            device: Device to run models on
        """
        self.model_path = model_path
        self.transcoder_path = transcoder_path
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.all_moves = get_all_possible_moves()
        self.all_moves_dict = {move: i for i, move in enumerate(self.all_moves)}
        self.elo_dict = create_elo_dict()
        
        print(f"Loading pre-trained model from {model_path}...")
        self.model = self._load_model()
        
        print(f"Loading transcoder from {transcoder_path}...")
        self.transcoder = self._load_transcoder()
        
        # Thread local storage for activation hooks
        self.thread_local = threading.local()
    
    def _load_model(self):
        """Load and prepare the Maia2 model"""
        try:
            ckpt = torch.load(self.model_path, map_location=self.device)
            
            model = MAIA2Model(len(self.all_moves), self.elo_dict, self.cfg).to(self.device)
            
            if all(k.startswith('module.') for k in ckpt['model_state_dict'].keys()):
                model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()})
            else:
                model.load_state_dict(ckpt['model_state_dict'])
            
            model.eval()
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_transcoder(self):
        """Load and prepare the Transcoder model"""
        try:
            ckpt = torch.load(self.transcoder_path, map_location=self.device)
            
            config = ckpt.get('config', {})
            hidden_dim = config.get('hidden_dim', 16384)
            top_k = config.get('top_k', 256)
            
            input_dim = self.cfg.dim_vit
            output_dim = self.cfg.dim_vit
            
            transcoder = Transcoder(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                k=top_k
            ).to(self.device)
            
            transcoder.load_state_dict(ckpt['model_state_dict'])
            transcoder.eval()
            
            print(f"Transcoder loaded successfully with hidden_dim={hidden_dim}, k={top_k}")
            return transcoder
        except Exception as e:
            print(f"Error loading transcoder: {e}")
            raise
    
    def setup_hooks(self):
        """
        Set up hooks for capturing and replacing activations in the model
        Returns the hooks that were registered
        """
        self.thread_local = threading.local()
        
        def capture_input_hook(name):
            def hook(module, input, output):
                # Capture output of LayerNorm
                self.thread_local.input_activations = output.detach().clone()
                return None
            return hook

        def replace_output_hook(name):
            def hook(module, input, output):
                # Replace output of final Dropout
                if hasattr(self.thread_local, 'modified_activations'):
                    return self.thread_local.modified_activations
                return output
            return hook
        
        if hasattr(self.model, 'module'):
            transformer_layers = self.model.module.transformer.elo_layers
        else:
            transformer_layers = self.model.transformer.elo_layers
        
        input_hook = transformer_layers[self.layer_idx][1].net[0].register_forward_hook(
            capture_input_hook(f'layer{self.layer_idx}_ff_input')
        )

        output_hook = transformer_layers[self.layer_idx][1].net[5].register_forward_hook(
            replace_output_hook(f'layer{self.layer_idx}_ff_output')
        )
        
        return [input_hook, output_hook]

    def evaluate_policy_agreement(self, test_dataloader, num_samples=None):
        """
        Evaluate policy head agreement between original model and transcoder model
        by enumerating all ELO combinations for each position, and collect value predictions
        """
        hooks = self.setup_hooks()
        
        # Dictionary to store agreement counts for each ELO combination
        agreement_counts = {}
        for elo_self in range(11):  # 0 to 10 for ELO categories
            for elo_oppo in range(11):
                agreement_counts[(elo_self, elo_oppo)] = {"agree": 0, "total": 0}
        
        # Lists to store value predictions
        original_values = []
        transcoder_values = []
        
        count = 0
        
        self.model.eval()
        self.transcoder.eval()
        
        try:
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Evaluating positions"):
                    if num_samples is not None and count >= num_samples:
                        break
                    
                    try:
                        if len(batch) == 7:
                            boards, moves, _, _, legal_moves, side_info, active_win = batch
                        elif len(batch) == 8:
                            boards, moves, _, _, legal_moves, side_info, active_win, _ = batch
                        else:
                            print(f"Unexpected batch size: {len(batch)}")
                            continue
                        
                        current_batch_size = boards.size(0)
                        if num_samples is not None and count + current_batch_size > num_samples:
                            # Trim batch to desired sample count
                            idx = num_samples - count
                            boards = boards[:idx]
                            moves = moves[:idx]
                            legal_moves = legal_moves[:idx]
                            current_batch_size = idx
                        
                        boards = boards.to(self.device)
                        moves = moves.to(self.device)
                        legal_moves = legal_moves.to(self.device)
                        
                        # For value head predictions, use middle ELO values (5,5)
                        middle_elo = 5
                        elos_self_value = torch.full((current_batch_size,), middle_elo, 
                                            dtype=torch.long, device=self.device)
                        elos_oppo_value = torch.full((current_batch_size,), middle_elo, 
                                            dtype=torch.long, device=self.device)
                        
                        # Get original value predictions
                        _, _, orig_values = self.model(boards, elos_self_value, elos_oppo_value)
                        
                        # Process with transcoder for value head
                        input_activations = self.thread_local.input_activations
                        
                        original_shape = input_activations.shape
                        if len(original_shape) == 3:
                            batch_size, seq_len, dim = original_shape
                            flattened_input = input_activations.reshape(-1, dim)
                            
                            new_output, _ = self.transcoder(flattened_input)
                            
                            self.thread_local.modified_activations = new_output.reshape(original_shape)
                        else:
                            self.thread_local.modified_activations = self.transcoder(input_activations)[0]
                        
                        # Get transcoder value predictions
                        _, _, trans_values = self.model(boards, elos_self_value, elos_oppo_value)
                        
                        # Save value predictions
                        original_values.extend(orig_values.cpu().numpy())
                        transcoder_values.extend(trans_values.cpu().numpy())
                        
                        # Clean up for value head evaluation
                        del self.thread_local.input_activations
                        del self.thread_local.modified_activations
                        
                        # For policy head agreement, enumerate all ELO combinations
                        for elo_self in range(11):
                            for elo_oppo in range(11):
                                # Create tensors with the current ELO values
                                elos_self = torch.full((current_batch_size,), elo_self, 
                                                    dtype=torch.long, device=self.device)
                                elos_oppo = torch.full((current_batch_size,), elo_oppo, 
                                                    dtype=torch.long, device=self.device)
                                
                                # Evaluate original model
                                original_logits, _, _ = self.model(boards, elos_self, elos_oppo)
                                original_preds = torch.argmax(original_logits * legal_moves, dim=1)
                                
                                # Process with transcoder
                                input_activations = self.thread_local.input_activations
                                
                                original_shape = input_activations.shape
                                if len(original_shape) == 3:
                                    batch_size, seq_len, dim = original_shape
                                    flattened_input = input_activations.reshape(-1, dim)
                                    
                                    new_output, _ = self.transcoder(flattened_input)
                                    
                                    self.thread_local.modified_activations = new_output.reshape(original_shape)
                                else:
                                    self.thread_local.modified_activations = self.transcoder(input_activations)[0]
                                
                                # Evaluate transcoder model
                                transcoder_logits, _, _ = self.model(boards, elos_self, elos_oppo)
                                transcoder_preds = torch.argmax(transcoder_logits * legal_moves, dim=1)
                                
                                # Update agreement counts
                                agreement = (original_preds == transcoder_preds).cpu().numpy()
                                agreement_counts[(elo_self, elo_oppo)]["total"] += current_batch_size
                                agreement_counts[(elo_self, elo_oppo)]["agree"] += agreement.sum()
                                
                                # Clean up thread_local to prevent memory leaks
                                del self.thread_local.input_activations
                                del self.thread_local.modified_activations
                        
                        count += current_batch_size
                        
                    except Exception as e:
                        print(f"Error in batch evaluation: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        print(f"Evaluated {count} positions with all ELO combinations")
        return agreement_counts, original_values, transcoder_values

    def generate_policy_agreement_heatmap(self, agreement_counts, output_path='policy_agreement.png'):
        """
        Generate a heatmap showing the agreement between original and transcoder models
        across different Elo ranges with larger font size and coolwarm colormap
        """
        # Convert agreement counts to a matrix
        n_elos = 11  # 0 to 10
        agreement_matrix = np.zeros((n_elos, n_elos))
        
        for elo_self in range(n_elos):
            for elo_oppo in range(n_elos):
                key = (elo_self, elo_oppo)
                if agreement_counts[key]["total"] > 0:
                    agreement_matrix[elo_self, elo_oppo] = agreement_counts[key]["agree"] / agreement_counts[key]["total"]
        
        # Set font sizes
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(12, 10))
        
        # Generate heatmap with coolwarm colormap
        ax = sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                        vmin=0.8, vmax=1.0, cbar_kws={'label': 'Degree of Agreement'}, 
                        annot_kws={"size": 14})
        
        # Set font sizes for labels and title
        ax.set_xlabel("Opponent Player's Skill Level", fontsize=24)
        ax.set_ylabel("Active Player's Skill Level", fontsize=24)
        ax.set_title("Policy Head Agreement: Original vs Transcoder", fontsize=28)
        
        # Set tick labels
        ax.set_xticks(np.arange(n_elos) + 0.5)
        ax.set_yticks(np.arange(n_elos) + 0.5)
        ax.set_xticklabels(range(n_elos), fontsize=18)
        ax.set_yticklabels(range(n_elos), fontsize=18)
        
        # Improve colorbar font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Degree of Agreement', fontsize=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Policy agreement heatmap saved to {output_path}")
        plt.close()

    def generate_value_head_qqplot(self, original_values, transcoder_values, output_path='value_qqplot.png'):
        """
        Generate a QQ-plot comparing the value predictions from original and transcoder models
        with larger font sizes
        """
        # Convert to numpy arrays if they aren't already
        original_values = np.array(original_values)
        transcoder_values = np.array(transcoder_values)
        
        # Normalize to [0, 1] range (from [-1, 1])
        original_values_norm = (original_values + 1) / 2
        transcoder_values_norm = (transcoder_values + 1) / 2
        
        # Sort values for QQ-plot
        original_sorted = np.sort(original_values_norm)
        transcoder_sorted = np.sort(transcoder_values_norm)
        
        # Set font sizes
        plt.rcParams.update({'font.size': 16})
        
        # Generate QQ-plot
        plt.figure(figsize=(10, 10))
        plt.scatter(original_sorted, transcoder_sorted, alpha=0.6, color='steelblue', s=20)
        
        # Add reference line
        min_val = min(original_sorted.min(), transcoder_sorted.min())
        max_val = max(original_sorted.max(), transcoder_sorted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Original Win Rate Prediction', fontsize=24)
        plt.ylabel('Transcoder Reconstructed Win Rate Prediction', fontsize=24)
        plt.title('Value Head QQ-Plot: Original vs Transcoder', fontsize=28)
        
        # Set tick label size
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Value head QQ-plot saved to {output_path}")
        plt.close()
    
    def prepare_test_data(self, pgn_path, num_samples=16384):
        """
        Prepare test data from PGN file
        
        Args:
            pgn_path: Path to PGN file
            num_samples: Number of samples to use for testing
            
        Returns:
            DataLoader with test data
        """
        print(f"Preparing test data from {pgn_path}")
        
        # Get chunks for processing
        pgn_chunks = read_or_create_chunks(pgn_path, self.cfg)
        
        # Process just enough chunks to get desired number of samples
        data = []
        chunks_used = 0
        
        for i, chunk in enumerate(pgn_chunks):
            chunk_data, _, _ = process_chunks(self.cfg, pgn_path, [chunk], self.elo_dict)
            data.extend(chunk_data)
            chunks_used += 1
            
            if len(data) >= num_samples:
                break
        
        print(f"Processed {chunks_used} chunks, got {len(data)} positions")
        
        # Truncate to exactly num_samples
        if len(data) > num_samples:
            data = data[:num_samples]
        
        # Create dataset and dataloader
        dataset = MAIA2Dataset(data, self.all_moves_dict, self.cfg)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.num_workers
        )
        
        return dataloader


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to Maia2 config file')
    parser.add_argument('--model_path', type=str, default='../weights.v2.pt', help='Path to pre-trained Maia2 model')
    parser.add_argument('--transcoder_path', type=str, default='best_transcoder_models/best_transcoder_layer0_16384_256.pt', help='Path to trained transcoder model')
    parser.add_argument('--data_root', type=str, default='/datadrive2/lichess_data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='transcoder_evaluation_results', help='Directory to save outputs')
    parser.add_argument('--layer_idx', type=int, default=0, help='Maia-2 Transformer layer index')
    parser.add_argument('--num_samples', type=int, default=8192, help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    try:
        with open(args.config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        cfg = Config(cfg_dict)
    except (FileNotFoundError, yaml.YAMLError):
        print(f"Config file {args.config_path} not found or invalid, using default config")
        cfg = parse_args(None)
    
    # Update config with command line arguments
    cfg.data_root = args.data_root
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    
    # Initialize evaluator
    evaluator = TranscoderEvaluator(
        model_path=args.model_path,
        transcoder_path=args.transcoder_path,
        layer_idx=args.layer_idx,
        cfg=cfg
    )
    
    # Prepare test data
    year, month = 2023, 11
    formatted_month = f"{month:02d}"
    pgn_path = f"{args.data_root}/lichess_db_standard_rated_{year}-{formatted_month}.pgn"
    
    test_dataloader = evaluator.prepare_test_data(pgn_path, args.num_samples)
    
    # Evaluate and generate plots
    agreement_counts, original_values, transcoder_values = evaluator.evaluate_policy_agreement(
        test_dataloader, args.num_samples
    )
    
    evaluator.generate_policy_agreement_heatmap(
        agreement_counts,
        os.path.join(args.output_dir, f"policy_agreement_layer{args.layer_idx}.png")
    )
    
    evaluator.generate_value_head_qqplot(
        original_values, 
        transcoder_values,
        os.path.join(args.output_dir, f"value_qqplot_layer{args.layer_idx}.png")
    )
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()