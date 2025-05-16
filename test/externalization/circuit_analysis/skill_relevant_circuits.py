import os
import pickle
import torch
import chess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from functools import partial
from matplotlib.colors import LinearSegmentedColormap

from transcoder_train import TopKActivation, Transcoder
from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves, get_side_info, seed_everything, Config, parse_args
from maia2.main import MAIA2Model

class SkillRelevantCircuitFinder:
    def __init__(self, 
                 cfg, 
                 original_model_path, 
                 finetuned_model_path,
                 transcoder_path, 
                 data_path,
                 output_dir="skill_relevant_circuits_e5_threat",
                 device="cuda:0",
                 seed=42):
        
        seed_everything(seed)
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.elo_dict = create_elo_dict()
        self.all_moves = get_all_possible_moves()
        self.moves_dict = {move: i for i, move in enumerate(self.all_moves)}
        self.skill_levels = list(range(len(self.elo_dict)))
        
        print("Loading original model...")
        self.model_original = self._load_model(original_model_path)
        print("Loading finetuned model...")
        self.model_finetuned = self._load_model(finetuned_model_path)
        
        print("Loading transcoders...")
        self.transcoders = self._load_transcoders(transcoder_path)
        self._setup_hooks()
        
        self.data_path = data_path
        print(f"Data path: {data_path}")
        self.positions = self._load_positions()

        self.feature_activations_original = {}
        self.feature_activations_finetuned = {}
        self.feature_connections = None
        self.skill_relevant_features = None
        self.skill_relevant_circuits = None

    def _load_model(self, model_path):
        model = MAIA2Model(len(self.all_moves), self.elo_dict, self.cfg)
        ckpt = torch.load(model_path, map_location=self.device)
        
        if all(k.startswith('module.') for k in ckpt['model_state_dict'].keys()):
            model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()})
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            
        return model.eval().to(self.device)

    def _load_transcoders(self, transcoder_path):
        transcoders = {}
        
        layer_paths = {
            0: "best_transcoder_models/best_transcoder_layer0_32768_512.pt",
            1: "best_transcoder_models/best_transcoder_layer1_32768_512.pt"
        }
        
        for layer in [0, 1]:
            model_path = layer_paths[layer]
            
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

    def _setup_hooks(self):
        self.activation_hooks = []
        self.activations_original = {}
        self.activations_finetuned = {}
        
        # Hooks for original model
        for layer_idx in self.transcoders.keys():
            # Pre-MLP hook
            def get_hook_orig(layer_i):
                def hook(module, input, output):
                    self.activations_original[f'layer{layer_i}_pre_mlp'] = output.detach()
                    return output
                return hook
            
            hook = self.model_original.transformer.elo_layers[layer_idx][1].net[0].register_forward_hook(
                get_hook_orig(layer_idx)
            )
            self.activation_hooks.append(hook)
        
        # Hooks for finetuned model
        for layer_idx in self.transcoders.keys():
            # Pre-MLP hook
            def get_hook_ft(layer_i):
                def hook(module, input, output):
                    self.activations_finetuned[f'layer{layer_i}_pre_mlp'] = output.detach()
                    return output
                return hook
            
            hook = self.model_finetuned.transformer.elo_layers[layer_idx][1].net[0].register_forward_hook(
                get_hook_ft(layer_idx)
            )
            self.activation_hooks.append(hook)

    def _cleanup_hooks(self):
        for hook in self.activation_hooks:
            hook.remove()

    def _load_positions(self):
        # Load positions with e5 defensive threat
        dataset_file = os.path.join(self.data_path, "train_moves_e5_defensive_threat.pkl")
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file {dataset_file} not found")
            
        with open(dataset_file, 'rb') as f:
            positions_raw = pickle.load(f)
            
        print(f"Loaded {len(positions_raw)} positions from {dataset_file}")
        
        positions = []
        seen_fens = set()
        
        for pos in positions_raw:
            if not isinstance(pos, tuple) or len(pos) < 2:
                continue
                
            fen = pos[0]
            move = pos[1]
            
            if fen in seen_fens:
                continue
            seen_fens.add(fen)
            
            positions.append({
                'fen': fen,
                'move': move,
            })
        
        print(f"Processed {len(positions)} unique positions")
        return positions

    def compute_feature_activations(self, batch_size=8192):
        # Compute transcoder feature activations for original and finetuned models
        cache_path = os.path.join(self.output_dir, "feature_activations.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached feature activations from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.feature_activations_original = data['original']
                self.feature_activations_finetuned = data['finetuned']
            return
        
        for layer in self.transcoders:
            transcoder = self.transcoders[layer]
            hidden_dim = transcoder.encoder.weight.shape[0]
            
            self.feature_activations_original[layer] = {
                'per_position': np.zeros((hidden_dim, len(self.skill_levels), len(self.positions))),
                'mean': None,
                'diff_variance': None,
            }
            
            self.feature_activations_finetuned[layer] = {
                'per_position': np.zeros((hidden_dim, len(self.skill_levels), len(self.positions))),
                'mean': None,
                'diff_variance': None,
            }
        
        num_batches = len(self.positions) // batch_size + (1 if len(self.positions) % batch_size > 0 else 0)
        
        for skill_idx, skill in enumerate(self.skill_levels):
            print(f"Processing skill level {skill}...")
            
            for batch_idx in tqdm(range(num_batches)):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(self.positions))
                batch_positions = self.positions[start_idx:end_idx]
                
                boards = [chess.Board(pos['fen']) for pos in batch_positions]
                board_tensors = torch.stack([board_to_tensor(board) for board in boards]).to(self.device)
                
                elos_self = torch.full((len(batch_positions),), skill, device=self.device).long()
                elos_oppo = torch.full((len(batch_positions),), skill, device=self.device).long()
                
                # Forward pass through original model
                with torch.no_grad():
                    _ = self.model_original(board_tensors, elos_self, elos_oppo)
                    
                    # Process each layer
                    for layer_idx in self.transcoders.keys():
                        transcoder = self.transcoders[layer_idx]
                        
                        # Get pre-MLP activations
                        pre_mlp_acts = self.activations_original[f'layer{layer_idx}_pre_mlp']
                        
                        # Process each token position
                        for token_idx in range(pre_mlp_acts.shape[1]):
                            token_acts = pre_mlp_acts[:, token_idx]
                            
                            # Get feature activations
                            _, feature_acts = transcoder(token_acts)
                            feature_acts_np = feature_acts.cpu().numpy()
                            
                            # Store activations (mean pooling over tokens)
                            batch_indices = np.arange(start_idx, end_idx)
                            self.feature_activations_original[layer_idx]['per_position'][:, skill_idx, batch_indices] += feature_acts_np.T / pre_mlp_acts.shape[1]
                
                # Forward pass through finetuned model
                with torch.no_grad():
                    _ = self.model_finetuned(board_tensors, elos_self, elos_oppo)
                    
                    # Process each layer
                    for layer_idx in self.transcoders.keys():
                        transcoder = self.transcoders[layer_idx]
                        
                        # Get pre-MLP activations
                        pre_mlp_acts = self.activations_finetuned[f'layer{layer_idx}_pre_mlp']
                        
                        # Process each token
                        for token_idx in range(pre_mlp_acts.shape[1]):
                            token_acts = pre_mlp_acts[:, token_idx]
                            
                            # Get feature activations
                            _, feature_acts = transcoder(token_acts)
                            feature_acts_np = feature_acts.cpu().numpy()
                            
                            # Store activations (mean pooling over tokens)
                            batch_indices = np.arange(start_idx, end_idx)
                            self.feature_activations_finetuned[layer_idx]['per_position'][:, skill_idx, batch_indices] += feature_acts_np.T / pre_mlp_acts.shape[1]
        
        # Compute statistics
        for layer in self.feature_activations_original:
            # Mean over positions
            self.feature_activations_original[layer]['mean'] = np.mean(
                self.feature_activations_original[layer]['per_position'], axis=2)
            self.feature_activations_finetuned[layer]['mean'] = np.mean(
                self.feature_activations_finetuned[layer]['per_position'], axis=2)
            
            # Compute difference variance (how much feature activations change before/after finetuning)
            activation_diff = np.abs(
                self.feature_activations_finetuned[layer]['mean'] - 
                self.feature_activations_original[layer]['mean']
            )
            self.feature_activations_original[layer]['diff_variance'] = np.mean(activation_diff, axis=1)
            self.feature_activations_finetuned[layer]['diff_variance'] = np.mean(activation_diff, axis=1)

            del self.feature_activations_finetuned[layer]['per_position']
            del self.feature_activations_original[layer]['per_position']
        
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'original': self.feature_activations_original,
                'finetuned': self.feature_activations_finetuned
            }, f)
        
        print(f"Computed and saved feature activations to {cache_path}")

    def identify_skill_relevant_features(self, top_k=500):
        # Identify features that change most between original and finetuned models
        cache_path = os.path.join(self.output_dir, "skill_relevant_features.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached skill-relevant features from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.skill_relevant_features = pickle.load(f)
            return self.skill_relevant_features
        
        self.skill_relevant_features = {}
        
        for layer in self.feature_activations_original:
            diff_variance = self.feature_activations_original[layer]['diff_variance']
            top_indices = np.argsort(diff_variance)[-top_k:][::-1]
            
            self.skill_relevant_features[layer] = {
                'feature_indices': top_indices,
                'diff_variance': diff_variance[top_indices],
                'original_patterns': self.feature_activations_original[layer]['mean'][top_indices],
                'finetuned_patterns': self.feature_activations_finetuned[layer]['mean'][top_indices]
            }
            
            print(f"Layer {layer} top skill-relevant features: {top_indices}")
            print(f"Mean activation differences: {diff_variance[top_indices]}")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(self.skill_relevant_features, f)
        
        print(f"Identified and saved skill-relevant features to {cache_path}")
        return self.skill_relevant_features

    def compute_feature_connections(self):
        # Compute input-invariant connections between layer 6 and layer 7 features
        cache_path = os.path.join(self.output_dir, "feature_connections.npy")
        if os.path.exists(cache_path):
            print(f"Loading cached feature connections from {cache_path}")
            self.feature_connections = np.load(cache_path)
            return self.feature_connections
        
        if 0 not in self.transcoders or 1 not in self.transcoders:
            raise ValueError("Need transcoders for both layers 6 and 7")
        
        # Get decoder vectors from layer 0 (corresponds to layer 6 in our discussion)
        f_0_dec = self.transcoders[0].decoder.weight  # Shape: [model_dim, hidden_dim]
        
        # Get encoder vectors from layer 1 (corresponds to layer 7 in our discussion)
        f_1_enc = self.transcoders[1].encoder.weight    # Shape: [hidden_dim, model_dim]
        
        # Compute connection strengths: (f_0_dec)^T * f_1_enc^T
        self.feature_connections = torch.matmul(f_0_dec.T, f_1_enc.T).cpu().detach().numpy()
        
        # Shape: [hidden_dim_0, hidden_dim_1]: the connection strength for each pair of features
        
        np.save(cache_path, self.feature_connections)
        
        print(f"Computed and saved feature connections to {cache_path}")
        return self.feature_connections

    def build_skill_relevant_circuits(self, top_k_features=500, top_k_connections=20, batch_size=8192):
        # Build circuits using both input-invariant and input-dependent components
        cache_path = os.path.join(self.output_dir, "skill_relevant_circuits.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached skill-relevant circuits from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.skill_relevant_circuits = pickle.load(f)
            return self.skill_relevant_circuits
        
        if self.feature_connections is None:
            self.compute_feature_connections()
        
        if self.skill_relevant_features is None:
            self.identify_skill_relevant_features()
        
        # Middle skill level for analysis
        skill_level = 5  
        self.skill_relevant_circuits = []
        
        # Start with top layer features
        layer1_features = self.skill_relevant_features[1]['feature_indices'][:top_k_features]
        
        # Initialize attributions matrix for all layer 0 features to each selected layer 1 feature
        all_attributions = {feat1_idx: np.zeros(self.feature_connections.shape[0]) for feat1_idx in layer1_features}
        total_tokens = 0
        
        # Process all positions by batch
        num_batches = len(self.positions) // batch_size + (1 if len(self.positions) % batch_size > 0 else 0)
        for batch_idx in tqdm(range(num_batches), desc="Processing position batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.positions))
            batch_positions = self.positions[start_idx:end_idx]
            
            boards = [chess.Board(pos['fen']) for pos in batch_positions]
            board_tensors = torch.stack([board_to_tensor(board) for board in boards]).to(self.device)
            
            elos_self = torch.full((len(batch_positions),), skill_level, device=self.device).long()
            elos_oppo = torch.full((len(batch_positions),), skill_level, device=self.device).long()
            
            # Forward pass through finetuned model
            with torch.no_grad():
                _ = self.model_finetuned(board_tensors, elos_self, elos_oppo)
                
                # Get layer 0 activations
                pre_mlp_acts_0 = self.activations_finetuned[f'layer0_pre_mlp']
                
                # For each token in the batch
                for token_idx in range(pre_mlp_acts_0.shape[1]):
                    # Get all feature activations for this token across all batch examples
                    token_acts = pre_mlp_acts_0[:, token_idx]
                    _, all_layer0_activations = self.transcoders[0](token_acts)
                    all_layer0_activations_np = all_layer0_activations.cpu().numpy()  # [batch_size, hidden_dim_0]
                    
                    # For each position
                    for pos_idx in range(all_layer0_activations_np.shape[0]):
                        # Get all layer 0 feature activations for this position
                        layer0_acts = all_layer0_activations_np[pos_idx]  # [hidden_dim_0]
                        
                        # For each layer 1 feature, compute attributions to all layer 0 features at once
                        for feat1_idx in layer1_features:
                            # Get connection weights (already computed)
                            feat1_connections = self.feature_connections[:, feat1_idx]  # [hidden_dim_0]
                            
                            # Matrix multiplication: activations * connections element-wise
                            # This calculates attributions for all layer 0 features at once
                            attributions = np.abs(layer0_acts * feat1_connections)
                            
                            # Add to total attributions
                            all_attributions[feat1_idx] += attributions
                    
                    total_tokens += all_layer0_activations_np.shape[0]  # Count positions processed
        
        # Normalize attributions by total tokens processed
        for feat1_idx in all_attributions:
            all_attributions[feat1_idx] /= max(1, total_tokens)
        
        # Build circuits for each layer 1 feature
        for i, feat1_idx in enumerate(tqdm(layer1_features, desc="Building circuits")):
            print(f"Building circuit for layer 1 feature {feat1_idx} ({i+1}/{len(layer1_features)})")
            
            # Get attributions for this feature
            feat_attributions = all_attributions[feat1_idx]
            
            # Find top connected features based on attribution scores
            top_feat0_indices = np.argsort(feat_attributions)[-top_k_connections:][::-1]
            
            # Create circuit data
            circuit = {
                'layer1_feature': int(feat1_idx),
                'layer1_diff_variance': float(self.skill_relevant_features[1]['diff_variance'][i]),
                'layer1_original_pattern': self.skill_relevant_features[1]['original_patterns'][i].tolist(),
                'layer1_finetuned_pattern': self.skill_relevant_features[1]['finetuned_patterns'][i].tolist(),
                'connections': []
            }
            
            # Add connection data for each top layer 0 feature
            for feat0_idx in top_feat0_indices:
                # Check if this feature is in the top skill-relevant features
                is_skill_relevant = feat0_idx in self.skill_relevant_features[0]['feature_indices']
                if is_skill_relevant:
                    l0_position = np.where(self.skill_relevant_features[0]['feature_indices'] == feat0_idx)[0][0]
                    l0_diff_variance = float(self.skill_relevant_features[0]['diff_variance'][l0_position])
                    l0_original_pattern = self.skill_relevant_features[0]['original_patterns'][l0_position].tolist()
                    l0_finetuned_pattern = self.skill_relevant_features[0]['finetuned_patterns'][l0_position].tolist()
                else:
                    l0_diff_variance = float(self.feature_activations_original[0]['diff_variance'][feat0_idx])
                    l0_original_pattern = self.feature_activations_original[0]['mean'][feat0_idx].tolist()
                    l0_finetuned_pattern = self.feature_activations_finetuned[0]['mean'][feat0_idx].tolist()
                
                # Input-invariant connection strength
                input_invariant_connection = float(self.feature_connections[feat0_idx, feat1_idx])
                
                # Add connection
                circuit['connections'].append({
                    'layer0_feature': int(feat0_idx),
                    'is_skill_relevant': is_skill_relevant,
                    'layer0_diff_variance': l0_diff_variance,
                    'layer0_original_pattern': l0_original_pattern,
                    'layer0_finetuned_pattern': l0_finetuned_pattern,
                    'input_invariant_connection': input_invariant_connection,
                    'attribution': float(feat_attributions[feat0_idx]),
                })
            
            self.skill_relevant_circuits.append(circuit)
            
            # Save intermediate results
            with open(cache_path, 'wb') as f:
                pickle.dump(self.skill_relevant_circuits, f)
            print(f"Saved intermediate results for {i+1}/{len(layer1_features)} layer 1 features")
        
        print(f"Built and saved skill-relevant circuits to {cache_path}")
        return self.skill_relevant_circuits

    def _is_square_under_defensive_threat(self, fen, square_idx):
        board = chess.Board(fen)
        piece = board.piece_at(square_idx)
        if piece is None or piece.color != chess.WHITE:
            return False
        return len(board.attackers(chess.BLACK, square_idx)) > len(board.attackers(chess.WHITE, square_idx))

    def visualize_skill_relevant_features(self):
        if self.skill_relevant_features is None:
            self.identify_skill_relevant_features()
        
        for layer, features in self.skill_relevant_features.items():
            layer_dir = os.path.join(self.output_dir, f"layer_{layer}")
            os.makedirs(layer_dir, exist_ok=True)
            
            top10_indices = features['feature_indices'][:10]
            original_patterns = features['original_patterns'][:10]
            finetuned_patterns = features['finetuned_patterns'][:10]
            
            plt.figure(figsize=(15, 20))
            
            for i, (feat_idx, orig, ft) in enumerate(zip(top10_indices, original_patterns, finetuned_patterns)):
                plt.subplot(5, 2, i+1)
                
                plt.plot(self.skill_levels, orig, marker='o', color='blue', label='Original', linewidth=2)
                plt.plot(self.skill_levels, ft, marker='o', color='red', label='Finetuned', linewidth=2)
                
                plt.title(f"Layer {layer} Feature {feat_idx}", fontsize=12)
                plt.xlabel("Skill Level")
                plt.ylabel("Mean Activation")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.xticks(self.skill_levels)
            
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, "top_features_comparison.png"))
            plt.close()
            
            plt.figure(figsize=(10, 8))
            diff_matrix = np.abs(finetuned_patterns - original_patterns)
            
            cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', ['white', 'lightblue', 'blue', 'darkblue'], N=256
            )
            
            sns.heatmap(
                diff_matrix, 
                cmap=cmap,
                xticklabels=self.skill_levels,
                yticklabels=[f"Feature {idx}" for idx in top10_indices]
            )
            
            plt.title(f"Layer {layer} Feature Activation Differences (Finetuned - Original)")
            plt.xlabel("Skill Level")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, "feature_diff_heatmap.png"))
            plt.close()

    def visualize_circuits(self):
        if self.skill_relevant_circuits is None:
            self.build_skill_relevant_circuits()
        
        for i, circuit in enumerate(self.skill_relevant_circuits):
            circuit_dir = os.path.join(self.output_dir, f"circuit_{i}")
            os.makedirs(circuit_dir, exist_ok=True)
            
            plt.figure(figsize=(15, 8))
            plt.plot(self.skill_levels, circuit['layer1_original_pattern'], marker='o', color='blue', 
                    label='Original', linewidth=2)
            plt.plot(self.skill_levels, circuit['layer1_finetuned_pattern'], marker='o', color='red', 
                    label='Finetuned', linewidth=2)
            
            plt.title(f"Layer 1 Feature {circuit['layer1_feature']}", fontsize=24)
            plt.xlabel("Skill Level", fontsize=18)
            plt.ylabel("Mean Activation", fontsize=18)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=18)
            plt.xticks(self.skill_levels, fontsize=16)
            plt.yticks(fontsize=16)
            
            l1_feat_name = f"L7TC{circuit['layer1_feature']}MLP.png"
            plt.savefig(os.path.join(circuit_dir, l1_feat_name), dpi=150, bbox_inches='tight')
            plt.close()
            
            connections = sorted(circuit['connections'], key=lambda x: abs(x['attribution']), reverse=True)
            
            for j, connection in enumerate(connections):
                plt.figure(figsize=(15, 8))
                
                layer0_feature = connection['layer0_feature']
                plt.plot(self.skill_levels, connection['layer0_original_pattern'], marker='o', 
                        color='blue', label='Original', linewidth=2)
                plt.plot(self.skill_levels, connection['layer0_finetuned_pattern'], marker='o', 
                        color='red', label='Finetuned', linewidth=2)
                
                plt.title(f"Layer 0 Feature {layer0_feature}", fontsize=24)
                plt.xlabel("Skill Level", fontsize=18)
                plt.ylabel("Mean Activation", fontsize=18)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=18)
                plt.xticks(self.skill_levels, fontsize=16)
                plt.yticks(fontsize=16)
                
                l0_feat_name = f"L6TC{layer0_feature}MLP.png"
                plt.savefig(os.path.join(circuit_dir, l0_feat_name), dpi=400, bbox_inches='tight')
                plt.close()
            
            with open(os.path.join(circuit_dir, "circuit_info.txt"), "w") as f:
                f.write(f"Circuit {i}: Layer 1 Feature {circuit['layer1_feature']}\n")
                f.write(f"Difference Variance: {circuit['layer1_diff_variance']:.6f}\n\n")
                
                f.write("Top Connections by Attribution:\n")
                for j, conn in enumerate(connections):
                    f.write(f"{j+1}. Layer 0 Feature {conn['layer0_feature']}:\n")
                    f.write(f"   Full Attribution: {conn['attribution']:.6f}\n")
                    f.write(f"   Input-Invariant Connection: {conn['input_invariant_connection']:.6f}\n")
                    f.write(f"   Difference Variance: {conn['layer0_diff_variance']:.6f}\n")
                    f.write(f"   Is Skill-Relevant: {conn['is_skill_relevant']}\n\n")

    def run_complete_analysis(self):
        try:
            print("Step 1: Computing feature activations...")
            self.compute_feature_activations()
            
            print("Step 2: Identifying skill-relevant features...")
            self.identify_skill_relevant_features()
            
            print("Step 3: Computing feature connections...")
            self.compute_feature_connections()
            
            print("Step 4: Building skill-relevant circuits...")
            self.build_skill_relevant_circuits()
            
            print("Step 5: Creating visualizations...")
            self.visualize_skill_relevant_features()
            self.visualize_circuits()
            
            print(f"Analysis complete! Results saved to {self.output_dir}")
            
        finally:
            self._cleanup_hooks()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find skill-relevant circuits in Maia-2 for e5 defensive threat task")
    
    parser.add_argument('--original_model_path', type=str, default='../weights.v2.pt',
                        help='Path to original model checkpoint')
    parser.add_argument('--finetuned_model_path', type=str, 
                        default='ft_model/threats_finetune_0.0001_8192_1e-05/best_model.pt',
                        help='Path to finetuned model checkpoint')
    parser.add_argument('--transcoder_path', type=str, default='transcoder_models',
                        help='Path to directory containing transcoder models')
    parser.add_argument('--data_path', type=str, default='../blundered-transitional-dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='skill_relevant_circuits_e5_threat',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    class ConfigForTranscoderAnalysis:
        def __init__(self):
            self.input_channels = 18
            self.dim_cnn = 256
            self.dim_vit = 1024
            self.num_blocks_cnn = 5
            self.num_blocks_vit = 2
            self.vit_length = 8
            self.elo_dim = 128
    
    cfg = ConfigForTranscoderAnalysis()
    
    analyzer = SkillRelevantCircuitFinder(
        cfg=cfg,
        original_model_path=args.original_model_path,
        finetuned_model_path=args.finetuned_model_path,
        transcoder_path=args.transcoder_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()