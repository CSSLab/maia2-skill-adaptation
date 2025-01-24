import pickle
import torch
import torch.nn as nn
import chess
import numpy as np
from scipy import stats
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import threading
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from ..maia2.main import *
from ..maia2.utils import *

_thread_local = threading.local()

def mirror_square(square):
    file = square[0]
    rank = str(9 - int(square[1]))
    return file + rank

def mirror_move(move_uci):
    # Check if the move is a promotion
    is_promotion = len(move_uci) > 4
    
    # Extract squares and promotion piece
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion_piece = move_uci[4:] if is_promotion else ""
    
    # Mirror squares
    mirrored_start = mirror_square(start_square)
    mirrored_end = mirror_square(end_square)
    
    return mirrored_start + mirrored_end + promotion_piece

def is_piece_no_longer_under_attack(board: chess.Board, move: str, square_index: int) -> bool:
    """Defensive willingness function."""
    piece_before_move = board.piece_at(square_index)

    def is_under_attack(board: chess.Board, square_index: int) -> bool:
        attackers = board.attackers(chess.BLACK, square_index) 
        defenders = board.attackers(chess.WHITE, square_index)  
        return len(attackers) > len(defenders)

    was_under_attack_before = (is_under_attack(board, square_index) and 
                             piece_before_move is not None and 
                             piece_before_move.color is not chess.BLACK)
    
    move_obj = chess.Move.from_uci(move)
    if not board.is_legal(move_obj):
        return False

    board.push(move_obj)
    piece_after_move = board.piece_at(square_index)
    is_under_attack_after = is_under_attack(board, square_index) and piece_after_move is not None

    return was_under_attack_before and not is_under_attack_after

def is_attacking_opportunity_taken(board: chess.Board, move: str, square_index: int) -> bool:
    """Offensive willingness function."""
    piece_before_move = board.piece_at(square_index)
    
    def has_attacking_advantage(board: chess.Board, square_index: int) -> bool:
        attackers = board.attackers(chess.WHITE, square_index)
        defenders = board.attackers(chess.BLACK, square_index)
        return len(attackers) > len(defenders)
    
    had_advantage_before = (has_attacking_advantage(board, square_index) and 
                          piece_before_move is not None and 
                          piece_before_move.color == chess.BLACK)
    
    move_obj = chess.Move.from_uci(move)
    if not board.is_legal(move_obj):
        return False
    
    is_capture_of_target = (move_obj.to_square == square_index)
    
    return had_advantage_before and is_capture_of_target

def apply_sae_to_activations(sae, activations, target_key_list):
    sae_activations = {}
    for key in target_key_list:
        act = torch.mean(activations[key], dim=1)
        pre_activation = nn.functional.linear(act, sae[key]['encoder_DF.weight'], sae[key]['encoder_DF.bias'])

        if 'threshold' in sae[key]:
            thresholds = sae[key]['threshold']
            encoded = pre_activation * (pre_activation >= thresholds.unsqueeze(0))
        else:
            encoded = nn.functional.relu(pre_activation)
            
        sae_activations[key] = encoded
    
    return sae_activations

def _enable_activation_hook(model, num_blocks_vit):
    def get_activation(name):
        def hook(model, input, output):
            if not hasattr(_thread_local, 'residual_streams'):
                _thread_local.residual_streams = {}
            _thread_local.residual_streams[name] = output.detach()
        return hook
    
    for i in range(num_blocks_vit):
        feedforward_module = model.module.transformer.elo_layers[i][1]
        feedforward_module.register_forward_hook(get_activation(f'transformer block {i} hidden states'))

def create_chessboard_heatmap(correlations_dict: dict, title: str) -> None:
    correlation_matrix = np.zeros((8, 8))

    for square_name, correlation in correlations_dict.items():
        file_idx = 'abcdefgh'.index(square_name[0])
        rank_idx = '12345678'.index(square_name[1])
        correlation_matrix[7-rank_idx, file_idx] = correlation

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(correlation_matrix, 
                     annot=np.around(correlation_matrix, decimals=2),
                     fmt='.2f',
                     cmap='RdBu_r',
                     square=True,
                     vmin=-1.0,
                     vmax=1.0,
                     cbar_kws={'label': 'Correlation'},
                     annot_kws={'size': 12}
                     )
    
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels('abcdefgh', fontsize=14)
    ax.set_yticklabels('87654321', fontsize=14)

    plt.title(title, pad=20, fontsize=16, weight='bold')

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Correlation', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    output_file = f'maia2-sae/figs/squarewise_threats/willingness_correlation_{title.lower().replace(" ", "_")}.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

def analyze_willingness_correlations(
    positions_path: str,
    layer6_features_path: str,
    layer7_features_path: str,
    model_path: str,
    sae_path: str,
    device: str = 'cpu'
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze correlations between willingness features and best move probabilities."""
    
    # Load positions
    with open(positions_path, 'rb') as f:
        positions = list(pickle.load(f))
    print(f"Loaded {len(positions)} positions")
    
    # Load best features for both layers
    with open(layer6_features_path, 'rb') as f:
        layer6_features = pickle.load(f)
    with open(layer7_features_path, 'rb') as f:
        layer7_features = pickle.load(f)
    
    # Load and prepare model
    ckpt = torch.load(model_path, map_location=torch.device(device))
    args = type('Args', (), {
        "num_blocks_vit": 2,
        "model": "ViT",
        "dim_vit": 1024,
        "dim_cnn": 256,
        "input_channels": 18,
        "vit_length": 8,
        "elo_dim": 128,
        "num_blocks_cnn": 5,
        "batch_size": 65536,
        "num_workers": 16
    })()
    
    all_moves = get_all_possible_moves()
    elo_dict = create_elo_dict()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    model = MAIA2Model(len(all_moves), elo_dict, args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Load SAE
    sae = torch.load(sae_path, map_location=device)['sae_state_dicts']
    
    # Enable activation hooks
    _enable_activation_hook(model, args.num_blocks_vit)
    
    # Initialize results structure
    results = {
        'layer6': {'defensive': {}, 'offensive': {}},
        'layer7': {'defensive': {}, 'offensive': {}}
    }
    
    # Process each square for defensive and offensive willingness
    for square_name in tqdm.tqdm(chess.SQUARE_NAMES, desc="Processing squares"):
        square_idx = chess.parse_square(square_name)
        
        # Get relevant feature indices
        def_key = f"is_piece_no_longer_under_attack_{square_name}"
        off_key = f"is_attacking_opportunity_taken_{square_name}"
        
        layer6_def_feat = layer6_features[def_key][0][0] if def_key in layer6_features else None
        layer7_def_feat = layer7_features[def_key][0][0] if def_key in layer7_features else None
        layer6_off_feat = layer6_features[off_key][0][0] if off_key in layer6_features else None
        layer7_off_feat = layer7_features[off_key][0][0] if off_key in layer7_features else None
        
        # Get filtered positions for this square
        def_positions = []
        off_positions = []
        original_indices = {'defensive': [], 'offensive': []}
        
        for pos_idx, position in enumerate(positions):
            board = chess.Board(position['fen'])
            orig_turn = board.turn
            
            if not orig_turn:  # if black to move
                board = board.mirror()
                best_move = mirror_move(position['best_move'])
                check_square = chess.square_mirror(square_idx)
            else:
                best_move = position['best_move']
                check_square = square_idx
                
            # Store original FEN before any moves
            current_fen = board.fen()
            
            # Check move legality
            move_obj = chess.Move.from_uci(best_move)
            if not board.is_legal(move_obj):
                continue
                    
            # Check defensive willingness using a copy of the board
            board_copy = chess.Board(current_fen)
            if is_piece_no_longer_under_attack(board_copy, best_move, check_square):
                def_positions.append((current_fen, best_move, 3, 3, 1))  # Use original FEN
                original_indices['defensive'].append(pos_idx)
                    
            # Check offensive willingness using a copy of the board
            board_copy = chess.Board(current_fen)
            if is_attacking_opportunity_taken(board_copy, best_move, check_square):
                off_positions.append((current_fen, best_move, 3, 3, 1))  # Use original FEN
                original_indices['offensive'].append(pos_idx)

        # Process defensive positions
        if def_positions:
            dataset = MAIA2Dataset(def_positions, all_moves_dict, args)
            dataloader = torch.utils.data.DataLoader(dataset, 
                              batch_size=args.batch_size, 
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers)
            
            def_activations = {
                'layer6': defaultdict(list),
                'layer7': defaultdict(list)
            }
            def_probs = defaultdict(list)
            
            position_data = {idx: {'probs': [], 'acts_6': [], 'acts_7': []} for idx in original_indices['defensive']}

            for elo in range(len(elo_dict)):
                for boards, _, _, _, _, _, _, _ in dataloader:
                    elos_self = torch.full((len(boards),), elo, device=device)
                    elos_oppo = torch.full((len(boards),), elo, device=device)
                    
                    with torch.no_grad():
                        _ = model(boards.to(device), elos_self, elos_oppo)
                    
                    activations = getattr(_thread_local, 'residual_streams', {})
                    sae_activations = apply_sae_to_activations(sae, activations, 
                                                            ['transformer block 0 hidden states', 
                                                            'transformer block 1 hidden states'])
                    
                    idx_count = 0
                    for orig_idx in original_indices['defensive']:
                        # Store probability for this position at this ELO
                        position_data[orig_idx]['probs'].append(positions[orig_idx]['elo_data'][elo]['best_move_prob'])
                        
                        if layer6_def_feat is not None:
                            layer6_acts = sae_activations['transformer block 0 hidden states'][idx_count, layer6_def_feat].cpu().item()
                            position_data[orig_idx]['acts_6'].append(layer6_acts)
                        
                        if layer7_def_feat is not None:
                            layer7_acts = sae_activations['transformer block 1 hidden states'][idx_count, layer7_def_feat].cpu().item()
                            position_data[orig_idx]['acts_7'].append(layer7_acts)
                        
                        idx_count += 1

            # Calculate correlations for each position (across all ELOs)
            correlations = {'layer6': [], 'layer7': []}

            for orig_idx in original_indices['defensive']:
                if layer6_def_feat is not None:
                    correlation, _ = stats.pearsonr(position_data[orig_idx]['probs'], 
                                                position_data[orig_idx]['acts_6'])
                    if not np.isnan(correlation):
                        correlations['layer6'].append(correlation)
                
                if layer7_def_feat is not None:
                    correlation, _ = stats.pearsonr(position_data[orig_idx]['probs'], 
                                                position_data[orig_idx]['acts_7'])
                    if not np.isnan(correlation):
                        correlations['layer7'].append(correlation)

            # Store average correlations across positions
            if layer6_def_feat is not None:
                avg_correlation = np.mean(correlations['layer6']) if correlations['layer6'] else 0
                print(f"layer 6 correlation for {square_name}: {avg_correlation}")
                results['layer6']['defensive'][square_name] = avg_correlation

            if layer7_def_feat is not None:
                avg_correlation = np.mean(correlations['layer7']) if correlations['layer7'] else 0
                print(f"layer 7 correlation for {square_name}: {avg_correlation}")
                results['layer7']['defensive'][square_name] = avg_correlation
        
        # Process offensive positions
        if off_positions:
            dataset = MAIA2Dataset(off_positions, all_moves_dict, args)
            dataloader = torch.utils.data.DataLoader(dataset, 
                              batch_size=args.batch_size, 
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers)
            
            off_activations = {
                'layer6': defaultdict(list),
                'layer7': defaultdict(list)
            }
            off_probs = defaultdict(list)
            
            position_data = {idx: {'probs': [], 'acts_6': [], 'acts_7': []} for idx in original_indices['offensive']}

            for elo in range(len(elo_dict)):
                for boards, _, _, _, _, _, _, _ in dataloader:
                    elos_self = torch.full((len(boards),), elo, device=device)
                    elos_oppo = torch.full((len(boards),), elo, device=device)
                    
                    with torch.no_grad():
                        _ = model(boards.to(device), elos_self, elos_oppo)
                    
                    activations = getattr(_thread_local, 'residual_streams', {})
                    sae_activations = apply_sae_to_activations(sae, activations, 
                                                            ['transformer block 0 hidden states', 
                                                            'transformer block 1 hidden states'])
                    
                    idx_count = 0
                    for orig_idx in original_indices['offensive']:
                        # Store probability for this position at this ELO
                        position_data[orig_idx]['probs'].append(positions[orig_idx]['elo_data'][elo]['best_move_prob'])
                        
                        if layer6_off_feat is not None:
                            layer6_acts = sae_activations['transformer block 0 hidden states'][idx_count, layer6_off_feat].cpu().item()
                            position_data[orig_idx]['acts_6'].append(layer6_acts)
                        
                        if layer7_off_feat is not None:
                            layer7_acts = sae_activations['transformer block 1 hidden states'][idx_count, layer7_off_feat].cpu().item()
                            position_data[orig_idx]['acts_7'].append(layer7_acts)
                        
                        idx_count += 1

            # Calculate correlations for each position (across all ELOs)
            correlations = {'layer6': [], 'layer7': []}

            for orig_idx in original_indices['offensive']:
                if layer6_off_feat is not None:
                    correlation, _ = stats.pearsonr(position_data[orig_idx]['probs'], 
                                                position_data[orig_idx]['acts_6'])
                    if not np.isnan(correlation):
                        correlations['layer6'].append(correlation)
                
                if layer7_off_feat is not None:
                    correlation, _ = stats.pearsonr(position_data[orig_idx]['probs'], 
                                                position_data[orig_idx]['acts_7'])
                    if not np.isnan(correlation):
                        correlations['layer7'].append(correlation)

            # Store average correlations across positions
            if layer6_off_feat is not None:
                avg_correlation = np.mean(correlations['layer6']) if correlations['layer6'] else 0
                print(f"layer 6 correlation for {square_name}: {avg_correlation}")
                results['layer6']['offensive'][square_name] = avg_correlation

            if layer7_off_feat is not None:
                avg_correlation = np.mean(correlations['layer7']) if correlations['layer7'] else 0
                print(f"layer 7 correlation for {square_name}: {avg_correlation}")
                results['layer7']['offensive'][square_name] = avg_correlation
                
        # Print progress for this square
        print(f"\nProcessed square {square_name}:")
        print(f"Defensive positions: {len(def_positions)}")
        print(f"Offensive positions: {len(off_positions)}")
    
    return results

# Example usage
if __name__ == "__main__":
    import os
    
    # Make sure directories exist
    os.makedirs('maia2-sae/figs/squarewise_threats_willingness', exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run analysis
    results = analyze_willingness_correlations(
        positions_path='maia2-sae/transmono_positions_for_correlation.pkl',
        layer6_features_path='maia2-sae/dataset/intervention/layer6_defensive_willingness.pickle',
        layer7_features_path='maia2-sae/dataset/intervention/layer7_defensive_willingness.pickle',
        model_path='maia2-sae/weights.v2.pt',
        sae_path='maia2-sae/sae/best_jrsaes_2023-11-16384-1-res.pt',
        device=device
    )
    
    # Create heatmaps for each layer and willingness type
    for layer in ['layer6', 'layer7']:
        for willingness_type in ['defensive', 'offensive']:
            create_chessboard_heatmap(
                results[layer][willingness_type],
                f"{willingness_type.capitalize()} Willingness Correlations ({layer})"
            )
    
    # Print summary statistics
    print("\nCorrelation Summary Statistics:")
    for layer in ['layer6', 'layer7']:
        for willingness_type in ['defensive', 'offensive']:
            correlations = np.array(list(results[layer][willingness_type].values()))
            correlations = correlations[~np.isnan(correlations)]  # Remove NaN values
            
            print(f"\n{willingness_type.capitalize()} {layer}:")
            print(f"Mean correlation: {np.mean(correlations):.3f}")
            print(f"Std correlation: {np.std(correlations):.3f}")
            print(f"Min correlation: {np.min(correlations):.3f}")
            print(f"Max correlation: {np.max(correlations):.3f}")
    
    # Save results
    output_path = 'maia2-sae/defensive_willingness_square_correlations_by_layer.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")