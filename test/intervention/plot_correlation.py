import pickle
import numpy as np
import chess
import matplotlib.pyplot as plt
import seaborn as sns

def create_chessboard_heatmap(correlations_dict: dict, title: str, ax, cbar_ax=None, vmin=-1.0, vmax=1.0) -> None:
    correlation_matrix = np.zeros((8, 8))

    for square_name, correlation in correlations_dict.items():
        file_idx = 'abcdefgh'.index(square_name[0])
        rank_idx = '12345678'.index(square_name[1])
        correlation_matrix[7 - rank_idx, file_idx] = correlation

    sns.heatmap(correlation_matrix, 
                annot=np.around(correlation_matrix, decimals=2),
                fmt='.2f',
                cmap='RdBu_r',
                square=True,
                vmin=vmin,
                vmax=vmax,
                cbar=cbar_ax is not None,
                cbar_ax=cbar_ax,
                cbar_kws={'label': 'Correlation'},
                annot_kws={'size': 16},
                ax=ax)
    
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels('abcdefgh', fontsize=18)
    ax.set_yticklabels('87654321', fontsize=18)

    ax.set_title(title, pad=20, fontsize=20, weight='bold')

    if cbar_ax is not None:
        cbar_ax.tick_params(labelsize=16)
        cbar_ax.set_ylabel('Correlation', fontsize=18)

def plot_correlation_heatmaps():
    awareness_path = 'maia2-sae/res/correlation/offensive_awareness_square_correlations_by_layer.pkl'
    willingness_path = 'maia2-sae/res/correlation/offensive_willingness_square_correlations_by_layer.pkl'
    random_awareness_path = 'maia2-sae/res/correlation/offensive_awareness_square_correlations_by_layer_random.pkl'
    random_willingness_path = 'maia2-sae/res/correlation/offensive_willingness_square_correlations_by_layer_random.pkl'
    
    with open(awareness_path, 'rb') as f:
        awareness_results = pickle.load(f)
    with open(willingness_path, 'rb') as f:
        willingness_results = pickle.load(f)
    with open(random_awareness_path, 'rb') as f:
        random_awareness_results = pickle.load(f)
    with open(random_willingness_path, 'rb') as f:
        random_willingness_results = pickle.load(f)

    best_awareness = {}
    best_willingness = {}
    random_awareness = {}
    random_willingness = {}

    for square_name in chess.SQUARE_NAMES:
        best_awareness[square_name] = max(
            awareness_results['layer6']['offensive'].get(square_name, -1),
            awareness_results['layer7']['offensive'].get(square_name, -1)
        )
        best_willingness[square_name] = max(
            willingness_results['layer6']['offensive'].get(square_name, -1),
            willingness_results['layer7']['offensive'].get(square_name, -1)
        )
        random_awareness[square_name] = max(
            random_awareness_results['layer6']['offensive'].get(square_name, -1),
            random_awareness_results['layer7']['offensive'].get(square_name, -1)
        )
        random_willingness[square_name] = max(
            random_willingness_results['layer6']['offensive'].get(square_name, -1),
            random_willingness_results['layer7']['offensive'].get(square_name, -1)
        )
    
    all_values = list(best_awareness.values()) + list(best_willingness.values()) + \
                 list(random_awareness.values()) + list(random_willingness.values())
    global_min = min(all_values)
    global_max = max(all_values)

    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Squarewise Threat Correlation Heatmaps (Best vs Random)', fontsize=24, weight='bold', y=1.02)
    
    cbar_ax = fig.add_axes([0.92, 0.10, 0.02, 0.8])  

    create_chessboard_heatmap(best_awareness, 'Best Offensive Awareness', axes[0, 0], cbar_ax=cbar_ax, vmin=global_min, vmax=global_max)
    create_chessboard_heatmap(best_willingness, 'Best Offensive Willingness', axes[0, 1], cbar_ax=cbar_ax, vmin=global_min, vmax=global_max)

    create_chessboard_heatmap(random_awareness, 'Random Offensive Awareness', axes[1, 0], cbar_ax=cbar_ax, vmin=global_min, vmax=global_max)
    create_chessboard_heatmap(random_willingness, 'Random Offensive Willingness', axes[1, 1], cbar_ax=cbar_ax, vmin=global_min, vmax=global_max)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    output_file = 'maia2-sae/figs/correlation/offensive_correlation_heatmaps_best_vs_random.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_correlation_heatmaps()