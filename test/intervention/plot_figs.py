import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_chessboard_heatmap(features_dict: dict, title: str) -> None:
    auc_matrix = np.zeros((8, 8))

    for concept_name, feature_results in features_dict.items():
        square = concept_name[-2:]  
        best_feature = feature_results[0]
        auc_score = best_feature[1]
        
        file_idx = 'abcdefgh'.index(square[0])
        rank_idx = '12345678'.index(square[1])
        auc_matrix[7-rank_idx, file_idx] = auc_score

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(auc_matrix, 
                     annot=np.around(auc_matrix, decimals=2),
                     fmt='.2f',
                     cmap='RdBu_r',
                     square=True,
                     vmin=0,
                     vmax=1.0,
                     cbar_kws={'label': 'AUC Score'},
                     annot_kws={'size': 12}
                     )
    
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels('abcdefgh', fontsize=14)
    ax.set_yticklabels('87654321', fontsize=14)

    plt.title(title, pad=20, fontsize=16, weight='bold')

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('AUC Score', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    output_file = f'maia2-sae/figs/squarewise_threats/finetuned_{title.lower().replace(" ", "_")}.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()


def plot_vanilla_intervened_acc(avg_original_accuracies, avg_intervened_accuracies, save_path='maia2-sae/figs/vanilla_intervened_acc.pdf'):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16
    })

    plt.figure(figsize=(12, 8))
    elo_ratings = sorted(avg_original_accuracies.keys())

    original_accs = [avg_original_accuracies[elo] for elo in elo_ratings]
    intervened_accs = [avg_intervened_accuracies[elo] for elo in elo_ratings]
    plt.plot(elo_ratings, original_accs, 'b-o', label='Original', 
            linewidth=3, markersize=10)
    plt.plot(elo_ratings, intervened_accs, 'r-o', label='Intervened', 
            linewidth=3, markersize=10)
    
    plt.xlabel('ELO Rating of Transitional Point', fontsize=16, labelpad=10)
    plt.ylabel('Prediction Accuracy', fontsize=16, labelpad=10)
    plt.title('Intervention Effects on Transitional Positions', 
             fontsize=20, pad=20)

    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    plt.legend(fontsize=16, frameon=True, framealpha=1, 
              edgecolor='black', loc='best', borderpad=1)

    plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    plt.tick_params(axis='both', which='minor', width=1.5, length=4)

    # for i, (orig, best) in enumerate(zip(original_accs, intervened_accs)):
    #     plt.annotate(f'{orig:.3f}', (elo_ratings[i], orig), 
    #                 textcoords="offset points", xytext=(0,10), ha='center')
    #     plt.annotate(f'{best:.3f}', (elo_ratings[i], best), 
    #                 textcoords="offset points", xytext=(0,-15), ha='center')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    with open('maia2-sae/dataset/finetuned_intervention/layer6_squarewise_features.pickle', 'rb') as f:
        layer6_features = pickle.load(f)
    
    with open('maia2-sae/dataset/finetuned_intervention/layer7_squarewise_features.pickle', 'rb') as f:
        layer7_features = pickle.load(f)
    
    create_chessboard_heatmap(layer6_features, "Layer 6 AUC Scores")
    create_chessboard_heatmap(layer7_features, "Layer 7 AUC Scores")

    # with open('maia2-sae/dataset/intervention/vanilla_intervention_accuracies.pickle', 'rb') as f:
    #     results = pickle.load(f)
    
    # avg_intervened_accuracies = results['best_accuracies']
    # avg_original_accuracies = results['original_accuracies']

    # plot_vanilla_intervened_acc(avg_original_accuracies, avg_intervened_accuracies)

if __name__ == "__main__":
    main()