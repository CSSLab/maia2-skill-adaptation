import pickle
import torch
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.metrics import precision_recall_fscore_support,  roc_auc_score
from .get_self_implemented_concepts import *
import random
import os

def select_indices(lst, max_positive=250, max_negative=250):
    # Get the indices of positive and negative values
    positive_indices = [i for i, val in enumerate(lst) if val == 1]
    negative_indices = [i for i, val in enumerate(lst) if val == 0]

    # Randomly select up to max_positive from positive_indices
    selected_positive = random.sample(positive_indices, min(max_positive, len(positive_indices)))

    # Randomly select up to max_negative from negative_indices
    selected_negative = random.sample(negative_indices, min(max_negative, len(negative_indices)))

    return selected_positive, selected_negative

def convert_indices(original_indices):
    larger_indices = []
    for index in original_indices:
        # Get 8 corresponding indices for each original index
        start_index = index * 8
        larger_indices.extend(range(start_index, start_index + 8))
    return larger_indices

def evaluate_sae_feature(ground_truth, feature_activations, threshold):

    # feature_activations_avg = feature_activations.view(-1, 8).mean(dim=1)

    feature_predictions = (feature_activations > threshold).int().cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, feature_predictions, average='macro',
                                                               zero_division=0)
    return precision, recall, f1

def process_sae_feature(args):
    feature_index, feature_activations, ground_truth = args
    if torch.sum(feature_activations) == 0:
        return feature_index, 0

    feature_activations_np = feature_activations.numpy()
    ground_truth_np = ground_truth.numpy()

    if np.all(ground_truth_np == ground_truth_np[0]):
        auc = 0.5

    else:
        # feature_activations = feature_activations.view(-1, 8).mean(dim=1)
        feature_activations_np = feature_activations.numpy()
        ground_truth_np = ground_truth.numpy()
        auc = roc_auc_score(ground_truth_np, feature_activations_np)

    return feature_index, auc

def cache_examples_and_activations_board_states(pieces, sae_activations, board_fens):
    samples_and_activations = []

    for j in range(len(pieces)):
        # print(pieces[j])
        temp = []
        for square in range(64):
            labels = [is_piece_on_square(pieces[j], square, fen) for fen in board_fens]
            positive_indices, negative_indices = select_indices(labels)

            temp_dic = {"positives": torch.tensor([1 for index in positive_indices]), 'negatives': torch.tensor([0 for index in negative_indices])}

            for key in sae_activations:
                temp_dic[key] = torch.cat((sae_activations[key][positive_indices], sae_activations[key][negative_indices]))

            temp.append(temp_dic)
        samples_and_activations.append(temp)
    return samples_and_activations

def plot_feature_histogram(feature_activations, ground_truth, concept_name, layer, auc):
    layer_cnt = 6 if layer == 'transformer block 0 hidden states' else 7
    feature_activations_np = feature_activations.numpy()
    activations_0 = feature_activations_np[ground_truth == 0]
    activations_1 = feature_activations_np[ground_truth == 1]

    min_activation = np.min(feature_activations_np)
    max_activation = np.max(feature_activations_np)
    bins = np.linspace(min_activation, max_activation, 51) 

    plt.figure(figsize=(10, 6))
    plt.hist(activations_0, bins=bins, alpha=0.5, label='Negative', color='blue', density=True)
    plt.hist(activations_1, bins=bins, alpha=0.5, label='Positive', color='red', density=True)
    
    plt.xlabel('SAE Activation', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.title(f'{concept_name} CC-AUC={auc:.2f}', fontsize=18)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    
    save_dir = f'maia2-sae/figs/sae_board_states_feature'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{concept_name}-{layer}.pdf')
    plt.close()

def main() -> None:
    test_dim = 16384
    sae_lr = 5e-05
    sae_site = "res"
    with open(f'maia2-sae/activations/maia2_activations_for_sae_{test_dim}_{sae_lr}_{sae_site}.pickle', 'rb') as f:
        maia2_activations = pickle.load(f)
    target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states']  # 'conv_last'

    sae_activations = maia2_activations['all_sae_activations']
    board_fens = maia2_activations['board_fen']
    num_cores = min(72, cpu_count())
    results = {}

    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

    samples_and_activations = cache_examples_and_activations_board_states(pieces, sae_activations, board_fens)

    for j in range(len(pieces)):
        for square in range(64):

            piece = pieces[j]
            concept_name = piece + "_is_on_" + square_name(square)

            # concept_func = partial(is_piece_on_square, piece, square)

            print(f"Evaluating concept: {concept_name}")

            for key in target_key_list:
                print(f"Processing {key}")
                sae_feature_activations = sae_activations[key]
                n_features = sae_feature_activations.shape[1]

                ground_truth = torch.cat((samples_and_activations[j][square]['positives'], samples_and_activations[j][square]['negatives']))
                sampled_sae_activations = samples_and_activations[j][square][key]

                args_list = [(i, sampled_sae_activations[:, i], ground_truth) for i in range(n_features)]

                with Pool(num_cores) as pool:
                    concept_results = list(tqdm.tqdm(pool.imap(process_sae_feature, args_list), total=n_features))

                # Sort features by F1 score to calcuate the "coverage"
                concept_results.sort(key=lambda x: x[-1], reverse=True)
                results[concept_name] = concept_results

                top_feature = concept_results[0]
                print(f"\nBest feature for {concept_name}:")
                print(f"Feature index: {top_feature[0]}")
                # print(f"Best threshold: {top_feature[1]:.4f}")
                # print(f"Precision: {top_feature[2]:.4f}")
                # print(f"Recall: {top_feature[3]:.4f}")
                print(f"AUC: {top_feature[1]:.4f}")

                plot_feature_histogram(sampled_sae_activations[:, top_feature[0]], ground_truth, concept_name, key, top_feature[1])

                N = 5
                print(f"\nTop {N} features for {concept_name}:")
                for i, (feature_index, auc) in enumerate(concept_results[:N], 1):
                    print(f"{i}. Feature {feature_index}: AUC={auc:.4f}")

    # with open('sae_feature_evaluation_results.pickle', 'wb') as f:
    #     pickle.dump(results, f)


if __name__ == "__main__":
    main()