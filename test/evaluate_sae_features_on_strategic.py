import pickle
import torch
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from .get_self_implemented_concepts import *
import random
import matplotlib.pyplot as plt

def select_indices(lst, max_samples=500):
    positive_indices = [i for i, val in enumerate(lst) if val == 1]
    negative_indices = [i for i, val in enumerate(lst) if val == 0]

    num_positive = len(positive_indices)
    num_negative = len(negative_indices)
    num_samples = min(max_samples, num_positive, num_negative)

    selected_positive = random.sample(positive_indices, num_samples)
    selected_negative = random.sample(negative_indices, num_samples)

    return selected_positive, selected_negative

def convert_indices(original_indices):
    larger_indices = []
    for index in original_indices:
        # Get 8 corresponding indices for each original index
        start_index = index * 8
        larger_indices.extend(range(start_index, start_index + 8))
    return larger_indices

def evaluate_sae_feature(feature_activations, ground_truth, threshold):
    # Average the activations for each board (8 tokens per board)
    feature_predictions = (feature_activations > threshold).int().cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, feature_predictions, average='macro', zero_division=0)
    return precision, recall, f1

# def evaluate_sae_feature(concept_func, feature_activations, board_fens, threshold):
#     ground_truth = np.array([concept_func(fen) for fen in board_fens])
    
#     # Average the activations for each board (8 tokens per board)
#     feature_activations_avg = feature_activations.view(-1, 8).mean(dim=1)
    
#     feature_predictions = (feature_activations_avg > threshold).int().cpu().numpy()
#     precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, feature_predictions, average='macro', zero_division=0)
#     return precision, recall, f1

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

# def process_sae_feature(args):
#     feature_index, feature_activations, ground_truth = args
#     if torch.sum(feature_activations) == 0:
#         return feature_index, 0, 0, 0, 0
#     else:
#         feature_activations = feature_activations.view(-1, 8).mean(dim=1)
#         thresholds = torch.linspace(feature_activations.min(), 0.4*feature_activations.max(), steps=8)
#         eval_func = partial(evaluate_sae_feature, feature_activations, ground_truth)
#         results = list(map(eval_func, thresholds))
        
#         best_threshold_index = max(range(len(results)), key=lambda i: results[i][-1])
#         best_precision, best_recall, best_f1 = results[best_threshold_index]
#         best_threshold = thresholds[best_threshold_index].item()
#         return feature_index, best_threshold, best_precision, best_recall, best_f1


def cache_examples_and_activations(concept_functions, concept_names, sae_activations, board_fens):

    samples_and_activations = {}
    for i in range(len(concept_functions)):
        concept_func = concept_functions[i]
        concept_name = concept_names[i]

        if concept_name not in custom_header:
            continue

        # print(concept_name)

        labels = [concept_func(fen) for fen in board_fens]
        positive_indices, negative_indices = select_indices(labels)
        temp_dic = {"positives": torch.tensor([1 for index in positive_indices]), 'negatives': torch.tensor([0 for index in negative_indices])}

        for key in sae_activations:
            temp_dic[key] = torch.cat((sae_activations[key][positive_indices], sae_activations[key][negative_indices]))

        samples_and_activations[concept_name] = temp_dic

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
    
    plt.savefig(f'figs/sae_strategic_feature/{concept_name}-{layer}.pdf')
    plt.close()

def main() -> None:
    test_dim = 2048
    sae_lr = 1e-5
    sae_site = "res"
    with open(f'activations/maia2_activations_for_sae_{test_dim}_{sae_lr}_{sae_site}.pickle', 'rb') as f:
        maia2_activations = pickle.load(f)
    target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states'] # 'transformer block 0 hidden states', 'conv_last'

    sae_activations = maia2_activations['all_sae_activations']
    board_fens = maia2_activations['board_fen']
    num_cores = min(72, cpu_count())
    all_layers_results = {}

    samples_and_activations = cache_examples_and_activations(list(function_map.values()), list(function_map.keys()), sae_activations, board_fens)

    for concept_name, concept_func in function_map.items():
        if concept_name not in custom_header:
            continue

        print(f"Evaluating concept: {concept_name}")
        results = {}
        for key in target_key_list:
            print(f"Processing {key}")
            sae_feature_activations = sae_activations[key][0]
            n_features = sae_feature_activations.shape[1]

            sampled_sae_activations = samples_and_activations[concept_name][key]
            print(samples_and_activations[concept_name]["positives"].shape, samples_and_activations[concept_name]["negatives"].shape)
            ground_truth = torch.cat((samples_and_activations[concept_name]["positives"], samples_and_activations[concept_name]["negatives"]))
            args_list = [(i, sampled_sae_activations[:, i], ground_truth) for i in range(n_features)]

            #args_list = [(i, sae_feature_activations[:, i], concept_func, board_fens) for i in range(n_features)]
            
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
            # for i, (feature_index, threshold, precision, recall, f1) in enumerate(concept_results[:N], 1):
            #     print(f"{i}. Feature {feature_index}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

        all_layers_results[key] = results

    # with open(f'res/sae_feature_eval/sae_feature_evaluation_results_{test_dim}.pickle', 'wb') as f:
    #     pickle.dump(all_layers_results, f)

if __name__=="__main__":
    main()