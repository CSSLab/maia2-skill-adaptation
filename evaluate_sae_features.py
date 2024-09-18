import pickle
import torch
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.metrics import precision_recall_fscore_support
from get_self_implemented_concepts import *

def evaluate_sae_feature(concept_func, feature_activations, board_fens, threshold):
    ground_truth = np.array([concept_func(fen) for fen in board_fens])
    
    # Average the activations for each board (8 tokens per board)
    feature_activations_avg = feature_activations.view(-1, 8).mean(dim=1)
    
    feature_predictions = (feature_activations_avg > threshold).int().cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, feature_predictions, average='macro', zero_division=0)
    return precision, recall, f1

def process_sae_feature(args):
    feature_index, feature_activations, concept_func, board_fens = args
    if torch.sum(feature_activations) == 0:
        return feature_index, 0, 0, 0, 0
    else:
        thresholds = torch.linspace(feature_activations.min(), 0.4*feature_activations.max(), steps=8)
        eval_func = partial(evaluate_sae_feature, concept_func, feature_activations, board_fens)
        results = list(map(eval_func, thresholds))
        
        best_threshold_index = max(range(len(results)), key=lambda i: results[i][-1])
        best_precision, best_recall, best_f1 = results[best_threshold_index]
        best_threshold = thresholds[best_threshold_index].item()
        return feature_index, best_threshold, best_precision, best_recall, best_f1

def main() -> None:
    with open('maia2_activations_for_sae.pickle', 'rb') as f:
        maia2_activations = pickle.load(f)
    target_key_list = ['transformer block 0 hidden states', 'transformer block 1 hidden states'] # 'conv_last'

    sae_activations = maia2_activations['all_sae_activations']
    board_fens = maia2_activations['board_fen']
    num_cores = min(72, cpu_count())
    results = {}

    for concept_name, concept_func in function_map.items():
        if concept_name not in custom_header:
            continue

        print(f"Evaluating concept: {concept_name}")

        for key in target_key_list:
            print(f"Processing {key}")
            sae_feature_activations = sae_activations[key][0]
            n_features = sae_feature_activations.shape[1]
        
            args_list = [(i, sae_feature_activations[:, i], concept_func, board_fens) for i in range(n_features)]
            
            with Pool(num_cores) as pool:
                concept_results = list(tqdm.tqdm(pool.imap(process_sae_feature, args_list), total=n_features))
            
            # Sort features by F1 score to calcuate the "coverage"
            concept_results.sort(key=lambda x: x[4], reverse=True)
            results[concept_name] = concept_results

            top_feature = concept_results[0]
            print(f"\nBest feature for {concept_name}:")
            print(f"Feature index: {top_feature[0]}")
            print(f"Best threshold: {top_feature[1]:.4f}")
            print(f"Precision: {top_feature[2]:.4f}")
            print(f"Recall: {top_feature[3]:.4f}")
            print(f"F1-score: {top_feature[4]:.4f}")

            N = 5
            print(f"\nTop {N} features for {concept_name}:")
            for i, (feature_index, threshold, precision, recall, f1) in enumerate(concept_results[:N], 1):
                print(f"{i}. Feature {feature_index}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    with open('sae_feature_evaluation_results.pickle', 'wb') as f:
        pickle.dump(results, f)

if __name__=="__main__":
    main()