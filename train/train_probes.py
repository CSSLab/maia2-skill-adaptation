import os
import sys
import pickle
import torch
import numpy as np
import random
import argparse
import pandas as pd
from tqdm import tqdm
import chess
import threading

ROOT = os.environ.get('SKILL_ADAPTATION_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chess_concept_oracle import get_all_concept_functions, CONCEPT_CATEGORIES
from maia2.main import MAIA2Model
from maia2.utils import get_all_possible_moves, create_elo_dict, board_to_tensor

_thread_local = threading.local()

class LogisticProbe:
    def __init__(self, C=1.0, penalty="l1", max_iter=1000, device="cuda", batch_size=256):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.trained = False
        self.best_val_acc = 0
        self.best_model_state = None

    def train(self, X, y, X_val=None, y_val=None, quick_eval=False, random_seed=42):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        input_dim = X.shape[1]
        self.model = torch.nn.Linear(input_dim, 1).to(self.device)

        torch.nn.init.kaiming_normal_(self.model.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.model.bias)

        l1_lambda = 1.0 / self.C
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        max_epochs = 100 if quick_eval else 500
        patience = 15 if quick_eval else 50
        patience_counter = 0

        self.best_val_acc = 0
        self.best_model_state = None

        self.trained = True
        self.model.train()

        for epoch in range(max_epochs):
            epoch_loss = 0
            num_batches = 0

            indices = torch.randperm(len(X))
            for i in range(0, len(X), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)

                if self.penalty == "l1":
                    l1_reg = torch.sum(torch.abs(self.model.weight)) + torch.sum(torch.abs(self.model.bias))
                    loss += l1_lambda * l1_reg

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model not trained yet")
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)
                outputs = self.model(batch_X).squeeze()
                probs = torch.sigmoid(outputs)
                batch_preds = (probs > 0.5).cpu().numpy().astype(int)
                predictions.extend(batch_preds)

        return np.array(predictions)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

def _enable_activation_hook(model, num_blocks):
    def get_activation(name):
        def hook(model, input, output):
            if not hasattr(_thread_local, 'residual_streams'):
                _thread_local.residual_streams = {}
            # residual stream after the block: feed-forward contribution added to its input
            _thread_local.residual_streams[name] = (output + input[0]).detach()
        return hook

    for i in range(num_blocks):
        feedforward_module = model.transformer.elo_layers[i][1]
        feedforward_module.register_forward_hook(get_activation(f'transformer block {i} hidden states'))

def collect_concept_positions(concept_func, data_path, max_pos=5000, max_neg=5000, seed=42):
    df = pd.read_csv(data_path)

    positive_rows = []
    negative_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning positions"):
        fen = row['fen']
        try:
            result = concept_func(fen)
            if result == 1:
                positive_rows.append(row)
            else:
                negative_rows.append(row)
        except Exception as e:
            continue

    random.seed(seed)
    random.shuffle(positive_rows)
    random.shuffle(negative_rows)

    n_pos = min(len(positive_rows), max_pos)
    n_neg = min(len(negative_rows), max_neg)
    n_samples = min(n_pos, n_neg)

    if n_samples == 0:
        return None

    selected_pos = positive_rows[:n_samples]
    selected_neg = negative_rows[:n_samples]

    all_rows = selected_pos + selected_neg
    labels = [1] * n_samples + [0] * n_samples

    combined = list(zip(all_rows, labels))
    random.shuffle(combined)

    rows, labels = zip(*combined)
    concept_df = pd.DataFrame(list(rows))
    concept_df['label'] = labels

    return concept_df

def split_train_val_test(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train+n_val]
    test_df = df_shuffled[n_train+n_val:]

    return train_df, val_df, test_df

def get_activations_for_df(model, df, elo_category, all_moves_dict, layer_key, device):
    boards_list = []
    elos_self_list = []
    elos_oppo_list = []

    for _, row in df.iterrows():
        fen = row['fen']
        board = chess.Board(fen)
        board_input = board_to_tensor(board)
        boards_list.append(board_input)
        elos_self_list.append(elo_category)
        elos_oppo_list.append(elo_category)

    boards = torch.stack(boards_list).to(device)
    elos_self = torch.tensor(elos_self_list).to(device)
    elos_oppo = torch.tensor(elos_oppo_list).to(device)

    _thread_local.residual_streams = {}

    with torch.no_grad():
        logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)

    activations = _thread_local.residual_streams[layer_key]
    activations = torch.mean(activations, dim=1)

    return activations.cpu().numpy()

def train_probe_for_concept(concept_name, concept_func, concept_positions_dir, data_path, model,
                            all_moves_dict, layer_key, C_values, device, output_dir,
                            max_pos=5000, max_neg=5000):

    concept_csv_path = os.path.join(concept_positions_dir, f"{concept_name}.csv")

    if not os.path.exists(concept_csv_path):
        concept_df = collect_concept_positions(concept_func, data_path, max_pos, max_neg)
        if concept_df is None or len(concept_df) < 100:
            return None
        os.makedirs(concept_positions_dir, exist_ok=True)
        concept_df.to_csv(concept_csv_path, index=False)
        n_pos = (concept_df['label'] == 1).sum()
        n_neg = (concept_df['label'] == 0).sum()
        print(f"  Collected {len(concept_df)} positions ({n_pos} pos, {n_neg} neg)")
    else:
        concept_df = pd.read_csv(concept_csv_path)
        if len(concept_df) < 100:
            return None
        n_pos = (concept_df['label'] == 1).sum()
        n_neg = (concept_df['label'] == 0).sum()
        print(f"  Loaded {len(concept_df)} positions ({n_pos} pos, {n_neg} neg)")

    train_df, val_df, test_df = split_train_val_test(concept_df)

    if len(train_df) < 20 or len(val_df) < 10 or len(test_df) < 10:
        return None

    results_per_elo = {}
    probe_weights_per_elo = {}

    for elo_category in range(11):
        X_train = get_activations_for_df(model, train_df, elo_category, all_moves_dict, layer_key, device)
        y_train = train_df['label'].values
        X_val = get_activations_for_df(model, val_df, elo_category, all_moves_dict, layer_key, device)
        y_val = val_df['label'].values
        X_test = get_activations_for_df(model, test_df, elo_category, all_moves_dict, layer_key, device)
        y_test = test_df['label'].values

        best_C = None
        best_val_acc = 0

        for C in C_values:
            probe = LogisticProbe(C=C, device=device, batch_size=256)
            probe.train(X_train, y_train, X_val, y_val, quick_eval=True)
            val_acc = probe.evaluate(X_val, y_val)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_C = C

        final_probe = LogisticProbe(C=best_C, device=device, batch_size=256)
        final_probe.train(X_train, y_train, X_val, y_val, quick_eval=False)

        train_acc = final_probe.evaluate(X_train, y_train)
        val_acc = final_probe.evaluate(X_val, y_val)
        test_acc = final_probe.evaluate(X_test, y_test)

        probe_weights_per_elo[elo_category] = final_probe.model.state_dict()

        results_per_elo[elo_category] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'best_C': best_C,
        }

    test_accs = [results_per_elo[elo]['test_acc'] for elo in range(11)]
    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)

    print(f"\n  Concept: {concept_name}")
    print(f"  Per-ELO test accuracies:")
    for elo in range(11):
        print(f"    ELO {elo}: train={results_per_elo[elo]['train_acc']:.4f} val={results_per_elo[elo]['val_acc']:.4f} test={results_per_elo[elo]['test_acc']:.4f}")
    print(f"  Mean test acc: {mean_test_acc:.4f}, Std: {std_test_acc:.4f}\n")

    probe_save_path = os.path.join(output_dir, f"{concept_name}_probes.pkl")
    probe_data = {
        'concept_name': concept_name,
        'probe_weights_per_elo': probe_weights_per_elo,
        'results_per_elo': results_per_elo,
        'mean_test_acc': mean_test_acc,
        'std_test_acc': std_test_acc,
        'min_test_acc': min(test_accs),
        'max_test_acc': max(test_accs),
        'n_samples': len(concept_df),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
    }
    with open(probe_save_path, 'wb') as f:
        pickle.dump(probe_data, f)

    return {
        'concept_name': concept_name,
        'results_per_elo': results_per_elo,
        'mean_test_acc': mean_test_acc,
        'std_test_acc': std_test_acc,
        'min_test_acc': min(test_accs),
        'max_test_acc': max(test_accs),
        'n_samples': len(concept_df),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=os.path.join(ROOT, 'dataset', 'blundered-transitional-dataset', 'train_moves.csv'), type=str)
    parser.add_argument('--model_path', default=os.path.join(ROOT, 'weights.v2.pt'), type=str)
    parser.add_argument('--concept_positions_dir', default=os.path.join(ROOT, 'concept_positions'), type=str)
    parser.add_argument('--output_dir', default=os.path.join(ROOT, 'probes', 'layer1'), type=str)
    parser.add_argument('--layer_key', default='transformer block 1 hidden states', type=str)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--max_pos', default=5000, type=int)
    parser.add_argument('--max_neg', default=5000, type=int)
    parser.add_argument('--C_values', nargs='+', type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument('--concepts', nargs='*', default=None, help='subset of concept names; default: all')
    parser.add_argument('--model', default='ViT', type=str)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=2, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=8, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    return parser.parse_args()

def main():
    args = parse_args()

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    device = torch.device(args.device)
    ckpt = torch.load(args.model_path, map_location=device)

    model = MAIA2Model(len(all_moves), elo_dict, args)

    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    _enable_activation_hook(model, args.num_blocks_vit)

    os.makedirs(args.output_dir, exist_ok=True)

    concept_functions = get_all_concept_functions()
    print(f"Total concepts: {len(concept_functions)}")

    results = []

    for category_name, concept_names in CONCEPT_CATEGORIES.items():
        print(f"\n{'='*80}")
        print(f"Category: {category_name}")
        print(f"{'='*80}")

        category_pbar = tqdm(concept_names, desc=f"Training probes for {category_name}")

        for concept_name in category_pbar:
            if args.concepts and concept_name not in args.concepts:
                continue
            category_pbar.set_description(f"Training: {concept_name}")

            if concept_name not in concept_functions:
                print(f"Warning: {concept_name} not found in concept_functions")
                continue

            concept_func = concept_functions[concept_name]

            try:
                result = train_probe_for_concept(
                    concept_name=concept_name,
                    concept_func=concept_func,
                    concept_positions_dir=args.concept_positions_dir,
                    data_path=args.data_path,
                    model=model,
                    all_moves_dict=all_moves_dict,
                    layer_key=args.layer_key,
                    C_values=args.C_values,
                    device=args.device,
                    output_dir=args.output_dir,
                    max_pos=args.max_pos,
                    max_neg=args.max_neg
                )

                if result is not None:
                    results.append(result)
                    category_pbar.set_postfix({
                        'mean_test': f"{result['mean_test_acc']:.3f}",
                        'std_test': f"{result['std_test_acc']:.3f}",
                        'n_samples': result['n_samples']
                    })
                else:
                    category_pbar.set_postfix({'status': 'insufficient_samples'})

            except Exception as e:
                print(f"\nError training probe for {concept_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    summary_path = os.path.join(args.output_dir, 'probe_results_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Total concepts with probes: {len(results)}")
    print(f"Total probes trained: {len(results) * 11} ({len(results)} concepts × 11 ELOs)")
    print(f"Results saved to: {args.output_dir}")

    if results:
        avg_mean_test_acc = np.mean([r['mean_test_acc'] for r in results])
        avg_std_test_acc = np.mean([r['std_test_acc'] for r in results])
        print(f"\nAverage mean test accuracy across all concepts: {avg_mean_test_acc:.4f}")
        print(f"Average std test accuracy across all concepts: {avg_std_test_acc:.4f}")

        print("\nTop 10 concepts by mean test accuracy:")
        sorted_results = sorted(results, key=lambda x: x['mean_test_acc'], reverse=True)
        for i, r in enumerate(sorted_results[:10], 1):
            print(f"  {i}. {r['concept_name']}: mean={r['mean_test_acc']:.4f} std={r['std_test_acc']:.4f} (n={r['n_samples']})")

        print("\nTop 10 concepts by cross-ELO variance (highest skill adaptation):")
        sorted_by_variance = sorted(results, key=lambda x: x['std_test_acc'], reverse=True)
        for i, r in enumerate(sorted_by_variance[:10], 1):
            print(f"  {i}. {r['concept_name']}: std={r['std_test_acc']:.4f} mean={r['mean_test_acc']:.4f} range=[{r['min_test_acc']:.4f}, {r['max_test_acc']:.4f}]")

        print("\nBottom 10 concepts by cross-ELO variance (lowest skill adaptation):")
        for i, r in enumerate(sorted_by_variance[-10:], 1):
            print(f"  {i}. {r['concept_name']}: std={r['std_test_acc']:.4f} mean={r['mean_test_acc']:.4f} range=[{r['min_test_acc']:.4f}, {r['max_test_acc']:.4f}]")

if __name__ == "__main__":
    main()
