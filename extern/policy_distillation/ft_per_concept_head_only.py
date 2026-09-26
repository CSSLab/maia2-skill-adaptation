import os
import sys
import glob
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import pickle
ROOT = os.environ.get('SKILL_ADAPTATION_ROOT',
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from maia2 import utils
from maia2.main import MAIA2Model

class ConceptDataset(Dataset):
    def __init__(self, csv_path, all_moves_dict):
        self.all_moves_dict = all_moves_dict
        self.elo_dict = utils.create_elo_dict()

        df = pd.read_csv(csv_path)
        self.data = []

        for _, row in df.iterrows():
            fen = row['fen']
            correct_move = row['correct_move']
            transition_point = row.get('transition_point', -1)

            for elo_level in range(len(self.elo_dict)):
                self.data.append((fen, correct_move, elo_level, elo_level, transition_point))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move, elo_self, elo_oppo, transition_point = self.data[idx]
        board = chess.Board(fen)
        board_input = utils.board_to_tensor(board)
        move_input = self.all_moves_dict[move]
        legal_moves, side_info = utils.get_side_info(board, move, self.all_moves_dict)
        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info, transition_point

class PolicyHeadModel(nn.Module):
    def __init__(self, base_model, fc_new):
        super(PolicyHeadModel, self).__init__()
        self.base_model = base_model
        self.fc_new = fc_new

    def forward(self, boards, elos_self, elos_oppo):
        with torch.no_grad():
            batch_size = boards.size(0)
            boards = boards.view(batch_size, self.base_model.cfg.input_channels, 8, 8)
            embs = self.base_model.chess_cnn(boards)
            embs = embs.view(batch_size, embs.size(1), 8 * 8)
            x = self.base_model.to_patch_embedding(embs)
            x += self.base_model.pos_embedding
            x = self.base_model.dropout(x)
            elos_emb_self = self.base_model.elo_embedding(elos_self)
            elos_emb_oppo = self.base_model.elo_embedding(elos_oppo)
            elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
            x = self.base_model.transformer(x, elos_emb).mean(dim=1)
            x = self.base_model.last_ln(x)

        logits_maia = self.fc_new(x)
        return logits_maia

def calculate_accuracy(logits, labels, legal_moves):
    logits = logits * legal_moves
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float().sum()
    total = labels.size(0)
    return (correct / total).item(), predictions

def find_transition_point(predictions, labels, elo_levels):
    skill_results = {}
    for i, elo in enumerate(elo_levels):
        skill_results[elo] = (predictions[i] == labels[i]).item()

    if all(skill_results[elo] for elo in elo_levels):
        return 0
    if all(not skill_results[elo] for elo in elo_levels):
        return 10

    for level_idx in range(len(elo_levels) - 1):
        curr_level = elo_levels[level_idx]
        next_level = elo_levels[level_idx + 1]
        if not skill_results[curr_level] and skill_results[next_level]:
            if all(skill_results[elo_levels[i]] for i in range(level_idx + 1, len(elo_levels))):
                return next_level
    return -1

def evaluate_model(model, test_loader, elo_dict, device, all_moves=None, return_predictions=False):
    model.eval()
    elo_accuracies = {i: {'correct': 0, 'total': 0} for i in range(len(elo_dict))}
    new_transition_points = []
    all_predictions_per_elo = {elo: [] for elo in range(11)}

    with torch.no_grad():
        for batch in test_loader:
            boards, labels, elos_self, elos_oppo, legal_moves, side_info, tps = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]

            logits_maia = model(boards, elos_self, elos_oppo)
            _, predictions = calculate_accuracy(logits_maia, labels, legal_moves)

            for i in range(0, len(predictions), len(elo_dict)):
                if i + len(elo_dict) <= len(predictions):
                    batch_predictions = predictions[i:i+len(elo_dict)]
                    batch_labels = labels[i:i+len(elo_dict)]
                    batch_elos = elos_self[i:i+len(elo_dict)]

                    if return_predictions and all_moves is not None:
                        for j in range(len(elo_dict)):
                            move_idx = batch_predictions[j].item()
                            all_predictions_per_elo[j].append(all_moves[move_idx])

                    tp = find_transition_point(batch_predictions, batch_labels, batch_elos.tolist())
                    if tp != -1:
                        new_transition_points.append(tp)

                    for j in range(len(elo_dict)):
                        elo_level = j
                        is_correct = (batch_predictions[j] == batch_labels[j]).item()
                        elo_accuracies[elo_level]['total'] += 1
                        if is_correct:
                            elo_accuracies[elo_level]['correct'] += 1

    elo_accs = {}
    for elo_level, stats in elo_accuracies.items():
        if stats['total'] > 0:
            elo_accs[elo_level] = stats['correct'] / stats['total']
        else:
            elo_accs[elo_level] = 0.0

    all_tps = [tp for tp in new_transition_points]
    avg_tp_with_0_10 = sum(all_tps) / len(all_tps) if all_tps else -1

    valid_tps = [tp for tp in new_transition_points if tp not in [0, 10]]
    avg_tp_exclude_0_10 = sum(valid_tps) / len(valid_tps) if valid_tps else -1

    if return_predictions:
        return elo_accs, avg_tp_with_0_10, avg_tp_exclude_0_10, all_predictions_per_elo
    return elo_accs, avg_tp_with_0_10, avg_tp_exclude_0_10

def train_concept(concept_name, concept_dir, cfg, device):
    all_moves = utils.get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = utils.create_elo_dict()

    train_csv = os.path.join(concept_dir, 'train_moves.csv')
    val_csv = os.path.join(concept_dir, 'val_moves.csv')
    test_csv = os.path.join(concept_dir, 'test_moves.csv')

    if not all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        return None

    train_df = pd.read_csv(train_csv)
    if len(train_df) < 500:
        return None

    test_df = pd.read_csv(test_csv)
    baseline_tps_all = test_df[~test_df['transition_point'].isin([-1])]['transition_point'].tolist()
    baseline_avg_tp_with_0_10 = sum(baseline_tps_all) / len(baseline_tps_all) if baseline_tps_all else -1

    baseline_tps_exclude = test_df[test_df['transition_point'].isin(range(1, 10))]['transition_point'].tolist()
    baseline_avg_tp_exclude_0_10 = sum(baseline_tps_exclude) / len(baseline_tps_exclude) if baseline_tps_exclude else -1

    baseline_elo_accs = {i: {'correct': 0, 'total': 0} for i in range(10)}
    num_test_positions = 0

    for _, row in test_df.iterrows():
        tp = row.get('transition_point', -1)
        if tp == -1:
            continue
        num_test_positions += 1

        for elo in range(10):
            baseline_elo_accs[elo]['total'] += 1
            if tp == 0:
                baseline_elo_accs[elo]['correct'] += 1
            elif tp == 10:
                pass
            else:
                if elo >= tp:
                    baseline_elo_accs[elo]['correct'] += 1

    baseline_elo_acc_values = {}
    for elo in range(10):
        if baseline_elo_accs[elo]['total'] > 0:
            baseline_elo_acc_values[elo] = baseline_elo_accs[elo]['correct'] / baseline_elo_accs[elo]['total']
        else:
            baseline_elo_acc_values[elo] = 0.0

    print(f"\n{'='*80}")
    print(f"Training concept: {concept_name}")
    print(f"Train samples: {len(train_df)}")
    print(f"Baseline avg TP (with 0,10): {baseline_avg_tp_with_0_10:.2f}")
    print(f"Baseline avg TP (excl 0,10): {baseline_avg_tp_exclude_0_10:.2f}")
    print(f"{'='*80}")

    train_dataset = ConceptDataset(train_csv, all_moves_dict)
    val_dataset = ConceptDataset(val_csv, all_moves_dict)
    test_dataset = ConceptDataset(test_csv, all_moves_dict)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=len(elo_dict), shuffle=False, num_workers=cfg.num_workers)

    base_model = MAIA2Model(len(all_moves), elo_dict, cfg)
    ckpt = torch.load(cfg.model_path, map_location=device)

    if all(k.startswith('module.') for k in ckpt['model_state_dict'].keys()):
        base_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()})
    else:
        base_model.load_state_dict(ckpt['model_state_dict'])

    base_model.to(device)
    base_model.eval()

    fc_new = nn.Linear(cfg.dim_vit, len(all_moves)).to(device)
    fc_new.weight.data = base_model.fc_1.weight.data.clone()
    fc_new.bias.data = base_model.fc_1.bias.data.clone()

    model = PolicyHeadModel(base_model, fc_new).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc_new.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(cfg.max_epochs):
        model.fc_new.train()
        epoch_loss = 0
        epoch_steps = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            boards, labels, elos_self, elos_oppo, legal_moves, side_info, _ = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]

            logits_maia = model(boards, elos_self, elos_oppo)
            loss = criterion(logits_maia, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

        avg_train_loss = epoch_loss / epoch_steps

        model.fc_new.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                boards, labels, elos_self, elos_oppo, legal_moves, side_info, _ = [
                    x.to(device) if torch.is_tensor(x) else x for x in batch
                ]

                logits_maia = model(boards, elos_self, elos_oppo)
                loss = criterion(logits_maia, labels)

                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.fc_new.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.fc_new.load_state_dict(best_model_state)

    base_model.eval()
    baseline_predictions_per_elo = {elo: [] for elo in range(11)}
    with torch.no_grad():
        for batch in test_loader:
            boards, labels, elos_self, elos_oppo, legal_moves, side_info, _ = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            logits_base = base_model(boards, elos_self, elos_oppo)[0]
            _, predictions = calculate_accuracy(logits_base, labels, legal_moves)
            for i in range(0, len(predictions), len(elo_dict)):
                if i + len(elo_dict) <= len(predictions):
                    for j in range(len(elo_dict)):
                        move_idx = predictions[i+j].item()
                        baseline_predictions_per_elo[j].append(all_moves[move_idx])

    elo_accs_all, ft_avg_tp_with_0_10, ft_avg_tp_exclude_0_10, finetuned_predictions_per_elo = evaluate_model(
        model, test_loader, elo_dict, device, all_moves, return_predictions=True
    )

    elo_accs = {elo: elo_accs_all[elo] for elo in range(10)}

    test_fens = test_df['fen'].tolist()

    print(f"\nResults for {concept_name}:")
    print(f"  Baseline avg TP (with 0,10): {baseline_avg_tp_with_0_10:.2f}")
    print(f"  Finetuned avg TP (with 0,10): {ft_avg_tp_with_0_10:.2f}")
    print(f"  Baseline avg TP (excl 0,10): {baseline_avg_tp_exclude_0_10:.2f}")
    print(f"  Finetuned avg TP (excl 0,10): {ft_avg_tp_exclude_0_10:.2f}")
    print(f"\n  Per-ELO test accuracy (0-9):")
    print(f"    {'ELO':<6} {'Baseline':>10} {'Finetuned':>10} {'Improvement':>12}")
    print(f"    {'-'*40}")
    for elo in range(10):
        baseline_acc = baseline_elo_acc_values[elo]
        finetuned_acc = elo_accs[elo]
        improvement = finetuned_acc - baseline_acc
        print(f"    {elo:<6} {baseline_acc:>10.4f} {finetuned_acc:>10.4f} {improvement:>12.4f}")

    return {
        'concept_name': concept_name,
        'baseline_avg_tp_with_0_10': baseline_avg_tp_with_0_10,
        'baseline_avg_tp_exclude_0_10': baseline_avg_tp_exclude_0_10,
        'finetuned_avg_tp_with_0_10': ft_avg_tp_with_0_10,
        'finetuned_avg_tp_exclude_0_10': ft_avg_tp_exclude_0_10,
        'baseline_elo_accuracies': baseline_elo_acc_values,
        'finetuned_elo_accuracies': elo_accs,
        'train_samples': len(train_df),
        'num_test_positions': num_test_positions,
        'test_fens': test_fens,
        'baseline_predictions_per_elo': baseline_predictions_per_elo,
        'finetuned_predictions_per_elo': finetuned_predictions_per_elo
    }

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetune_config.yaml'), 'r') as f:
        cfg_dict = yaml.safe_load(f)

    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    cfg = Config(cfg_dict)
    if not hasattr(cfg, 'model_path'):
        cfg.model_path = cfg_dict.get('pretrained_path', 'weights.v2.pt')
    if not os.path.isabs(cfg.model_path):
        cfg.model_path = os.path.join(ROOT, cfg.model_path)
    if not hasattr(cfg, 'patience'):
        cfg.patience = 3

    utils.seed_everything(cfg.seed)

    concept_base_dir = os.path.join(ROOT, 'dataset', 'concept-filtered-externalization')
    output_dir = os.path.join(ROOT, 'extern', 'policy_distillation', 'results', 'results_per_concept_head_only')
    os.makedirs(output_dir, exist_ok=True)

    concept_dirs = glob.glob(os.path.join(concept_base_dir, '*'))
    results = []

    for concept_dir in sorted(concept_dirs):
        if not os.path.isdir(concept_dir):
            continue

        concept_name = os.path.basename(concept_dir)
        result = train_concept(concept_name, concept_dir, cfg, device)

        if result is not None:
            results.append(result)

            with open(os.path.join(output_dir, f'{concept_name}_result.pkl'), 'wb') as f:
                pickle.dump(result, f)

    summary_path = os.path.join(output_dir, 'all_results_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'='*80}")
    print("All concepts completed!")
    print(f"{'='*80}")
    print(f"\n{'Concept':<40} {'Samples':>8} {'Base(w)':>9} {'FT(w)':>8} {'Base(x)':>9} {'FT(x)':>8} {'Impr(x)':>9}")
    print("-" * 110)

    for result in results:
        tp_improve = result['baseline_avg_tp_exclude_0_10'] - result['finetuned_avg_tp_exclude_0_10']
        print(f"{result['concept_name']:<40} {result['train_samples']:>8} "
              f"{result['baseline_avg_tp_with_0_10']:>9.2f} {result['finetuned_avg_tp_with_0_10']:>8.2f} "
              f"{result['baseline_avg_tp_exclude_0_10']:>9.2f} {result['finetuned_avg_tp_exclude_0_10']:>8.2f} "
              f"{tp_improve:>9.2f}")

    total_weights = sum(r['num_test_positions'] for r in results if r['baseline_avg_tp_with_0_10'] != -1)

    avg_baseline_tp_with = sum(r['baseline_avg_tp_with_0_10'] * r['num_test_positions']
                                for r in results if r['baseline_avg_tp_with_0_10'] != -1) / total_weights if total_weights > 0 else -1
    avg_finetuned_tp_with = sum(r['finetuned_avg_tp_with_0_10'] * r['num_test_positions']
                                 for r in results if r['finetuned_avg_tp_with_0_10'] != -1) / total_weights if total_weights > 0 else -1

    total_weights_exclude = sum(r['num_test_positions'] for r in results if r['baseline_avg_tp_exclude_0_10'] != -1)
    avg_baseline_tp_exclude = sum(r['baseline_avg_tp_exclude_0_10'] * r['num_test_positions']
                                   for r in results if r['baseline_avg_tp_exclude_0_10'] != -1) / total_weights_exclude if total_weights_exclude > 0 else -1
    avg_finetuned_tp_exclude = sum(r['finetuned_avg_tp_exclude_0_10'] * r['num_test_positions']
                                    for r in results if r['finetuned_avg_tp_exclude_0_10'] != -1) / total_weights_exclude if total_weights_exclude > 0 else -1

    avg_baseline_elo_accs = {elo: 0.0 for elo in range(10)}
    avg_finetuned_elo_accs = {elo: 0.0 for elo in range(10)}

    for elo in range(10):
        weighted_baseline = sum(r['baseline_elo_accuracies'][elo] * r['num_test_positions'] for r in results)
        weighted_finetuned = sum(r['finetuned_elo_accuracies'][elo] * r['num_test_positions'] for r in results)
        total_weight = sum(r['num_test_positions'] for r in results)

        avg_baseline_elo_accs[elo] = weighted_baseline / total_weight if total_weight > 0 else 0.0
        avg_finetuned_elo_accs[elo] = weighted_finetuned / total_weight if total_weight > 0 else 0.0

    avg_baseline_grouped = {
        '(,1200]': (avg_baseline_elo_accs[0] + avg_baseline_elo_accs[1]) / 2,
        '(1200,1400]': (avg_baseline_elo_accs[2] + avg_baseline_elo_accs[3]) / 2,
        '(1400,1600]': (avg_baseline_elo_accs[4] + avg_baseline_elo_accs[5]) / 2,
        '(1600,1800]': (avg_baseline_elo_accs[6] + avg_baseline_elo_accs[7]) / 2,
        '(1800,2000]': (avg_baseline_elo_accs[8] + avg_baseline_elo_accs[9]) / 2,
    }
    avg_finetuned_grouped = {
        '(,1200]': (avg_finetuned_elo_accs[0] + avg_finetuned_elo_accs[1]) / 2,
        '(1200,1400]': (avg_finetuned_elo_accs[2] + avg_finetuned_elo_accs[3]) / 2,
        '(1400,1600]': (avg_finetuned_elo_accs[4] + avg_finetuned_elo_accs[5]) / 2,
        '(1600,1800]': (avg_finetuned_elo_accs[6] + avg_finetuned_elo_accs[7]) / 2,
        '(1800,2000]': (avg_finetuned_elo_accs[8] + avg_finetuned_elo_accs[9]) / 2,
    }

    print(f"\n{'='*80}")
    print("OVERALL AVERAGE ACROSS ALL CONCEPTS")
    print(f"{'='*80}")
    print(f"\nTransition Points:")
    print(f"  Baseline avg TP (with 0,10):    {avg_baseline_tp_with:.3f}")
    print(f"  Finetuned avg TP (with 0,10):   {avg_finetuned_tp_with:.3f}")
    print(f"  Improvement (with 0,10):        {avg_baseline_tp_with - avg_finetuned_tp_with:.3f}")
    print(f"\n  Baseline avg TP (excl 0,10):    {avg_baseline_tp_exclude:.3f}")
    print(f"  Finetuned avg TP (excl 0,10):   {avg_finetuned_tp_exclude:.3f}")
    print(f"  Improvement (excl 0,10):        {avg_baseline_tp_exclude - avg_finetuned_tp_exclude:.3f}")

    print(f"\nPer-ELO Group Accuracies (weighted average across all concepts):")
    print(f"{'Range':<15} {'Baseline':>10} {'Finetuned':>10} {'Improvement':>12}")
    print("-" * 50)
    for range_name in ['(,1200]', '(1200,1400]', '(1400,1600]', '(1600,1800]', '(1800,2000]']:
        baseline_acc = avg_baseline_grouped[range_name]
        finetuned_acc = avg_finetuned_grouped[range_name]
        improvement = finetuned_acc - baseline_acc
        print(f"{range_name:<15} {baseline_acc:>10.4f} {finetuned_acc:>10.4f} {improvement:>12.4f}")


    final_summary = {
        'num_concepts': len(results),
        'transition_points': {
            'baseline_with_0_10': avg_baseline_tp_with,
            'finetuned_with_0_10': avg_finetuned_tp_with,
            'improvement_with_0_10': avg_baseline_tp_with - avg_finetuned_tp_with,
            'baseline_exclude_0_10': avg_baseline_tp_exclude,
            'finetuned_exclude_0_10': avg_finetuned_tp_exclude,
            'improvement_exclude_0_10': avg_baseline_tp_exclude - avg_finetuned_tp_exclude
        },
        'per_elo_accuracies': {
            'baseline': avg_baseline_elo_accs,
            'finetuned': avg_finetuned_elo_accs,
            'improvement': {elo: avg_finetuned_elo_accs[elo] - avg_baseline_elo_accs[elo] for elo in range(10)}
        },
        'per_elo_grouped': {
            'baseline': avg_baseline_grouped,
            'finetuned': avg_finetuned_grouped,
            'improvement': {k: avg_finetuned_grouped[k] - avg_baseline_grouped[k] for k in avg_baseline_grouped.keys()}
        },
        'per_concept_results': results
    }

    summary_json_path = os.path.join(output_dir, 'final_summary.json')
    import json
    with open(summary_json_path, 'w') as f:
        json.dump(final_summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Final summary saved to: {summary_json_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    import chess
    main()
