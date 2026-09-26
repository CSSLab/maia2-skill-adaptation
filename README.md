# Playing Dumb: How a Chess Transformer Simulates Users Across Skill Levels

We study Maia-2, a chess transformer that predicts human moves conditioned on player skill, and ask whether its lower-skill play reflects less internal knowledge or a different use of retained knowledge. Linear probes for 172 chess concepts stay nearly constant across skill conditions, while fine-tuning only the policy head or steering sparse autoencoder features recovers stronger decisions from lower-skill representations.

## Setup

```bash
conda env create -f environment.yml
conda activate maiainterp
```

<!--
Pretrained Maia-2 (Rapid) checkpoint: `weights.v2.pt` — [download](https://drive.google.com/file/d/1zG5m2rFqoXtdMiBKCZolYO6LonPy5I_u/view?usp=sharing)

SAE on residual streams: `sae/best_jrsaes_2023-11-16384-1-res.pt` — [download](https://drive.google.com/file/d/1p_IapA5qm6UO9YkF9MGAAVUvV4NRPi1j/view?usp=sharing)
-->

Place `weights.v2.pt` and `sae/best_jrsaes_2023-11-16384-1-res.pt` in the repository root. Scripts resolve paths relative to the repository root, or to `SKILL_ADAPTATION_ROOT` if set.

## Pipeline

### 1. Knowledge encoding: concept probes

Train linear probes on the residual stream after each Transformer block, for every concept and skill condition:

```bash
python train/train_probes.py --layer_key "transformer block 0 hidden states" --output_dir probes/layer0_efficient
python train/train_probes.py --layer_key "transformer block 1 hidden states" --output_dir probes/layer1_efficient
```

### 2. Knowledge externalization

#### 2a. Policy head fine-tuning (Head SFT)

Fine-tune only the policy head, with the backbone frozen, on the concept-filtered Blundered Transitional Dataset:

```bash
python extern/policy_distillation/ft_per_concept_head_only.py
```

Configuration: `extern/policy_distillation/finetune_config.yaml`

#### 2b. SAE feature steering

Select the SAE features most associated with each concept, then amplify them at Transformer block 2 during inference. `--mode random` runs the random-feature control. Probes trained in step 1 are evaluated on the steered representations.

```bash
python extern/feature_steering/select_sae_features.py
python extern/feature_steering/sae_intervention.py --layer 1 --mode salient
python extern/feature_steering/sae_intervention.py --layer 1 --mode random
```

## Data

`dataset/blundered-transitional-dataset/test_moves.csv` is the test split of the Blundered Transitional Dataset: positions where Maia-2's prediction moves from a blunder at lower skill conditions to the best move at higher conditions.
