# Mechanisms of Skill Adaptation in Generative Models: Chess as a Model System

## SAE Training

The training process supports the choice of SAE hyperparameters and hook sites to extract Maia-2 internal representations. To start a standard SAE training, run:

```bash
python -m maia2-sae.train.train [arguments]
```

or for training SAEs with JumpRelu activations:

```bash
python -m maia2-sae.train.train_with_jumprelu [arguments]
```

Key Arguments for SAE Training
The following arguments control the SAE training process:

--sae_dim: Dimension of the SAE \
--l1_coefficient: L1 regularization coefficient for SAE training loss\
--sae_attention_heads: Whether to attach hooks on attention heads for SAE training \
--sae_residual_streams: Whether to attach hooks on residual streams for SAE training \
--sae_mlp_outputs: Whether to attach hooks on MLP outputs for SAE training

## Concept-Aware Feature Extraction

To get the internal activations of our trained SAEs on Maia-2 test positions, run

```bash
python -m maia2-sae.train.generate_activations
```

Then, with the SAE internals we can extract most salient SAE features for offensive and defensive square-wise threat concepts by:

```bash
python -m maia2-sae.test.threat_awareness
```

Afterwards, to visualize the salient features in SAE hidden layer, run:

```bash
python -m maia2-sae.test.plot_best_feature_awareness
```

## Feature-Mediated Intervention

Run with:

```bash
python -m maia2-sae.test.run_sae_intervention
```

to examine how model's behaviour changes when increasing the concept understanding level of it!
