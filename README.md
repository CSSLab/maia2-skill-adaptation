# Mechanisms of Skill Adaptation in Generative Models: Chess as a Model System

This repo contains the code for our paper investigating how generative models adapt their outputs to different skill levels to reveal whether adaptation occurs through dynamic internal concept awareness or through modulation of concept externalization.

## Adjusting Concept Externalization

### Model Calibration

Run with

```bash
python -m maia2-skill-adaptation.test.externalization.maia2_ft
```

to callibrate the concept externalization of model at lower skill levels to higher ones in move prediction.

### Transcoder Training

We implement sparse and wide transcoders to simulate the behavior of the original model's FFN MLPs. The training process uses topK ReLU activations to directly control sparsity and mitigate dead latents. Run the transcoder training with:

```bash
python -m maia2-skill-adaptation.train.transcoder_train \
  --model_path /path/to/maia-2-weights \
  --data_root /path/to/lichess_data \
  --hidden_dim 16384 \
  --k 256 \
  --layer_idx 6
```

After training, we can evaluate and visualize the reconstruction fidelity by:

```bash
python -m maia2-skill-adaptation.train.transcoder_eval
```

and relevant codes for circuit analysis are included in test/externalization/circuit_analysis/.

## Adjusting Concept Awareness

### SAE Training

The training process supports the choice of SAE hyperparameters and hook sites to extract Maia-2 internal representations. To start a standard SAE training, run:

```bash
python -m maia2-skill-adaptation.train.train_sae [arguments]
```

Key Arguments for SAE Training
The following arguments control the SAE training process:

--sae_dim: Dimension of the SAE \
--l1_coefficient: L1 regularization coefficient for SAE training loss\
--sae_attention_heads: Whether to attach hooks on attention heads for SAE training \
--sae_residual_streams: Whether to attach hooks on residual streams for SAE training \
--sae_mlp_outputs: Whether to attach hooks on MLP outputs for SAE training

### Concept-Aware Feature Extraction

To get the internal activations of our trained SAEs on Maia-2 test positions, run

```bash
python -m maia2-skill-adaptation.train.generate_activations
```

Then, with the SAE internals we can extract most salient SAE features for offensive and defensive square-wise threat concepts by:

```bash
python -m maia2-skill-adaptation.test.threat_awareness
```

### Feature-Mediated Intervention

Run with:

```bash
python -m maia2-skill-adaptation.test.intervention.run_sae_intervention
```

to examine how model's behaviour changes when increasing the concept understanding level of it!
