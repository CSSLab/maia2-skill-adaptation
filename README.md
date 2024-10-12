# MAIA2-SAE: Understanding Skill Adaptation in Transformers Using Sparse Autoencoders

## SAE Training

The training process supports the choice of SAE hyperparameters and hook sites to extract Maia-2 internal representations. To start a standard SAE training, run:

```bash
python -m maia2-sae.train.train [arguments]
```

Key Arguments for SAE Training
The following arguments control the SAE training process:

--sae_dim: Dimension of the SAE \
--l1_coefficient: L1 regularization coefficient for SAE training loss\
--sae_attention_heads: Whether to attach hooks on attention heads for SAE training \
--sae_residual_streams: Whether to attach hooks on residual streams for SAE training \
--sae_mlp_outputs: Whether to attach hooks on MLP outputs for SAE training

## Model Steering with SAE

After training the SAE, we can use it to perform mediated interventions on the frozen Maia-2 model. To run an intervention, run:

```bash
python -m maia2-sae.test.intervention --intervention [intervention_type]
```

The --intervention argument supports three types of interventions:

vanilla: Amplify the activations of selected SAE features with a selected range of magnitude \
random: Amplify the activations of random SAE features with a selected range of magnitude \
patching: Activation-patching style intervention
