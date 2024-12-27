import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

with open("maia2_activations_for_steer.pickle", "rb") as f:
    activations_dict = pickle.load(f)

#dictionary of maia-2 activations on positive and negative examples of structure [square_name][positive/negative][layer] for probe/steering vector training

print(activations_dict['e4']['positive']['transformer block 1 hidden states'].shape)

maia2_activations = {}
for key in activations_dict:
    maia2_activations[key] = {}
    for label in activations_dict[key]:
        maia2_activations[key][label] = activations_dict[key][label]['transformer block 1 hidden states']

print(maia2_activations['b2']['negative'].shape)


def train_steering_vectors(activations_dict):
    """
    Train steering vectors for each concept based on positive and negative activations.

    Args:
        activations_dict (dict): Dictionary containing positive and negative activations for each concept.

    Returns:
        dict: Dictionary mapping concept names to trained steering vectors (torch.Tensor).
    """
    steering_vectors = {}

    for concept, data in activations_dict.items():
        print(f"Training steering vector for concept: {concept}")

        # Extract positive and negative activations
        pos_activations = data['positive']  # Shape: [num_pos_samples, 8, 1024]
        neg_activations = data['negative']  # Shape: [num_neg_samples, 8, 1024]

        # Reduce activations to [num_samples, 1024] by taking the mean across the 8 dimension
        pos_activations_reduced = pos_activations.mean(dim=1)  # Shape: [num_pos_samples, 1024]
        neg_activations_reduced = neg_activations.mean(dim=1)  # Shape: [num_neg_samples, 1024]

        # Compute the mean activation for positive and negative examples
        pos_mean = pos_activations_reduced.mean(dim=0)  # Shape: [1024]
        neg_mean = neg_activations_reduced.mean(dim=0)  # Shape: [1024]

        # Compute the steering vector (difference between positive and negative means)
        steering_vector = pos_mean - neg_mean

        # Normalize the steering vector
        steering_vector_normalized = steering_vector / steering_vector.norm()

        # Store the normalized steering vector
        steering_vectors[concept] = steering_vector_normalized

        print(f"Steering vector trained for concept: {concept}")

    return steering_vectors

steering_vecs = train_steering_vectors(maia2_activations)

torch.save(steering_vecs, "./steering_vectors_2.pth")



