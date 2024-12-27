import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

with open("maia2_activations_for_probe.pickle", "rb") as f:
    activations_dict = pickle.load(f)


# dictionary of maia-2 activations on positive and negative examples of structure [square_name][positive/negative][layer] for probe/steering vector training

def train_linear_probe(activations_dict, save_path="./probes", num_epochs=10, batch_size=32, learning_rate=0.01):
    """
    Train linear probes for each concept in activations_dict, save them, and measure test accuracy.

    Args:
        activations_dict (dict): Dictionary containing positive and negative activations for each concept.
        save_path (str): Directory to save the trained probes.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        dict: Trained probes for each concept.
    """
    import os
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists

    probes = {}  # Dictionary to store trained linear probes for each concept

    for concept, data in activations_dict.items():
        print(f"Training linear probe for concept: {concept}")

        # Extract positive and negative activations
        pos_activations = data['positive']  # Shape: [num_pos, 8, 1024]
        neg_activations = data['negative']  # Shape: [num_neg, 8, 1024]

        # Reduce activations to [num_samples, 1024] by taking the mean across dimension 1
        pos_activations = pos_activations.mean(dim=1)
        neg_activations = neg_activations.mean(dim=1)

        # Balance the dataset
        num_pos = pos_activations.size(0)
        max_neg = int(1.4 * num_pos)  # Allow at most 1.3× the number of positive examples

        # Randomly sample from negative examples if there are too many
        if neg_activations.size(0) > max_neg:
            indices = torch.randperm(neg_activations.size(0))[:max_neg]
            neg_activations = neg_activations[indices]

        # Create labels: 1 for positive, 0 for negative
        pos_labels = torch.ones(pos_activations.size(0))
        neg_labels = torch.zeros(neg_activations.size(0))

        # Concatenate activations and labels
        inputs = torch.cat([pos_activations, neg_activations], dim=0)  # Shape: [num_samples, 1024]
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Split into training and test sets (80-20 split)
        dataset = TensorDataset(inputs, labels)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Define a simple linear layer as the probe
        probe = nn.Linear(inputs.size(1), 1)  # input_dim -> 1 output (binary classification)
        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
        optimizer = optim.SGD(probe.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            for batch_inputs, batch_labels in train_loader:
                optimizer.zero_grad()

                # Forward pass
                logits = probe(batch_inputs).squeeze()
                loss = criterion(logits, batch_labels)

                # Backward pass
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Save the trained probe for the concept
        torch.save(probe.state_dict(), os.path.join(save_path, f"{concept}_probe.pth"))
        probes[concept] = probe

        # Compute test accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_inputs, batch_labels in test_loader:
                logits = probe(batch_inputs).squeeze()
                predictions = (torch.sigmoid(logits) > 0.5).long()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
            test_accuracy = correct / total
            print(f"Test accuracy for concept '{concept}': {test_accuracy:.4f}")

    return probes


print(activations_dict['e4']['positive']['transformer block 1 hidden states'].shape)

maia2_activations = {}
for key in activations_dict:
    maia2_activations[key] = {}
    for label in activations_dict[key]:
        maia2_activations[key][label] = activations_dict[key][label]['transformer block 1 hidden states']

print(maia2_activations['b2']['negative'].shape)

# probes = train_linear_probe(maia2_activations)


import torch
import os


def load_linear_probes(probe_dir='./probes', input_dim=1024):
    """
    Load all linear probes from a directory.

    Args:
        probe_dir (str): Directory containing the saved linear probes.
        input_dim (int): Dimensionality of the probe input (default: 1024).

    Returns:
        dict: Dictionary mapping concept names to loaded linear probes.
    """
    probes = {}
    for filename in os.listdir(probe_dir):
        if filename.endswith('_probe.pth'):
            concept_name = filename.split('_probe.pth')[0]
            probe = torch.nn.Linear(input_dim, 1)  # Initialize a linear probe
            probe.load_state_dict(torch.load(os.path.join(probe_dir, filename)))
            probes[concept_name] = probe
    return probes


def reverse_pool_probe_weights(probe):
    """
    Reverse pool the linear probe weights to match activation shape [8, 1024].

    Args:
        probe (nn.Linear): Trained linear probe.

    Returns:
        torch.Tensor: Reverse pooled weights of shape [8, 1024].
    """
    with torch.no_grad():
        W = probe.weight.squeeze()  # Shape: [1024]
        W_normalized = W / W.norm()  # Normalize the weights
        W_reversed = W_normalized.unsqueeze(0).expand(8, -1)  # Shape: [8, 1024]
    return W_reversed


def save_reverse_pooled_weights(probes, output_file):
    """
    Save reverse pooled weights for all probes into a dictionary.

    Args:
        probes (dict): Dictionary of linear probes.
        output_file (str): Path to save the reverse pooled weights dictionary.

    Returns:
        None
    """
    reverse_pooled_weights = {}
    for concept_name, probe in probes.items():
        reverse_pooled_weights[concept_name] = reverse_pool_probe_weights(probe).numpy()

    # Save as a PyTorch dictionary file
    torch.save(reverse_pooled_weights, output_file)
    print(f"Reverse pooled weights saved to {output_file}")


# output_file = "./reverse_pooled_weights.pth"

# # Load probes
# probes = load_linear_probes()

# # Save reverse pooled weights
# save_reverse_pooled_weights(probes, output_file)


# Example usage
probes = train_linear_probe(activations_dict, save_path="./retrained_probes")
