import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

def visualize_and_save_attention(attention_weights, save_dir="attention_maps_1"):
    """
    Visualizes and saves the attention maps.
    
    Parameters:
    - attention_weights: A torch.Tensor of shape [batch_size, num_heads, seq_length_text, seq_length_image]
    - save_dir: Directory where to save the attention maps
    """
    # Ensure the saving directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size, num_heads, seq_length_text, seq_length_image = attention_weights.shape
    # Loop through each sample in the batch
    for batch_idx in range(batch_size):
        # Loop through each attention head
        for head_idx in range(num_heads):
            # Extract the attention weights for the current head and sample
            attention_map = attention_weights[batch_idx, head_idx, :, :].cpu().detach().numpy()
            
            # Plot the attention map
            plt.figure(figsize=(12, 10))
            sns.heatmap(attention_map, cmap='viridis')
            
            # Save the figure
            plt.savefig(os.path.join(save_dir, f"v_t_attention_b{batch_idx}_h{head_idx}.png"))
            plt.close()  # Close the figure to free memory

# Example usage
# Assuming attention_weights is your attention weights tensor with shape [14, 8, 60, 808]
# attention_weights = torch.rand(14, 8, 60, 808)  # Example data
# visualize_and_save_attention(attention_weights)
