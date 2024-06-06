import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
from PIL import Image



def visualize_and_save_attention(attention_weights, save_dir="attentions", formats=("png", "pdf")):
    """
    Visualizes and saves the attention maps.
    
    Parameters:
    - attention_weights: A torch.Tensor of shape [batch_size, num_heads, seq_length_text, seq_length_image]
    - save_dir: Directory where to save the attention maps
    - formats: A tuple of formats to save the image, e.g., ("png", "pdf")
    """
    # Ensure the saving directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size, num_heads, seq_length_text, seq_length_image = attention_weights.shape
    
    # Plot only the first batch's 8 heads in one row
    if batch_size > 0 and num_heads >= 8:
        # Extract attention weights for normalization
        attention_maps = attention_weights[0, :8, :, :].cpu().detach().numpy()
        attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min()) * 10
        vmin = 0
        vmax = 1
        
        fig, axes = plt.subplots(1, 8, figsize=(24, 3))  # Create a figure with 8 subplots
        
        for head_idx in range(8):
            # Extract the attention weights for the current head and sample
            attention_map = attention_maps[head_idx, :, :]
            
            # Create a subplot for each head
            ax = axes[head_idx]
            sns.heatmap(attention_map, cmap='plasma', ax=ax, cbar=True, cbar_kws={'shrink': 0.5}, vmin=vmin, vmax=vmax)
            ax.set_title(f'Head {head_idx+1}')
            ax.axis('off')
        
        # Adjust the layout and save the combined figure as PNG
        plt.tight_layout()
        png_path = os.path.join(save_dir, "v_t_attention_batch0_heads.png")
        plt.savefig(png_path, format="png", dpi=600)
        plt.close()  # Close the figure to free memory

        # Convert PNG to PDF
        if "pdf" in formats:
            image = Image.open(png_path)
            rgb_image = image.convert('RGB')  # Convert RGBA to RGB
            pdf_path = os.path.join(save_dir, "v_t_attention_batch0_heads.pdf")
            rgb_image.save(pdf_path, "PDF", quality=95)


# Example usage
# Assuming attention_weights is your attention weights tensor with shape [14, 8, 60, 808]
# attention_weights = torch.rand(14, 8, 60, 808)  # Example data
# visualize_and_save_attention(attention_weights)
