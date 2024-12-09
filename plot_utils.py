import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional

def plot_data(
    data: Union[torch.Tensor, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: tuple = (10, 8),
    cbar: bool = True,
    line_color: str = 'blue',
    line_style: str = '-',
    **kwargs
):
    """
    Plots data as a line plot (for 1D) or heatmaps (for 2D and 3D tensors/arrays).

    Parameters:
    - data (torch.Tensor or np.ndarray): Input data to plot. Must be 1D, 2D, or 3D.
    - save_path (str, optional): Path to save the figure. If None, the figure is not saved.
    - show (bool, default=True): Whether to display the plot.
    - title (str, optional): Title for the entire figure.
    - cmap (str, default='viridis'): Colormap for the heatmaps.
    - figsize (tuple, default=(10, 8)): Size of the figure.
    - cbar (bool, default=True): Whether to display color bars (for heatmaps).
    - line_color (str, default='blue'): Color of the line plot (for 1D data).
    - line_style (str, default='-'): Line style of the line plot (for 1D data).
    - **kwargs: Additional keyword arguments passed to seaborn.heatmap or matplotlib.pyplot.plot.

    Raises:
    - ValueError: If input data is not 1D, 2D, or 3D tensor/array.
    - TypeError: If input data is not a torch.Tensor or np.ndarray.
    """
    
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(data, torch.Tensor):
        if data.dim() not in [1, 2, 3]:
            raise ValueError(f"Expected 1D, 2D, or 3D tensor, but got {data.dim()}D tensor.")
        data_np = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        if data.ndim not in [1, 2, 3]:
            raise ValueError(f"Expected 1D, 2D, or 3D array, but got {data.ndim}D array.")
        data_np = data
    else:
        raise TypeError(f"Input data must be a torch.Tensor or np.ndarray, but got {type(data)}.")
    
    # Determine the dimension of the data
    ndim = data_np.ndim

    # Create figure
    fig = None  # Placeholder for figure

    if ndim == 1:
        # 1D Data: Plot a line plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data_np, color=line_color, linestyle=line_style, **kwargs)
        ax.set_title(title if title else '1D Line Plot')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        plt.tight_layout()
        
    elif ndim == 2:
        # 2D Data: Plot a single heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data_np, ax=ax, cmap=cmap, cbar=cbar, **kwargs)
        ax.set_title(title if title else '2D Heatmap')
        plt.tight_layout()
        
    elif ndim == 3:
        # 3D Data: Plot multiple heatmaps in a grid
        n_heatmaps = data_np.shape[0]
        data_to_plot = [data_np[i] for i in range(n_heatmaps)]
        
        # Determine grid size for subplots (square-ish)
        cols = int(np.ceil(np.sqrt(n_heatmaps)))
        rows = int(np.ceil(n_heatmaps / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        fig.suptitle(title, fontsize=16) if title else None
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten()
        
        # Plot each heatmap
        for idx, ax in enumerate(axes_flat):
            if idx < n_heatmaps:
                sns.heatmap(
                    data_to_plot[idx],
                    ax=ax,
                    cmap=cmap,
                    cbar=(cbar and n_heatmaps == 1),
                    **kwargs
                )
                ax.set_title(f'Heatmap {idx + 1}') if not title else None
            else:
                # Hide any unused subplots
                ax.axis('off')
        
        # Add a single color bar if multiple heatmaps and color bars are desired
        if n_heatmaps > 1 and cbar:
            # Positioning the color bar on the right side of the subplots
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            # Determine the common scale
            vmin = min([d.min() for d in data_to_plot])
            vmax = max([d.max() for d in data_to_plot])
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, label='Value')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1]) if n_heatmaps > 1 else plt.tight_layout()
        
    else:
        # This should not happen due to earlier checks
        raise ValueError("Data must be 1D, 2D, or 3D.")
    
    # Save the figure if save_path is provided
    if save_path and fig:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show the plot if requested
    if show and fig:
        plt.show()
    
    # Close the figure to free memory
    if fig:
        plt.close(fig)
