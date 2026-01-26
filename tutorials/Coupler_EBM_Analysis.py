import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from squadds.ml.pipeline import SQuADDSAnalysisPipeline
from interpret import show

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

def plot_ebm_shape_functions(
    ebm_result: dict,
    param_name: str,
    top_n: int = 5,
    save_path: str = None
) -> None:
    """
    Plot the learned shape functions for a trained EBM.
    Adapted from Tutorial 10.
    """
    # Handle both direct dictionary from pipeline or the EBM object wrapper
    if 'ebm_model' in ebm_result:
        ebm = ebm_result['ebm_model'].model # Extract underlying EBM from wrapper
        term_names = ebm.term_names_
        term_importances = ebm.term_importances()
    else:
        # Fallback for manual training dicts
        ebm = ebm_result['model']
        term_names = ebm_result['term_names']
        term_importances = ebm_result['term_importances']
    
    # Get top N terms by importance
    sorted_idx = np.argsort(term_importances)[::-1][:top_n]
    
    # Determine grid layout
    n_cols = min(3, len(sorted_idx))
    n_rows = (len(sorted_idx) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for plot_idx, term_idx in enumerate(sorted_idx):
        ax = axes[plot_idx]
        term_name = term_names[term_idx]
        importance = term_importances[term_idx]
        
        # Extract the shape function data
        term_data = ebm.term_scores_[term_idx]
        
        # Safely get bins for this term
        try:
            term_bins = ebm.bins_[term_idx]
        except (IndexError, KeyError):
            term_bins = []
        
        # Check if data is 2D (interaction term)
        is_2d = hasattr(term_data, 'ndim') and term_data.ndim == 2
        
        # Check if it's a main effect or interaction
        if ' x ' in term_name or is_2d:
            # Interaction term - show as heatmap
            if is_2d:
                im = ax.imshow(
                    term_data.T, 
                    aspect='auto', 
                    origin='lower',
                    cmap='RdBu_r'
                )
                plt.colorbar(im, ax=ax, label='Contribution')
                if ' x ' in term_name:
                    ax.set_xlabel(term_name.split(' x ')[0])
                    ax.set_ylabel(term_name.split(' x ')[1])
            else:
                ax.text(0.5, 0.5, f'Interaction\\n{term_name}', 
                       transform=ax.transAxes, ha='center', va='center')
        elif len(term_bins) > 0 and hasattr(term_bins[0], '__len__') and len(term_bins[0]) > 0:
            # Main effect - show as line plot
            bins = np.array(term_bins[0])
            
            # Ensure term_data is 1D
            plot_data = np.array(term_data).flatten()
            n_scores = len(plot_data)
            
            if n_scores > 0:
                # Compute bin width for padding
                bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
                
                # Create evenly spaced x values spanning the bin range
                x_min = bins[0] - bin_width / 2
                x_max = bins[-1] + bin_width / 2
                x_vals = np.linspace(x_min, x_max, n_scores)
                
                ax.plot(x_vals, plot_data, linewidth=2, color='steelblue')
                ax.fill_between(x_vals, plot_data, alpha=0.3, color='steelblue')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.set_xlabel(term_name)
            ax.set_ylabel('Contribution to prediction')
        else:
            # Intercept or term without bins - show simple visualization
            plot_data = np.array(term_data).flatten()
            n_scores = len(plot_data)
            
            if n_scores > 1:
                x_vals = np.arange(n_scores)
                ax.plot(x_vals, plot_data, linewidth=2, color='steelblue')
                ax.fill_between(x_vals, plot_data, alpha=0.3, color='steelblue')
            else:
                val = float(plot_data[0]) if n_scores == 1 else 0.0
                ax.axhline(y=val, color='steelblue', linewidth=2)
                ax.text(0.5, 0.5, f'Value: {val:.4f}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.set_xlabel(term_name)
            ax.set_ylabel('Contribution')
        
        ax.set_title(f'{term_name}\\n(importance: {importance:.4f})')
    
    # Hide empty subplots
    for idx in range(len(sorted_idx), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Shape Functions for {param_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    # 1. Load Data
    print("Loading data...")
    # Adjust path if running from root or tutorials folder
    data_path = "coupler_capacitance_data.csv"
    if not os.path.exists(data_path):
        data_path = "../coupler_capacitance_data.csv"
    if not os.path.exists(data_path):
        # Last resort try relative to script location
        data_path = os.path.join(os.path.dirname(__file__), "../coupler_capacitance_data.csv")
        
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # 2. Define Features and Targets
    features = ['cap_gap', 'cap_width', 'finger_count', 'finger_length']
    targets = ['bottom_to_bottom', 'bottom_to_ground', 'ground_to_ground', 
               'top_to_bottom', 'top_to_ground', 'top_to_top']

    # 3. Initialize Pipeline
    pipeline = SQuADDSAnalysisPipeline(random_state=42)

    # 4. Run Analysis with Tutorial 10 Hyperparameters
    # Matching Tutorial 10:
    # test_size=0.2 (from line 487)
    # interactions=10
    # max_interaction_bins=32
    # outer_bags=8
    # inner_bags=4
    # learning_rate=0.01
    # min_samples_leaf=5
    
    print("Running EBM Analysis...")
    results = pipeline.analyze(
        df,
        features,
        targets,
        test_size=0.2, # Matched Tutorial 10
        ebm_kwargs={
            'interactions': 10,       # Matched Tutorial 10 default
            'outer_bags': 8,          # Matched Tutorial 10
            'inner_bags': 4,          # Matched Tutorial 10
            'learning_rate': 0.01,    # Matched Tutorial 10
            'min_samples_leaf': 5,    # Matched Tutorial 10
            'max_bins': 256,          # Matched Tutorial 10
            'max_interaction_bins': 32 # Matched Tutorial 10 arg
        },
        symbolic_kwargs={
            'niterations': 20, 
            'populations': 15, 
            'population_size': 33,
            'maxsize': 25,
        },
        feature_threshold=0.01 
    )

    # 5. Visualize Results
    print("Generating plots...")
    for target in targets:
        # Save plots to figures directory
        save_file = f"figures/ebm_shape_{target}.png"
        plot_ebm_shape_functions(results[target], target, top_n=6, save_path=save_file)

if __name__ == "__main__":
    main()
