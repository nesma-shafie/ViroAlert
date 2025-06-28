import matplotlib.pyplot as plt
import seaborn as sns


# Function to create bar plots for a given hyperparameter
def plot_hyperparameter_results(metrics,hyperparam, values, results_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    fig.suptitle(f'Performance Metrics by {hyperparam.replace("_", " ").title()}', fontsize=16)
    
    for i, metric in enumerate(metrics):
        # Calculate mean metric value for each hyperparameter value
        means = [results_df[results_df[hyperparam] == val][metric].mean() for val in values]
        
        # Create bar plot
        axes[i].bar([str(val) for val in values], means, color=sns.color_palette("husl", 3)[i])
        axes[i].set_title(metric)
        axes[i].set_xlabel(hyperparam.replace("_", " ").title())
        axes[i].set_ylabel(f'Mean {metric}')
        
        # Rotate x-axis labels for readability if needed
        if hyperparam == 'learning_rate':
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'hyperparam_{hyperparam}_plot.png')
    plt.show()

def plot_hyperparameter_results_enhanced(metrics,hyperparam, values, results_df):
    metrics_colors = {
        'MSE': 'tab:blue',
        'RMSE': 'tab:orange',
        'Pearson': 'tab:green'
    }

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        means = []
        stds = []

        for val in values:
            subset = results_df[results_df[hyperparam] == val]
            means.append(subset[metric].mean())
            stds.append(subset[metric].std())

        plt.errorbar(
            values, means, yerr=stds, label=metric,
            marker='o', linestyle='-', capsize=5,
            color=metrics_colors[metric]
        )

    plt.title(f'Performance Metrics vs {hyperparam.replace("_", " ").title()}', fontsize=14)
    plt.xlabel(hyperparam.replace("_", " ").title())
    plt.ylabel('Metric Value')
    plt.xticks(values if hyperparam != 'learning_rate' else [float(f"{v:.0e}") for v in values])
    if hyperparam == 'learning_rate':
        plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'hyperparam_{hyperparam}_enhanced_plot.png')
    plt.show()
