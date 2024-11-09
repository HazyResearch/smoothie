"""
This script computes the correlations between Smoothie weights and model rankings on all sets of tasks.
"""

import sys
sys.path.append("../")

from pathlib import Path
import json
from src.ensembles import MODEL_GROUPS
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from typing import List, Dict
from scipy import stats
from collections import Counter

# CONSTANTS
RESULTS_DIR = Path("smoothie_data/multi_model_results")
TASK2METRIC = {
    "cnn_dailymail": "rouge2",
    "definition_extraction": "definition_extraction_acc",
    "e2e_nlg": "rouge2",
    "squad": "squad_acc",
    "trivia_qa": "trivia_qa_acc",
    "web_nlg": "rouge2",
    "xsum": "rouge2",
}
dataset2name = {
    "cnn_dailymail": "CNN",
    "definition_extraction": "Def. Ext.",
    "e2e_nlg": "E2E",
    "squad": "SQuAD",
    "trivia_qa": "TriviaQA",
    "web_nlg": "WebNLG",
    "xsum": "XSum",
    "acc_group": "\\distacc",
    "rouge2_group": "\\distr",
}
plot_dataset2name = {
    "acc_group": "Distr-Acc",
    "rouge2_group": "Distr-Rouge2",
}

# PLOT CONSTANTS
FONT_SIZE = 28
PLOT_DPI = 600
PLOT_BBOX_INCHES = "tight"
PLOT_TITLE_FONTWEIGHT = "bold"
PLOT_TICK_FONTWEIGHT = "bold"
PLOT_TITLE_PAD = 20
PLOT_AXIS_LABEL_FONTWEIGHT = "bold"
PLOT_AXIS_LABEL_PAD = 15
PLOT_COLORS = ["lightblue", "lightorange"]
FIGURE_SIZE = (10, 7)
#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = FONT_SIZE

def compute_smoothie_latex_table(table_output_path=None):
    """
    Generates a LaTeX table comparing Smoothie results across datasets and model sizes.
    Table is transposed with datasets as columns. Scores are multiplied by 100 for percentage representation.
    
    Args:
        table_output_path (str, optional): Path where the LaTeX table file should be saved.
                                         If None, prints to console.
    """
    import json
    import numpy as np
    from pathlib import Path
    
    datasets = [
        "acc_group",
        "rouge2_group",
    ]
    
    model_groups = ["3b_ensemble", "7b_ensemble"]
    
    # Methods to evaluate
    smoothie_local = "smoothie_sample_dependent_test_time_1_all-mpnet-base-v2"
    smoothie_global = "smoothie_sample_independent_test_time_all-mpnet-base-v2"
    
    # First collect all results
    results = {}
    for model_group in model_groups:
        results[model_group] = {}
        for dataset in datasets:
            scores_fpath = RESULTS_DIR / dataset / model_group / "scores.json"
            scores = json.loads(scores_fpath.read_text())
            
            # Calculate the best in the ensemble
            ensemble_scores = list(scores["ensemble"].values())
            best_in_ensemble = max(ensemble_scores)
            
            # Get Smoothie scores and oracle score
            smoothie_local_score = scores["smoothie"][smoothie_local]
            smoothie_global_score = scores["smoothie"][smoothie_global]
            pair_rm_score = scores["pair_rm"]
            labeled_knn_score = scores["labeled_knn"]
            pick_random_score = scores["pick_random"]
            
            results[model_group][dataset] = {
                "random": np.round(pick_random_score * 100, 1),
                "pair_rm": np.round(pair_rm_score * 100, 1),
                "labeled_knn": np.round(labeled_knn_score * 100, 1),
                "best_in_ensemble": np.round(best_in_ensemble * 100, 1),
                "smoothie_local": np.round(smoothie_local_score * 100, 1),
                "smoothie_global": np.round(smoothie_global_score * 100, 1),
            }
    
    # Create LaTeX table
    latex_content = [
        "\\begin{table}[t]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.3}",
        "\\setlength{\\tabcolsep}{10pt}",
        "\\begin{tabular}{lccccc}"  # Modified column specification
    ]
    
    # Header rows with model size groups
    latex_content.extend([
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{3B}} & \\multicolumn{2}{c}{\\textbf{7B}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(l){4-5}",
        "\\textbf{Method} & \\distacc & \\distr & \\distacc & \\distr \\\\ \\midrule"
    ])
    
    # Define methods and their display names
    methods = [
        ("random", "\\small{\\random}"),
        ("pair_rm", "\\small{\\pairrm}"),
        ("labeled_knn", "\\small{\\lknn}"),
        ("best_in_ensemble", "\\small{\\bestensemble}"),
        ("smoothie_global", "\\small{\\nameglobal}"),
        ("smoothie_local", "\\small{\\namelocal}"),
    ]
    
    # For each method, create a row with scores from both model sizes
    for method, method_name in methods:
        scores = []
        # Get best scores for determining bold formatting
        best_scores = {
            model_group: {
                dataset: max(results[model_group][dataset].values())
                for dataset in datasets
            }
            for model_group in model_groups
        }
        
        # Collect scores for this method across all configurations
        for model_group in model_groups:
            for dataset in datasets:
                score = results[model_group][dataset][method]
                is_best = np.isclose(score, best_scores[model_group][dataset])
                score_str = f"\\textbf{{{score:.1f}}}" if is_best else f"{score:.1f}"
                scores.append(score_str)
        
        # Add the row to the table
        if method == "smoothie_local":
            # Don't add a \midrule after the last row
            latex_content.append(f"{method_name} & {' & '.join(scores)} \\\\")
        else:
            latex_content.append(f"{method_name} & {' & '.join(scores)} \\\\ \\midrule")
    
    # Table footer
    caption = "Comparing \\namelocal to baseline methods on the 3B and 7B ensembles for multi-task distributions. \\distacc and \\distr are measured with accuracy and rouge2 respectively. Bold values indicate the best performing method for each dataset and model size. Metrics are scaled to 0-100."
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{" + caption + "}",
        "\\label{tab:smoothie-local-comparison}",
        "\\end{table}"
    ])
    
    # Join all lines with newlines
    latex_table = "\n".join(latex_content)
    
    if table_output_path:
        # Save to file
        Path(table_output_path).write_text(latex_table)
        print(f"LaTeX table saved to {table_output_path}")
    else:
        # Print to console
        print(latex_table)


def compute_smoothie_sample_ranks(plot_output_path=None):
    """
    We plot the distribution of model ranks that Smoothie selects.
    """

    datasets = [
        "acc_group",
        "rouge2_group",
    ]
    ensembles = {
        "3b_ensemble": ["dolly-3b", "incite-3b", "pythia-2.8b", "gemma-2b"],
        "7b_ensemble": ["llama-2-7b", "mistral-7b", "vicuna-7b", "gemma-7b", "nous-capybara"],
    }
    
    model_groups = ["3b_ensemble", "7b_ensemble"]
    
    # Methods to evaluate
    smoothie_local = "smoothie_sample_dependent_test_time_1_all-mpnet-base-v2"

    for dataset in datasets:
        for model_group in model_groups:
            results_fpath = RESULTS_DIR / dataset / model_group / "sample_scores.json"
            sample_scores = json.loads(results_fpath.read_text())["ensemble"]
            
            num_samples = len(list(sample_scores.values())[0])

            # Sample scores is a dictionary of model to a list of scores (of length equal to the dataset)
            sample_ranks = []
            for sample_id in range(num_samples):
                scores = {model: sample_scores[model][sample_id] for model in ensembles[model_group]}
                sample_ranks.append(compute_ranks(scores))
            
            # Load smoothie selections
            smoothie_selections_fpath = RESULTS_DIR / dataset / model_group / f"{smoothie_local}_test.json"
            smoothie_weights = json.loads(smoothie_selections_fpath.read_text())["smoothie_weights"]
            selected_models = [ensembles[model_group][np.argmax(weights)] for weights in smoothie_weights]
            selected_ranks = [sample_ranks[i][selected_models[i]] for i in range(num_samples)]
            
            rank_counter = Counter(selected_ranks)
            rank_relative_freqs = [rank_counter[rank] / num_samples for rank in range(1, len(ensembles[model_group]) + 1)]
            print(rank_relative_freqs)
            # Plot the relative frequency of ranks as a barplot
            plt.figure(figsize=FIGURE_SIZE)
            
            # Create barplot for relative frequency
            plt.bar(
                x=list(range(1, len(ensembles[model_group]) + 1)),
                height=rank_relative_freqs,
                color='lightblue',
                edgecolor='black',  # Add edge color
                linewidth=3  # Make edges thicker
            )
            model_group_name = "3B" if model_group == "3b_ensemble" else "7B"
            
            plt.title(f'{plot_dataset2name[dataset]} ({model_group_name})', 
                     pad=PLOT_TITLE_PAD, 
                     fontweight=PLOT_TITLE_FONTWEIGHT)

            plt.xlabel('Rank', labelpad=PLOT_AXIS_LABEL_PAD, fontweight=PLOT_AXIS_LABEL_FONTWEIGHT)
            plt.ylabel('Relative Frequency', labelpad=PLOT_AXIS_LABEL_PAD, fontweight=PLOT_AXIS_LABEL_FONTWEIGHT)
            
            # Center x-labels relative to bars
            num_ranks = len(ensembles[model_group])
            plt.xticks(np.arange(num_ranks) + 1, range(1, num_ranks + 1))
            
            # Only show horizontal grid lines
            plt.grid(axis='y', linestyle='--', linewidth=2.5)

            # Set y-axis limit from 0 to 1
            plt.ylim(0, 1.1)

            

            # Remove top and right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            # Make the edges of the plot thicker
            for spine in plt.gca().spines.values():
                spine.set_linewidth(3)
        
            # Adjust layout and display the plot
            plt.tight_layout()

            if plot_output_path:
                plot_fpath = Path(plot_output_path) / f"{dataset}_{model_group}_rank_relative_frequency.png"
                plt.savefig(plot_fpath, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
                print(f"Plot saved to {plot_fpath}")
            else:
                plt.show()


def compute_ranks(scores):
    """
    Convert a dictionary of scores to a dictionary of ranks.
    Handles ties by assigning the same rank and incrementing next ranks accordingly.
    
    Args:
        scores (dict): Dictionary mapping models to their scores
        
    Returns:
        dict: Dictionary mapping models to their ranks
        
    Example:
        >>> scores = {'model_a': 95, 'model_b': 90, 'model_c': 95, 'model_d': 85}
        >>> compute_ranks(scores)
        {'model_a': 1, 'model_c': 1, 'model_b': 3, 'model_d': 4}
    """
    # Sort models by score in descending order
    sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    
    ranks = {}
    current_rank = 1
    prev_score = None
    tied_count = 0
    
    for i, (model, score) in enumerate(sorted_items):
        if score == prev_score:
            # Same score as previous model, assign same rank
            ranks[model] = current_rank - tied_count
        else:
            # Different score, assign current rank
            ranks[model] = current_rank
            tied_count = 0
        
        prev_score = score
        current_rank += 1
        tied_count += 1
    
    return ranks



compute_smoothie_latex_table(table_output_path="tables/smoothie_combined_routing_table.tex")
compute_smoothie_sample_ranks(plot_output_path="plots")