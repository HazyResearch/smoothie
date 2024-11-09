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
    "acc_group": "\textsc{Dist-Acc}",
    "rouge2_group": "\textsc{Dist-R2}",
}
ALPACA_MODELS = [
    "yi-large-preview",
    "Storm-7B",
    "Nanbeige-Plus-Chat-v0.1",
    "Qwen1.5-110B-Chat",
    "FsfairX-Zephyr-Chat-v0.1",
    "Meta-Llama-3-70B-Instruct",
    "mistral-large-2402",
    "Ein-70B-v0.1",
    "gemini-pro",
    "claude-2"
]

# PLOT CONSTANTS
FONT_SIZE = 20
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


    
def compute_smoothie_train_test_selection_results_latex_table(table_output_path=None):
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
        "cnn_dailymail",
        "definition_extraction",
        "e2e_nlg",
        "squad",
        "trivia_qa",
        "web_nlg",
        "xsum",
    ]
    
    model_groups = ["3b_ensemble", "7b_ensemble"]
    
    # Methods to evaluate
    smoothie_test = "smoothie_sample_independent_test_time_all-mpnet-base-v2"
    smoothie_train = "smoothie_sample_independent_train_time_all-mpnet-base-v2"
    
    # First collect all results
    results = {}
    for model_group in model_groups:
        results[model_group] = {}
        for dataset in datasets:
            scores_fpath = RESULTS_DIR / dataset / model_group / "scores.json"
            scores = json.loads(scores_fpath.read_text())[TASK2METRIC[dataset]]
            
            # Calculate the "random" baseline (average of ensemble scores)
            ensemble_scores = list(scores["ensemble"].values())
            random_score = np.mean(ensemble_scores)
            
            # Get Smoothie scores and oracle score
            smoothie_test_score = scores["smoothie"][smoothie_test]
            smoothie_train_score = scores["smoothie"][smoothie_train]
            best_on_val_score = scores["labeled_oracle"]

            results[model_group][dataset] = {
                "random": np.round(random_score * 100, 1),
                "test": np.round(smoothie_test_score * 100, 1),
                "train": np.round(smoothie_train_score * 100, 1),
                "oracle": np.round(best_on_val_score * 100, 1)
            }
    
    # Create table column specification
    column_spec = "@{}l" + "c" * (len(datasets) + 1) + "@{}"
    
    # Create LaTeX table
    latex_content = [
        "\\begin{table}[t]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{7pt}",
        f"\\begin{{tabular}}{{{column_spec}}}"
    ]
    
    # Header rows
    latex_content.extend([
        "\\toprule",
        "& & " + " & ".join([dataset2name.get(d, d.title()) for d in datasets]) + " \\\\ \\midrule"
    ])
    
    # For each model group and metric type
    for model_group in ["3", "7"]:
        model_key = f"{model_group.lower()}b_ensemble"
        
        # Add multi-row label for model group
        latex_content.append(f"\\multirow{{3}}{{*}}{{{model_group}B}}")
        
        # Collect scores for each dataset
        all_scores = {dataset: {metric: results[model_key][dataset][metric] for metric in ["random", "test", "train", "oracle"]} for dataset in datasets}
        
        # Determine the best unsupervised and overall methods for each dataset
        best_unsupervised = {dataset: [m for m in ["random", "test"] if all_scores[dataset][m] == max(all_scores[dataset][m] for m in ["random", "test"])] for dataset in datasets}
        best_overall = {dataset: [m for m in all_scores[dataset] if all_scores[dataset][m] == max(all_scores[dataset].values())] for dataset in datasets}
        
        # Add each metric row
        for metric, metric_name in [
            ("random", "\\small{\\textsc{Random}}"), 
            ("test", "\\small{\\nameglobal}"), 
            ("train", "\\small{\\nameglobal-train}"),
            ("oracle", "\\small{\\textsc{Best-on-Val}}")
        ]:
            scores = [results[model_key][dataset][metric] for dataset in datasets]
            score_str = " & ".join([
                f"\\underline{{\\textbf{{{score:.1f}}}}}" if metric in ["random", "test", "train"] and metric in best_unsupervised[dataset] and metric in best_overall[dataset] else
                f"\\underline{{{score:.1f}}}" if metric in ["random", "test", "train"] and metric in best_unsupervised[dataset] else
                f"\\textbf{{{score:.1f}}}" if metric in best_overall[dataset] else
                f"{score:.1f}"
                for score, dataset in zip(scores, datasets)
            ])
            latex_content.append(f"& {metric_name} & {score_str} \\\\")
        
        # Add midrule after each model group, except for the last one
        if model_group != "7":
            latex_content.append("\\midrule")
    
    # Table footer
    caption = "We compare \\nameglobal to \\nameglobal-train, for which weights are learned on a hold-out set. We provide results from baseline methods for reference. Underlined values are the best performing \\textit{unsupervised} methods. Bold values are the best performing \\textit{overall} methods. We report rouge2 scores for CNN, XSum, WebNLG, and E2E, and accuracy for the rest. All metrics are scaled to 0-100."
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{" + caption + "}",
        "\\label{tab:smoothie-train-test-selection-comparison}",
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



def compute_smoothie_train_test_routing_results_latex_table(table_output_path=None):
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
    smoothie_local_test = "smoothie_sample_dependent_test_time_1_all-mpnet-base-v2"
    smoothie_global_test = "smoothie_sample_independent_test_time_all-mpnet-base-v2"
    smoothie_local_train = "smoothie_sample_dependent_train_time_50_all-mpnet-base-v2"
    smoothie_global_train = "smoothie_sample_independent_train_time_all-mpnet-base-v2"
    
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
            smoothie_local_test_score = scores["smoothie"][smoothie_local_test]
            smoothie_global_test_score = scores["smoothie"][smoothie_global_test]
            smoothie_local_train_score = scores["smoothie"][smoothie_local_train]
            smoothie_global_train_score = scores["smoothie"][smoothie_global_train]
            pair_rm_score = scores["pair_rm"]
            labeled_knn_score = scores["labeled_knn"]
            pick_random_score = scores["pick_random"]
            
            results[model_group][dataset] = {
                "random": np.round(pick_random_score * 100, 1),
                "pair_rm": np.round(pair_rm_score * 100, 1),
                "labeled_knn": np.round(labeled_knn_score * 100, 1),
                "best_in_ensemble": np.round(best_in_ensemble * 100, 1),
                "smoothie_local_test": np.round(smoothie_local_test_score * 100, 1),
                "smoothie_global_test": np.round(smoothie_global_test_score * 100, 1),
                "smoothie_local_train": np.round(smoothie_local_train_score * 100, 1),
                "smoothie_global_train": np.round(smoothie_global_train_score * 100, 1),
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
        ("smoothie_global_test", "\\small{\\nameglobal}"),
        ("smoothie_local_test", "\\small{\\namelocal}"),
        ("smoothie_global_train", "\\small{\\nameglobal-\textsc{train}}"),
        ("smoothie_local_train", "\\small{\\namelocal-\textsc{train}}"),
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
        if method == "smoothie_local_test":
            # Don't add a \midrule after the last row
            latex_content.append(f"{method_name} & {' & '.join(scores)} \\\\")
        else:
            latex_content.append(f"{method_name} & {' & '.join(scores)} \\\\ \\midrule")
    
    # Table footer
    caption = "We compare \\namelocal to \\namelocal-train, for which weights are learned on a hold-out set, on the 3B and 7B ensembles for multi-task distributions. \\distacc and \\distr are measured with accuracy and rouge2 respectively. Bold values indicate the best performing method for each dataset and model size. Metrics are scaled to 0-100. Other baseline methods are provided for comparison."
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{" + caption + "}",
        "\\label{tab:smoothie-train-test-local-comparison}",
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


def plot_smoothie_neighborhood_size_ablation(plot_output_path=None):
    """
    Generates a LaTeX table comparing Smoothie results across datasets and model sizes.
    Table is transposed with datasets as columns. Scores are multiplied by 100 for percentage representation.


    
    Args:
        plot_output_path (str, optional): Path where the plot file should be saved.
                                         If None, prints to console.
    """
    import json
    import numpy as np
    from pathlib import Path
    
    datasets = [
        "acc_group",
        "rouge2_group",
    ]
    dataset2name = {
        "acc_group": "Distr-Acc",
        "rouge2_group": "Distr-Rouge2",
    }
    
    model_groups = ["3b_ensemble", "7b_ensemble"]
    
    # Methods to evaluate
    smoothie_methods = [
        "smoothie_sample_dependent_test_time_1_all-mpnet-base-v2",
        "smoothie_sample_dependent_test_time_5_all-mpnet-base-v2",
        "smoothie_sample_dependent_test_time_10_all-mpnet-base-v2",
        "smoothie_sample_dependent_test_time_20_all-mpnet-base-v2",
        "smoothie_sample_dependent_test_time_50_all-mpnet-base-v2",
        "smoothie_sample_dependent_test_time_100_all-mpnet-base-v2",
    ]
    neighborhood_sizes = [1, 5, 10, 20, 50, 100]
    
    
    # First collect results
    results = {}
    for model_group in model_groups:
        results[model_group] = {}
        for dataset in datasets:
            results[model_group][dataset] = {}
            scores_fpath = RESULTS_DIR / dataset / model_group / "scores.json"
            scores = json.loads(scores_fpath.read_text())
                    
            # Get Smoothie scores for different neighborhood sizes
            for method, neighborhood_size in zip(smoothie_methods, neighborhood_sizes):
                smoothie_score = scores["smoothie"][method]
                results[model_group][dataset][f"smoothie_{neighborhood_size}"] = np.round(smoothie_score * 100, 1)
            
    import matplotlib.pyplot as plt

    for dataset in datasets:
        plt.figure(figsize=(16, 10))
        
        for model_group in model_groups:
            neighborhood_scores = [results[model_group][dataset][f"smoothie_{size}"] for size in neighborhood_sizes]
            plt.plot(neighborhood_sizes, neighborhood_scores, marker='o', label=model_group, linewidth=5)
        
        plt.title(f"Effect of Neighborhood Size\n({dataset2name[dataset]})", pad=PLOT_TITLE_PAD, fontweight=PLOT_TITLE_FONTWEIGHT)
        plt.xlabel("$n_0$", labelpad=PLOT_AXIS_LABEL_PAD, fontweight=PLOT_AXIS_LABEL_FONTWEIGHT)
        plt.ylabel("Score", labelpad=PLOT_AXIS_LABEL_PAD, fontweight=PLOT_AXIS_LABEL_FONTWEIGHT)
        plt.xticks(neighborhood_sizes)
        #plt.legend()
        plt.grid(True)

        ax = plt.gca()
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        plot_file_path = Path(plot_output_path).parent / f"smoothie_neighborhood_size_ablation_{dataset}.png"
        plt.savefig(plot_file_path, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
        plt.close()
        print(f"Plot saved to {plot_file_path}")


def compute_smoothie_embedding_ablation_latex_table(table_output_path=None):
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
    smoothie_local_sentence_bert = "smoothie_sample_dependent_test_time_1_all-mpnet-base-v2"
    smoothie_local_bge_small = "smoothie_sample_dependent_test_time_1_bge-small-en-v1.5"
    
    
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
            smoothie_local_score = scores["smoothie"][smoothie_local_sentence_bert]
            smoothie_bge_small_score = scores["smoothie"][smoothie_local_bge_small]
            pair_rm_score = scores["pair_rm"]
            labeled_knn_score = scores["labeled_knn"]
            pick_random_score = scores["pick_random"]
            
            results[model_group][dataset] = {
                "random": np.round(pick_random_score * 100, 1),
                "pair_rm": np.round(pair_rm_score * 100, 1),
                "labeled_knn": np.round(labeled_knn_score * 100, 1),
                "best_in_ensemble": np.round(best_in_ensemble * 100, 1),
                "smoothie_local_sentence_bert": np.round(smoothie_local_score * 100, 1),
                "smoothie_local_bge_small": np.round(smoothie_bge_small_score * 100, 1),
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
        ("smoothie_local_bge_small", "\\small{\\namelocal} (BGE-small)"),
        ("smoothie_local_sentence_bert", "\\small{\\namelocal} (SBERT)"),
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
        if method == "smoothie_local_sentence_bert":
            # Don't add a \midrule after the last row
            latex_content.append(f"{method_name} & {' & '.join(scores)} \\\\")
        else:
            latex_content.append(f"{method_name} & {' & '.join(scores)} \\\\ \\midrule")
    
    # Table footer
    caption = "Comparing \\namelocal with different embeddings on the 3B and 7B ensembles for multi-task distributions. \\distacc and \\distr are measured with accuracy and rouge2 respectively. Bold values indicate the best performing method for each dataset and model size. Metrics are scaled to 0-100."
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{" + caption + "}",
        "\\label{tab:smoothie-routing-embedding-ablation}",
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


def compute_smoothie_global_many_ensembles_plot(plot_output_path=None):
    """
    We consider the 50 different ensembles and plot a bar plot of the rank of the model Smoothie-global selects.
    """
    datasets = [
        "cnn_dailymail",
        "definition_extraction",
        "e2e_nlg",
        "squad",
        "trivia_qa",
        "web_nlg",
        "xsum",
    ]
    
    
    
    # Methods to evaluate
    smoothie = "smoothie_sample_independent_test_time_all-mpnet-base-v2"
    
    # First collect all results
    smoothie_ranks = [0] * 7 # the maximum rank is 7
    for model_group in MODEL_GROUPS:
        if model_group in ["3b_ensemble", "7b_ensemble"]:
            continue

        for dataset in datasets:
            scores_fpath = RESULTS_DIR / dataset / model_group / "scores.json"
            scores = json.loads(scores_fpath.read_text())[TASK2METRIC[dataset]]
            
            # Get the scores of the models in the ensemble, ranked
            ensemble_scores = list(scores["ensemble"].values())
            ensemble_scores_sorted = sorted(ensemble_scores, reverse=True)
            
            # Get rank of Smoothie in the ensemble
            rank = ensemble_scores_sorted.index(scores["smoothie"][smoothie])
            smoothie_ranks[rank] += 1
             
    print(f"Total number of ensembles: {sum(smoothie_ranks)}")
    print(smoothie_ranks)

    # Plot the relative frequency of ranks as a barplot
    plt.figure(figsize=FIGURE_SIZE)
    
    # Create barplot for relative frequency
    plt.bar(
        x=list(range(1, len(smoothie_ranks) + 1)),
        height=smoothie_ranks,
        color='lightblue',
        edgecolor='black',  # Add edge color
        linewidth=3  # Make edges thicker
    )
    plt.title(f'Rank of Model Selected by Smoothie-Global', 
                pad=PLOT_TITLE_PAD, 
                fontweight=PLOT_TITLE_FONTWEIGHT)

    plt.xlabel('Rank', labelpad=PLOT_AXIS_LABEL_PAD, fontweight=PLOT_AXIS_LABEL_FONTWEIGHT)
    plt.ylabel('Frequency', labelpad=PLOT_AXIS_LABEL_PAD, fontweight=PLOT_AXIS_LABEL_FONTWEIGHT)
    
    # Center x-labels relative to bars
    num_ranks = len(smoothie_ranks)
    plt.xticks(np.arange(num_ranks) + 1, range(1, num_ranks + 1))
    
    # Only show horizontal grid lines
    plt.grid(axis='y', linestyle='--', linewidth=2.5)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Make the edges of the plot thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)

    # Adjust layout and display the plot
    plt.tight_layout()

    if plot_output_path:
        plt.savefig(plot_output_path, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
        print(f"Plot saved to {plot_output_path}")
    else:
        plt.show()

compute_smoothie_train_test_selection_results_latex_table(table_output_path="tables/smoothie_train_test_selection_comparison.tex")
compute_smoothie_train_test_routing_results_latex_table(table_output_path="tables/smoothie_train_test_routing_comparison.tex")
plot_smoothie_neighborhood_size_ablation(plot_output_path="plots/smoothie_neighborhood_size_ablation.png")
compute_smoothie_embedding_ablation_latex_table(table_output_path="tables/smoothie_routing_embedding_ablation.tex")
compute_smoothie_global_many_ensembles_plot(plot_output_path="plots/smoothie_global_many_ensembles.png")