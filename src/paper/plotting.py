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
    "definition_extraction": "Def Ext.",
    "e2e_nlg": "E2E",
    "squad": "SQuAD",
    "trivia_qa": "Trivia QA",
    "web_nlg": "Web NLG",
    "xsum": "XSum",
}
SINGLE_TASK_DATASETS = ["squad", "trivia_qa", "definition_extraction", "cnn_dailymail",  "xsum", "e2e_nlg", "web_nlg"]



def plot_smoothie_selection_ranks(datasets: List[str], smoothie_train: str, smoothie_test: str, title: str, save_path: str= None):
    """
    Plot the average rank of the generation selected by smoothie train, smoothie test, and random selection.

    Args:
        datasets (List[str]): List of dataset names to include in the plot.
        smoothie_train (str): The name of the Smoothie train selection method.
        smoothie_test (str): The name of the Smoothie test selection method.
        title (str): The title for the plot.
        save_path (str, optional): The path to save the plot. If None, the plot is not saved.

    This function reads scores from JSON files, calculates ranks for Smoothie train, Smoothie test,
    and random selections, and creates a grouped bar plot comparing their average ranks across
    different datasets. The plot is customized with appropriate labels, colors, and styling.

    Note: This function assumes the existence of global variables RESULTS_DIR, TASK2METRIC,
    dataset2name, and MODEL_GROUPS.
    """

    np.random.seed(0)
    
    data = []
    for dataset in datasets:
        for model_group in MODEL_GROUPS:
            scores_fpath = RESULTS_DIR / dataset / model_group / "scores.json"
            scores = json.loads(scores_fpath.read_text())[TASK2METRIC[dataset]]
            ensemble_scores = scores["ensemble"]
            smoothie_train_score = scores["smoothie"][smoothie_train]
            smoothie_test_score = scores["smoothie"][smoothie_test]
            
            # Calculate rank
            ensemble_scores_list = list(ensemble_scores.values())
            smoothie_train_rank = sorted(ensemble_scores_list, reverse=True).index(smoothie_train_score) + 1
            smoothie_test_rank = sorted(ensemble_scores_list, reverse=True).index(smoothie_test_score) + 1

            # IMPORTANT: We only consider ensembles where there are 5 models
            if len(ensemble_scores) != 5:
                continue
            
            data.append({
                "Dataset": dataset2name[dataset],
                "Model Group": model_group,
                "Smoothie Train Rank": smoothie_train_rank,
                "Smoothie Test Rank": smoothie_test_rank,
            })
    
    df = pd.DataFrame(data)
    # Group by Dataset and calculate mean ranks
    mean_ranks = df.groupby('Dataset')[['Smoothie Train Rank', 'Smoothie Test Rank']].mean().reset_index()

    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))  # Increased figure size

    # Create the grouped bar plot
    x = np.arange(len(mean_ranks['Dataset']))
    width = 0.25  # Reduced width to accommodate three bars

    ax.bar(x - width, mean_ranks['Smoothie Train Rank'], width, label='Smoothie Train', color='#4e79a7', alpha=0.8)
    ax.bar(x, mean_ranks['Smoothie Test Rank'], width, label='Smoothie Test', color='#59a14f', alpha=0.8)

    # Customize the plot
    ax.set_ylabel('Average Rank', fontsize=16)  # Increased font size
    ax.set_title(title, fontsize=20, fontweight='bold')  # Increased font size
    ax.set_xticks(x)
    ax.set_xticklabels(mean_ranks['Dataset'], ha='center', fontsize=14)  # Increased font size
    ax.legend(fontsize=14, loc='upper left')  # Moved legend to upper left

    # Set y-axis to go from 1 to 5
    ax.set_ylim(1, 5)
    ax.set_yticks(range(1, 6))
    ax.tick_params(axis='y', labelsize=14)  # Increased y-axis tick label size

    # Remove gridlines
    ax.grid(False)

    # Add value labels on top of each bar
    for i, v in enumerate(mean_ranks['Smoothie Train Rank']):
        ax.text(i - width, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=12)  # Increased font size
    for i, v in enumerate(mean_ranks['Smoothie Test Rank']):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=12)  # Increased font size

    # Add dashed line for Pick-Random Baseline
    ax.axhline(y=3, color='red', linestyle='--', linewidth=2)
    ax.text(ax.get_xlim()[1], 3, 'Pick-Random Baseline', ha='right', va='bottom', color='red', fontsize=14)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def generate_smoothie_selection_ranking_tables(datasets: List[str], smoothie_train: str, smoothie_test: str) -> str:
    """
    Generate two side-by-side LaTeX tables showing rankings and scores for Smoothie methods.
    Scores are shown as percentages, with Oracle and Random as reference points.
    """
    np.random.seed(0)
    
    data = []
    for dataset in datasets:
        dataset_results = []
        for model_group in MODEL_GROUPS:
            scores_fpath = RESULTS_DIR / dataset / model_group / "scores.json"
            scores = json.loads(scores_fpath.read_text())[TASK2METRIC[dataset]]
            ensemble_scores = scores["ensemble"]
            
            # Skip if not exactly 5 models
            if len(ensemble_scores) != 5:
                continue
                
            smoothie_train_score = scores["smoothie"][smoothie_train]
            smoothie_test_score = scores["smoothie"][smoothie_test]
            ensemble_scores_list = list(ensemble_scores.values())
            oracle_score = max(ensemble_scores_list)
            random_score = np.mean(ensemble_scores_list)
            
            # Calculate ranks
            smoothie_train_rank = sorted(ensemble_scores_list, reverse=True).index(smoothie_train_score) + 1
            smoothie_test_rank = sorted(ensemble_scores_list, reverse=True).index(smoothie_test_score) + 1
            
            dataset_results.append({
                "train_rank": smoothie_train_rank,
                "test_rank": smoothie_test_rank,
                "train_score": smoothie_train_score * 100,  # Convert to percentage
                "test_score": smoothie_test_score * 100,
                "oracle_score": oracle_score * 100,
                "random_score": random_score * 100,
                "train_is_best": smoothie_train_rank == 1,
                "test_is_best": smoothie_test_rank == 1
            })
        
        # Calculate metrics for this dataset
        if dataset_results:
            data.append({
                "dataset": dataset2name[dataset],
                # Rankings
                "train_rank": np.mean([r["train_rank"] for r in dataset_results]),
                "test_rank": np.mean([r["test_rank"] for r in dataset_results]),
                "train_best_rate": np.mean([r["train_is_best"] for r in dataset_results]) * 100,
                "test_best_rate": np.mean([r["test_is_best"] for r in dataset_results]) * 100,
                # Scores 
                "train_score": np.mean([r["train_score"] for r in dataset_results]),
                "test_score": np.mean([r["test_score"] for r in dataset_results]),
                "random_score": np.mean([r["random_score"] for r in dataset_results]),
                "oracle_score": np.mean([r["oracle_score"] for r in dataset_results])
            })
    
    # Generate side-by-side tables
    latex_tables = [
        "\\begin{table}[ht]",
        "\\begin{minipage}[t]{0.48\\textwidth}",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{l@{\\hspace{1em}}cc@{\\hspace{1em}}cc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{Avg Rank}} & \\multicolumn{2}{c}{\\textbf{R1 (\\%)}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "\\textbf{Dataset} & Train & Test & Train & Test \\\\",
        "\\midrule"
    ]
    
    # Add data rows for rankings table
    for row in data:
        latex_row = (
            f"{row['dataset']} & "
            f"{row['train_rank']:.2f} & {row['test_rank']:.2f} & "
            f"{row['train_best_rate']:.1f} & {row['test_best_rate']:.1f} \\\\"
        )
        latex_tables.append(latex_row)
    
    # Close rankings table and start scores table
    latex_tables.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Ranking performance of Smoothie variants. \\textit{Avg Rank} shows mean ranking (1 is best), "
        "\\textit{R1} shows percentage of rank-1 selections. A random selection strategy would achieve a rank of 3, and select the optimal model 20\% of the time.}",
        "\\label{tab:smoothie-selection-rankings-nlg}",
        "\\end{minipage}%",
        "\\hfill",
        "\\begin{minipage}[t]{0.48\\textwidth}",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{l@{\\hspace{1em}}cc@{\\hspace{1em}}cc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{Smoothie}} & \\multicolumn{2}{c}{\\textbf{Reference}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "\\textbf{Dataset} & Train & Test & Random & Oracle \\\\",
        "\\midrule"
    ])
    
    # Add data rows for scores table
    for row in data:
        latex_row = (
            f"{row['dataset']} & "
            f"{row['train_score']:.1f} & {row['test_score']:.1f} & "
            f"{row['random_score']:.1f} & {row['oracle_score']:.1f} \\\\"
        )
        latex_tables.append(latex_row)
    
    # Close scores table
    latex_tables.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Score performance (\\%) of Smoothie variants compared to random selection (\\textit{Random}) "
        "and optimal model selection (\\textit{Oracle}).}",
        "\\label{tab:smoothie-selection-scores-nlg}",
        "\\end{minipage}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_tables)


def save_smoothie_selection_ranking_tables(datasets: List[str], smoothie_train: str, smoothie_test: str, output_path: str):
    """
    Generate and save the LaTeX tables to a file.
    """
    latex_tables = generate_smoothie_selection_ranking_tables(datasets, smoothie_train, smoothie_test)
    with open(output_path, 'w') as f:
        f.write(latex_tables)


def generate_smoothie_routing_tables(datasets: List[str], smoothie_train: str, smoothie_test: str, output_path: str = "routing_results_table.tex") -> str:
    """
    Generates Smoothie routing performance and creates a LaTeX table of results
    """
    np.random.seed(0)
    
    # Dictionary to store results for each dataset
    results_by_dataset = {}
    
    for dataset in datasets:
        dataset_results = []
        for model_group in MODEL_GROUPS:
            scores_fpath = RESULTS_DIR / dataset / model_group / "scores.json"
            scores = json.loads(scores_fpath.read_text())[TASK2METRIC[dataset]]
            ensemble_scores = scores["ensemble"]
            
            # Skip if not exactly 5 models
            if len(ensemble_scores) != 5:
                continue
                
            smoothie_train_score = scores["smoothie"][smoothie_train]
            smoothie_test_score = scores["smoothie"][smoothie_test]
            ensemble_scores_list = list(ensemble_scores.values())
            best_individual_score = max(ensemble_scores_list)
            labeled_oracle_score = scores["labeled_oracle"]
            random_score = scores["pick_random"]
            pair_rm_score = scores["pair_rm"]
            
            dataset_results.append({
                "train_score": smoothie_train_score * 100,
                "test_score": smoothie_test_score * 100,
                "oracle_score": labeled_oracle_score * 100,
                "best_individual_score": best_individual_score * 100,
                "pair_rm_score": pair_rm_score * 100,
                "random_score": random_score * 100,
            })
        
        # Calculate mean across trials
        results_by_dataset[dataset] = {
            metric: np.mean([trial[metric] for trial in dataset_results])
            for metric in dataset_results[0].keys()
        }
    
    # Generate the table
    latex_table = create_latex_table(results_by_dataset, datasets)
    
    # Save the table to file
    save_latex_table(latex_table, output_path)
    
    return latex_table

def create_latex_table(results_by_dataset: Dict, datasets: List[str]) -> str:
    """
    Creates a LaTeX table with column groups and highlighted best scores
    """
    # Define metric groups and their display names
    unlabeled_metrics = ["random_score", "train_score", "test_score"]
    labeled_metrics = ["oracle_score", "best_individual_score", "pair_rm_score"]
    
    metrics_display = {
        "train_score": "Smoothie (Train)",
        "test_score": "Smoothie (Test)",
        "oracle_score": "Oracle",
        "best_individual_score": "Best Ind.",
        "pair_rm_score": "Pair RM",
        "random_score": "Random"
    }
    
    # Start table
    latex_str = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{l|ccc|ccc}",  # Vertical lines to separate groups
        "\\toprule",
        # Column group headers
        "& \\multicolumn{3}{c|}{Unlabeled Methods} & \\multicolumn{3}{c}{Labeled Methods} \\\\",
        "Dataset & " + " & ".join(metrics_display[m] for m in unlabeled_metrics + labeled_metrics) + " \\\\"
        "\\midrule"
    ]
    
    # Add results rows (one per dataset)
    for dataset in datasets:
        display_name = dataset2name[dataset]
        row = [display_name]
        
        # Get scores for this dataset
        scores = results_by_dataset[dataset]
        
        # Find best unlabeled and overall scores
        unlabeled_scores = [scores[m] for m in unlabeled_metrics]
        all_scores = [scores[m] for m in unlabeled_metrics + labeled_metrics]
        best_unlabeled = max(unlabeled_scores)
        best_overall = max(all_scores)
        
        # Add scores with appropriate highlighting
        for metric in unlabeled_metrics + labeled_metrics:
            score = scores[metric]
            cell = f"{score:.1f}"
            
            # Add underlining for best unlabeled method
            if metric in unlabeled_metrics and score == best_unlabeled:
                cell = f"\\underline{{{cell}}}"
            
            # Add asterisk for best overall method
            if score == best_overall:
                cell = f"{cell}*"
            
            row.append(cell)
        
        latex_str.append(" & ".join(row) + " \\\\")
    
    # Close table
    latex_str.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Performance comparison across different routing methods. " +
        "All scores are percentages. " +
        "Best unlabeled method per row is underlined. " +
        "Best overall method per row is marked with *.}",
        "\\label{tab:routing_results}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_str)

def save_latex_table(latex_table: str, output_path: str) -> None:
    """
    Saves the LaTeX table to a file
    """
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # Save the table
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to: {output_path}")


plot_smoothie_selection_ranks(
    datasets=SINGLE_TASK_DATASETS,
    smoothie_train="smoothie_sample_independent_train_time_all-mpnet-base-v2",
    smoothie_test="smoothie_sample_independent_test_time_all-mpnet-base-v2",
    title="Average Rank of Selected Model",
    save_path="analysis/smoothie_selection_ranks_nlg_tasks.png"
)
save_smoothie_selection_ranking_tables(
    datasets=SINGLE_TASK_DATASETS,
    smoothie_train="smoothie_sample_independent_train_time_all-mpnet-base-v2",
    smoothie_test="smoothie_sample_independent_test_time_all-mpnet-base-v2",
    output_path="analysis/smoothie_selection_ranks_nlg_tasks.tex"
)
generate_smoothie_routing_tables(
    datasets=SINGLE_TASK_DATASETS,
    smoothie_train="smoothie_sample_dependent_train_time_50_bge-small-en-v1.5",
    smoothie_test="smoothie_sample_dependent_test_time_1_bge-small-en-v1.5",
    output_path="analysis/smoothie_routing_nlg_tasks.tex"
)

