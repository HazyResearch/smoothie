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


def compute_alpaca_selection_results(plot_output_path=None):
    """
    This function computes selection results for the Alpaca dataset.
    """
    print(f"Alpaca Selection Results " + "#"*40)
    # Load leaderboard 
    results_df = pd.read_csv("smoothie_data/alpaca/leaderboard.csv")
    results_df = results_df.rename(columns={'Unnamed: 0': 'method'})

    # Compute model win rates
    model_win_rates = {}
    model_lc_win_rates = {}
    pick_random_win_rates = [0] * 10 # ith entry is win rate of pick_random_trial_num=i
    pick_random_lc_win_rates = [0] * 10 # ith entry is lc-win rate of pick_random_trial_num=i
    smoothie_win_rates = [0] * 10 # ith entry is win rate of smoothie_independent_all-mpnet-base-v2_trial_num=i
    smoothie_lc_win_rates = [0] * 10 # ith entry is lc-win rate of smoothie_independent_all-mpnet-base-v2_trial_num=i
    for idx in range(len(results_df)):
        method = results_df.iloc[idx]["method"]
        if method in ALPACA_MODELS:
            model_win_rates[method] = results_df.iloc[idx]["win_rate"]
            model_lc_win_rates[method] = results_df.iloc[idx]["length_controlled_winrate"]
        elif method.startswith("pick_random_trial_num="):   
            trial_num = int(method.split("=")[1])
            pick_random_win_rates[trial_num] = results_df.iloc[idx]["win_rate"]
            pick_random_lc_win_rates[trial_num] = results_df.iloc[idx]["length_controlled_winrate"]
        elif method.startswith("smoothie_independent_all-mpnet-base-v2_"):
            trial_num = int(method.split("_trial_num=")[1])
            smoothie_win_rates[trial_num] = results_df.iloc[idx]["win_rate"]
            smoothie_lc_win_rates[trial_num] = results_df.iloc[idx]["length_controlled_winrate"]


    # Calculate differences
    win_rate_diffs = [smoothie_win_rates[i] - pick_random_win_rates[i] for i in range(10)]
    lc_win_rate_diffs = [smoothie_lc_win_rates[i] - pick_random_lc_win_rates[i] for i in range(10)]

    # Calculate average differences
    avg_win_rate_diff = sum(win_rate_diffs) / len(win_rate_diffs)
    avg_lc_win_rate_diff = sum(lc_win_rate_diffs) / len(lc_win_rate_diffs)

    # Find the largest differences
    largest_win_rate_diff = max(win_rate_diffs)
    largest_lc_win_rate_diff = max(lc_win_rate_diffs)

    # Count the number of trials with differences
    num_trials_with_diff = sum(1 for diff in win_rate_diffs if diff > 0)
    num_trials_with_lc_diff = sum(1 for diff in lc_win_rate_diffs if diff > 0)

    # Print the results
    print(f"Average win-rate difference: {avg_win_rate_diff:.2f}")
    print(f"Largest win-rate difference: {largest_win_rate_diff:.2f}")
    print(f"Number of trials with win-rate difference: {num_trials_with_diff}")

    print(f"Average lc win-rate difference: {avg_lc_win_rate_diff:.2f}")
    print(f"Largest lc win-rate difference: {largest_lc_win_rate_diff:.2f}")
    print(f"Number of trials with lc win-rate difference: {num_trials_with_lc_diff}")

    # Create box and whisker plot
    create_box_and_whisker_plot(
        data=[win_rate_diffs],
        labels=[""],
        title="Smoothie vs\nRandom on Alpaca",
        ylabel="Win Rate\nImprovement",
        plot_output_path=plot_output_path.replace(".png", "_win_rate_diff.png") if plot_output_path else None,
        yticks=[-10, 0, 10, 20, 30]
    )

    create_box_and_whisker_plot(
        data=[lc_win_rate_diffs],
        labels=[""],
        title="Smoothie vs\nRandom on Alpaca",
        ylabel="LC Win Rate\nImprovement",
        plot_output_path=plot_output_path.replace(".png", "_lc_win_rate_diff.png") if plot_output_path else None,
        yticks=[-10, 0, 10, 20, 30]
    )


    # Iterate through each trial and compute the following:
    #   - Whether Smoothie selects the best model by win-rate
    #   - Whether Smoothie selects the best model by lc-win-rate
    #   - The correlation coefficient between the weights smoothie assigns and the relative model win-rates
    selected_best_by_win_rate = 0
    selected_best_by_lc_win_rate = 0
    correlations = []
    for i in range(10): 

        # Load Smoothie results
        fpath = Path(f"smoothie_data/alpaca/algorithm_outputs/smoothie_independent_all-mpnet-base-v2_{i}_n_neighbors=1_trial_num={i}.json")
        results = json.loads(fpath.read_text())
        model_selected = results[0]["selected_model"]
        ensemble = eval(results[0]["models_in_trial"])
        smoothie_weights = eval(results[0]["smoothie_weights"])
        smoothie_weights = [float(c) for c in smoothie_weights] 

        # Count whether Smoothie selects best model (by win-rate) 
        sorted_models_wr = sorted(ensemble, key=lambda x: model_win_rates[x], reverse=True)
        if sorted_models_wr[0] == model_selected: 
            selected_best_by_win_rate += 1

        # Count if Smoothie selects best model (by lc-win-rate)
        sorted_models_lcwr = sorted(ensemble, key=lambda x: model_lc_win_rates[x], reverse=True)
        if sorted_models_lcwr[0] == model_selected: 
            selected_best_by_lc_win_rate += 1

        # Compute correlation between Smoothie weights and win-rate
        ensemble_win_rates = [model_win_rates[m] for m in ensemble]
        spearman_corr = stats.spearmanr(smoothie_weights, ensemble_win_rates).correlation
        correlations.append(spearman_corr)
        
    print(f"Selected best by win rate: {selected_best_by_win_rate}/10")
    print(f"Selected best by LC win rate: {selected_best_by_lc_win_rate}/10")
    print(f"Average correlation: {np.mean(correlations):.2f}")
    print()

    


def compute_smoothie_nlg_results(plot_output_path=None):    
    model_groups = ["3b_ensemble", "7b_ensemble"]
    datasets = dataset2name = [
        "cnn_dailymail",
        "definition_extraction",
        "e2e_nlg",
        "squad",
        "trivia_qa",
        "web_nlg",
        "xsum",
    ]
    # Methods to evaluate 
    smoothie_test = "smoothie_sample_independent_test_time_all-mpnet-base-v2"

    # Track correlation coefficients between Smoothie weights over models and actual model rankings for each task
    smoothie_test_correlations = []
    for model_group in model_groups:
        test_time_picks_best = 0
        for dataset in datasets:
            scores_fpath = RESULTS_DIR / dataset / model_group / "scores.json"
            scores = json.loads(scores_fpath.read_text())[TASK2METRIC[dataset]]
            ensemble_scores_list = list(scores["ensemble"].values())
            best_individual_score = max(ensemble_scores_list)
            smoothie_test_time_score = scores["smoothie"][smoothie_test]

            if smoothie_test_time_score == best_individual_score:
                test_time_picks_best += 1

            # Compute model order for correlations
            model_scores = [scores["ensemble"][model] for model in MODEL_GROUPS[model_group]]

            # Compute smoothie test correlations
            generations_fpath = RESULTS_DIR / dataset / model_group / f"{smoothie_test}_test.json"
            smoothie_weights = json.loads(generations_fpath.read_text())["smoothie_weights"][0]
            spearman_corr = stats.spearmanr(smoothie_weights, model_scores).correlation
            smoothie_test_correlations.append(spearman_corr)

        print("NLG Results " + "#"*40)
        print(f"Model Group: {model_group}")
        print(f"Test-Time picks best: {test_time_picks_best}/{len(datasets)}")
        print()
    
    print("Overall NLG Results " + "#"*40)
    print(f"Test-Time correlation: {np.mean(smoothie_test_correlations):.2f}")
    print(f"Number of correlations below 0: {np.sum(np.array(smoothie_test_correlations) < 0)}/{len(smoothie_test_correlations)}")

    # Create box plot
    if plot_output_path:
        create_box_and_whisker_plot(
            data=[smoothie_test_correlations],
            labels=[''],
            title='NLG Smoothie\nCorrelations Distribution',
            ylabel='Correlation Coeff.',
            plot_output_path=plot_output_path,
            yticks=[0, 0.25, 0.5, 0.75, 1],
            ylim=[0, 1.1]
        )


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
            best_on_val_score = scores["labeled_oracle"]
            
            results[model_group][dataset] = {
                "random": np.round(random_score * 100, 1),
                "test": np.round(smoothie_test_score * 100, 1),
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
        all_scores = {dataset: {metric: results[model_key][dataset][metric] for metric in ["random", "test", "oracle"]} for dataset in datasets}
        
        # Determine the best unsupervised and overall methods for each dataset
        best_unsupervised = {dataset: [m for m in ["random", "test"] if all_scores[dataset][m] == max(all_scores[dataset][m] for m in ["random", "test"])] for dataset in datasets}
        best_overall = {dataset: [m for m in all_scores[dataset] if all_scores[dataset][m] == max(all_scores[dataset].values())] for dataset in datasets}
        
        # Add each metric row
        for metric, metric_name in [
            ("random", "\\small{\\textsc{Random}}"), 
            ("test", "\\small{\\nameglobal}"), 
            ("oracle", "\\small{\\textsc{Best-on-Val}}")
        ]:
            scores = [results[model_key][dataset][metric] for dataset in datasets]
            score_str = " & ".join([
                f"\\underline{{\\textbf{{{score:.1f}}}}}" if metric in ["random", "test"] and metric in best_unsupervised[dataset] and metric in best_overall[dataset] else
                f"\\underline{{{score:.1f}}}" if metric in ["random", "test"] and metric in best_unsupervised[dataset] else
                f"\\textbf{{{score:.1f}}}" if metric in best_overall[dataset] else
                f"{score:.1f}"
                for score, dataset in zip(scores, datasets)
            ])
            latex_content.append(f"& {metric_name} & {score_str} \\\\")
        
        # Add midrule after each model group, except for the last one
        if model_group != "7":
            latex_content.append("\\midrule")
    
    # Table footer
    caption = "Comparing \\nameglobal to baseline methods on different ensembles across NLG datasets. Underlined values are the best performing \\textit{unsupervised} methods. Bold values are the best performing \\textit{overall} methods. We report rouge2 scores for CNN, XSum, WebNLG, and E2E, and accuracy for the rest. All metrics are scaled to 0-100."
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{" + caption + "}",
        "\\label{tab:smoothie-global-comparison}",
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


def compute_mix_instruct_correlations():
    """
    This function computes the correlations between Smoothie weights and model rankings on the Mix-Instruct dataset.
    """
    mix_instruct_models = [
        "alpaca-native",
        "chatglm-6b",
        "dolly-v2-12b",
        "flan-t5-xxl",
        "koala-7B-HF",
        "llama-7b-hf-baize-lora-bf16",
        "moss-moon-003-sft",
        "mpt-7b",
        "oasst-sft-4-pythia-12b-epoch-3.5",
        "stablelm-tuned-alpha-7b",
        "vicuna-13b-1.1"
    ]

    # Load generations and get smoothie weights
    smoothie_test = "smoothie_sample_independent_test_time_all-mpnet-base-v2"
    generations_fpath = RESULTS_DIR / "mix_instruct" / "mi_all" / f"{smoothie_test}_test.json"
    generations = json.loads(generations_fpath.read_text())
    smoothie_weights = generations["smoothie_weights"][0]

    # Load scores and get model rankings    
    scores_fpath = RESULTS_DIR / "mix_instruct" / "mi_all" / "scores.json"
    scores = json.loads(scores_fpath.read_text())["mix_instruct_rank"]
    model_scores = [scores["ensemble"][model] for model in mix_instruct_models]

    # Because the model rankings are in descending order, we need to negate the smoothie weights
    smoothie_weights = [-1 * w for w in smoothie_weights]

    # Compute correlation
    spearman_corr = stats.spearmanr(smoothie_weights, model_scores).correlation
    print("#"*40)
    print(f"Mix-Instruct correlation: {spearman_corr:.2f}")
    print(f"Average model score: {np.mean(model_scores):.2f}")


def create_box_and_whisker_plot(data, labels, title, ylabel, plot_output_path=None, ylim=None, yticks=None):
    """
    Creates a box-and-whisker plot for the given data.

    Args:
        data (List[List[float]]): A list of lists, where each inner list represents a distribution.
        labels (List[str]): A list of labels for each distribution.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        plot_output_path (str, optional): Path where the plot should be saved. If None, the plot is shown.
        ylim (tuple, optional): A tuple specifying the y-axis limits (ymin, ymax). If None, the limits are determined automatically.
        yticks (List[float], optional): A list specifying the positions of yticks. If None, the positions are determined automatically.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    boxprops = dict(facecolor=PLOT_COLORS[0], color='black', linewidth=3)
    medianprops = dict(color='black', linewidth=3)
    whiskerprops = dict(color='black', linewidth=3)
    capprops = dict(color='black', linewidth=3)
    flierprops = dict(marker='o', color='black', markersize=5)
    ax.yaxis.grid(True, linestyle='--', linewidth=2.5)
    ax.xaxis.grid(True, linestyle='--', linewidth=2.5)
    ax.boxplot(data, 
               patch_artist=True,   
               boxprops=boxprops,
               medianprops=medianprops,
               whiskerprops=whiskerprops,
               capprops=capprops,
               flierprops=flierprops)
    
    ax.set_title(title, 
                 fontweight=PLOT_TITLE_FONTWEIGHT, 
                 pad=PLOT_TITLE_PAD)
    ax.set_ylabel(ylabel, 
                 fontweight=PLOT_AXIS_LABEL_FONTWEIGHT, 
                 labelpad=PLOT_AXIS_LABEL_PAD)
    ax.set_xticklabels(labels, fontweight='bold')
    
    if ylim:
        ax.set_ylim(ylim)
    
    if yticks:
        ax.set_yticks(yticks)
    
    # Make the edges of the plot thicker
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    
    if plot_output_path:
        fig.savefig(plot_output_path, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
        plt.close(fig)
    else:
        plt.show()

def compute_smoothie_global_ensembles_plot(plot_output_path=None):
    """
    Construct a bar plot of the rank of the model Smoothie-global selects in ensembles.
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
    smoothie_ranks = [0] * 5 # the maximum rank is 5
    for model_group in ["3b_ensemble", "7b_ensemble"]:

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


compute_alpaca_selection_results(plot_output_path="plots/alpaca_selection_results.png")
compute_smoothie_nlg_results(plot_output_path="plots/smoothie_nlg_weight_correlations.png")
compute_smoothie_latex_table(table_output_path="tables/smoothie_nlg_selection_table.tex")
compute_mix_instruct_correlations()
compute_smoothie_global_ensembles_plot(plot_output_path="plots/smoothie_global_nlg_ensembles.png")