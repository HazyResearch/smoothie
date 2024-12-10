"""
This script computes the correlations between Smoothie weights and model rankings on all sets of tasks.
"""

import sys

sys.path.append("../")

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from src.ensembles import MODEL_GROUPS

# CONSTANTS
RESULTS_DIR = Path("smoothie_data/multi_prompt_results")
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
}


def compute_smoothie_latex_table(table_output_path=None):
    """
    Generates a LaTeX table comparing Smoothie results across datasets and model sizes.
    Table is transposed with datasets as columns. Scores are multiplied by 100 for percentage representation.

    Args:
        table_output_path (str, optional): Path where the LaTeX table file should be saved.
                                         If None, prints to console.
    """
    import json
    from pathlib import Path

    import numpy as np

    datasets = [
        "cnn_dailymail",
        "definition_extraction",
        "e2e_nlg",
        "squad",
        "trivia_qa",
        "web_nlg",
        "xsum",
    ]

    models = ["falcon-1b", "llama-2-7b"]
    model2name = {
        "falcon-1b": "Falcon",
        "llama-2-7b": "Llama-2",
    }
    # Methods to evaluate
    smoothie_global = "smoothie_sample_independent_test_time_all-mpnet-base-v2"
    smoothie_local = "smoothie_sample_dependent_test_time_1_all-mpnet-base-v2"

    # First collect all results
    results = {}
    for model in models:
        results[model] = {}
        for dataset in datasets:
            scores_fpath = RESULTS_DIR / dataset / model / "scores.json"
            scores = json.loads(scores_fpath.read_text())[TASK2METRIC[dataset]]

            # Calculate the "random" baseline (average of ensemble scores)
            ensemble_scores = list(scores["ensemble"].values())
            random_score = np.mean(ensemble_scores)
            best_in_ensemble_score = max(ensemble_scores)

            # Get Smoothie scores and oracle score
            smoothie_global_score = scores["smoothie"][smoothie_global]
            smoothie_local_score = scores["smoothie"][smoothie_local]
            best_on_val_score = scores["labeled_oracle"]

            results[model][dataset] = {
                "random": np.round(random_score * 100, 1),
                "global": np.round(smoothie_global_score * 100, 1),
                "local": np.round(smoothie_local_score * 100, 1),
                "oracle": np.round(best_on_val_score * 100, 1),
                "best_in_ensemble": np.round(best_in_ensemble_score * 100, 1),
            }

    for model in models:
        best_model_selections = 0
        for dataset in datasets:
            scores_fpath = RESULTS_DIR / dataset / model / "scores.json"
            scores = json.loads(scores_fpath.read_text())[TASK2METRIC[dataset]]

            # Get Smoothie global score and best in ensemble score
            smoothie_global_score = scores["smoothie"][smoothie_global]
            best_in_ensemble_score = max(scores["ensemble"].values())

            # Check if Smoothie global selects the best model
            if np.isclose(smoothie_global_score, best_in_ensemble_score):
                best_model_selections += 1

        print(
            f"Smoothie-global selects the best model {best_model_selections} times for {model}."
        )

    # Create table column specification
    column_spec = "@{}l" + "c" * (len(datasets) + 1) + "@{}"

    # Create LaTeX table
    latex_content = [
        "\\begin{table}[t]",
        "\\centering",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{7pt}",
        f"\\begin{{tabular}}{{{column_spec}}}",
    ]

    # Header rows
    latex_content.extend(
        [
            "\\toprule",
            "& & "
            + " & ".join(
                [f"\\small{{{dataset2name.get(d, d.title())}}}" for d in datasets]
            )
            + " \\\\ \\midrule",
        ]
    )

    # For each model and metric type
    for model in models:
        # Add multi-row label for model group
        latex_content.append(f"\\multirow{{3}}{{*}}{{\\small{{{model2name[model]}}}}}")

        # Collect scores for each dataset
        all_scores = {
            dataset: {
                metric: results[model][dataset][metric]
                for metric in ["random", "global", "local", "oracle"]
            }
            for dataset in datasets
        }

        # Determine the best unsupervised and overall methods for each dataset
        best_unsupervised = {
            dataset: [
                m
                for m in ["random", "global"]
                if all_scores[dataset][m]
                == max(all_scores[dataset][m] for m in ["random", "global"])
            ]
            for dataset in datasets
        }
        best_overall = {
            dataset: [
                m
                for m in all_scores[dataset]
                if all_scores[dataset][m] == max(all_scores[dataset].values())
            ]
            for dataset in datasets
        }

        # Add each metric row
        for metric, metric_name in [
            ("random", "\\small{\\random}"),
            ("global", "\\small{\\nameglobal}"),
            ("local", "\\small{\\namelocal}"),
            ("oracle", "\\small{\\textsc{Best-on-Val}}"),
        ]:
            scores = [results[model][dataset][metric] for dataset in datasets]
            score_str = " & ".join(
                [
                    f"\\small{{\\underline{{\\textbf{{{score:.1f}}}}}}}"
                    if metric in ["random", "global"]
                    and metric in best_unsupervised[dataset]
                    and metric in best_overall[dataset]
                    else f"\\small{{\\underline{{{score:.1f}}}}}"
                    if metric in ["random", "global"]
                    and metric in best_unsupervised[dataset]
                    else f"\\small{{\\textbf{{{score:.1f}}}}}"
                    if metric in best_overall[dataset]
                    else f"\\small{{{score:.1f}}}"
                    for score, dataset in zip(scores, datasets)
                ]
            )
            latex_content.append(f"& {metric_name} & {score_str} \\\\")

        # Add midrule after each model, except for the last one
        if model != "llama-2-7b":
            latex_content.append("\\midrule")

    # Table footer
    caption = "Comparing \\nameglobal and \\namelocal to baseline methods in the prompt-selection setting. Underlined values are the best performing \\textit{unsupervised} methods. Bold values are the best performing \\textit{overall} methods. We report rouge2 scores for CNN, XSum, WebNLG, and E2E, and accuracy for the rest. All metrics are scaled to 0-100."
    latex_content.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{" + caption + "}",
            "\\label{tab:smoothie-prompt-comparison}",
            "\\end{table}",
        ]
    )

    # Join all lines with newlines
    latex_table = "\n".join(latex_content)

    if table_output_path:
        # Save to file
        Path(table_output_path).write_text(latex_table)
        print(f"LaTeX table saved to {table_output_path}")
    else:
        # Print to console
        print(latex_table)


compute_smoothie_latex_table(
    table_output_path="tables/smoothie_nlg_prompting_table.tex"
)
