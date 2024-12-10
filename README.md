# Smoothie

<div align="center" >
    <img src="assets/smoothie_logo.jpg" height=350 alt="Smoothie logo" style="margin-bottom:px"/> 
</div>
<br/>


This repository contains replication code for the following paper:

> Smoothie: Label Free Language Model Routing \
> Neel Guha*, Mayee Chen*, Trevor Chow, Ishan Khare, Christopher Ré \
> NeurIPS 2024 \
> [paper](https://arxiv.org/abs/2412.04692) | [blog](https://hazyresearch.stanford.edu/blog/2024-12-10-smoothie)


## Dependencies

Install the dependencies using the following commands:

```
> conda create -n "smoothie" python=3.10 -y
> conda activate smoothie
> pip install -r requirements.txt
```

## Data and model generations

We store all datasets, predictions, and results from the paper in a [HuggingFace dataset repository](https://huggingface.co/datasets/hazyresearch/smoothie_data/). You can download the dataset from HuggingFace by running the following command:

```bash
> huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
> git clone https://huggingface.co/datasets/hazyresearch/smoothie_data
```


## Using Smoothie

In `tutorials/tutorial.ipynb`, we walk through how to use the Smoothie algorithm. The tutorial can be easily adapted for your use case given that you provide a .jsonl file with the dataset inputs, and several json files each containing a different model/prompt's generations.

If you are interested in the mathematical derivation of Smoothie, check out `tutorials/algorithm.ipynb`.

## Reproducing the paper

See [reproducing_experiments.md](reproducing_experiments.md) for instructions on how to reproduce the experiments in the paper.


## Repository structure

The repository contains the following folders:

- `dataset_configs`: Contains the configuration files for all single-task and multi-task datasets.
- `plots`: Contains plots for the paper.
- `prompt_templates`: Contains the prompt templates for all single-task and multi-task datasets.
- `replication_scripts`: Contains bash scripts for running experiments in the paper.
- `src`: Contains the source code for formatting datasets, getting generations, running routing methods, and evaluating results. The subfolder `paper` contains code for producing the tables and plots in the paper.
- `tables`: Contains latex tables for the paper.
- `tutorials`: Contains tutorials for using Smoothie.


## Citation 

If you use Smoothie in your work, please cite the following paper:

```
@misc{guha2024smoothielabelfreelanguage,
      title={Smoothie: Label Free Language Model Routing}, 
      author={Neel Guha and Mayee F. Chen and Trevor Chow and Ishan S. Khare and Christopher Ré},
      year={2024},
      eprint={2412.04692},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.04692}, 
}
```
