# TODO: Remove unused constants
HF_CACHE_DIR = "../cache"

# This dictionary contains the name of each test dataset, the subset to use, and the split to use.
HF_TEST_DATASETS = {
    "cnn_dailymail": ("cnn_dailymail", "3.0.0", "test"),
    "xsum": ("EdinburghNLP/xsum", None, "test"),
    "e2e_nlg": ("e2e_nlg", None, "test"),
    "web_nlg": ("web_nlg", "release_v3.0_en", "dev"),
    "squad": ("hazyresearch/based-squad", None, "validation"),
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc", "validation"),
    "definition_extraction": ("nguha/legalbench", "definition_extraction", "test"),
    "mix_instruct": ("llm-blender/mix-instruct", None, "test"),
    "gsm8k": ("gsm8k", "main", "test"),
}

# This dictionary contains the name of each train dataset, the subset to use, and the split to use. This is only used for selecting in-context demonstrations for prompts.
HF_TRAIN_DATASETS = {
    "cnn_dailymail": ("cnn_dailymail", "3.0.0", "train"),
    "e2e_nlg": ("e2e_nlg", None, "train"),
    "web_nlg": ("web_nlg", "release_v3.0_en", "train"),
    "xsum": ("EdinburghNLP/xsum", None, "train"),
    "common_gen": ("allenai/common_gen", None, "train"),
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc", "train[:10%]"),
    "mix_instruct": ("llm-blender/mix-instruct", None, "train"),
    "gsm8k": ("gsm8k", "main", "train"),
}

# HF URLS for each model
# TODO: Remove unused models
HF_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "falcon-1b": "tiiuae/falcon-rw-1b",
    "snorkel-7b": "snorkelai/Snorkel-Mistral-PairRM-DPO",
    "phi-2": "microsoft/phi-2",
    "dolly-3b": "databricks/dolly-v2-3b",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "llema-7b": "EleutherAI/llemma_7b",
    "flan-t5-xl": "google/flan-t5-xl",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "codellama-7b": "meta-llama/CodeLlama-7b-hf",
    "codellama-7b-py": "meta-llama/CodeLlama-7b-Python-hf",
    "codellama-7b-instruct": "meta-llama/CodeLlama-7b-Instruct-hf",
    "nous-hermes-llama-2-7b": "NousResearch/Nous-Hermes-llama-2-7b",
    "together-llama-2-7b": "togethercomputer/LLaMA-2-7B-32K",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "storm-7b": "jieliu/Storm-7B",
    "gemma-7b": "google/gemma-7b",
    "qwen-7b": "Qwen/Qwen-7B",
    "gemma-2b": "google/gemma-2b-it",
    "incite-3b": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    "nous-capybara": "NousResearch/Nous-Capybara-7B-V1.9",
    "qwen-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "flan-t5-large": "google/flan-t5-large"
}

# MODEL MAX LENGTHS
# TODO: Remove unused models
HF_MODEL_MAX_LENGTHS = {
    "pythia-1b": 2048,
    "pythia-2.8b": 2048,
    "pythia-6.9b": 2048,
    "pythia-410m": 2048,
    "falcon-1b": 1024,
    "snorkel-7b": 4096,
    "mistral-7b": 32000,
    "dolly-3b": 4096,
    "llema-7b": 16000,
    "phi-2": 2048,
    "flan-t5-xl": 1024,
    "phi-3": 4000,
    "codellama-7b": 4096,
    "codellama-7b-py": 4096,
    "codellama-7b-instruct": 4096,
    "nous-hermes-llama-2-7b": 4096,
    "together-llama-2-7b": 4096,
    "llama-2-7b": 4096,
    "vicuna-7b": 4096,
    "storm-7b": 4096,
    "gemma-7b": 4096,
    "qwen-7b": 4096,
    "gemma-2b": 4096,
    "incite-3b": 4096,
    "nous-capybara": 4096,
    "qwen-1.5b": 4096,
    "flan-t5-large": 4096
}
