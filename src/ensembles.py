# This file lists the ensemble combinations for different groups of tasks

# Model groups for GSM8-k
GSM_8K_GROUPS = {"math_group": ["gemma-7b", "llema-7b", "phi-2"]}

# Model groups for MixInstruct. We use the same ensemble as the original MixInstruct paper.
MIX_INSTRUCT_GROUPS = {
    "mi_all": [
        "alpaca-native",
        "chatglm-6b",
        "dolly-v2-12b",
        "flan-t5-xxl",
        "koala-7B-HF",
        "llama-7b-hf-baize-lora-bf16",
        "moss-moon-003-sft",
        "mpt-7b-instruct",
        "oasst-sft-4-pythia-12b-epoch-3.5",
        "stablelm-tuned-alpha-7b",
        "vicuna-13b-1.1",
    ]
}

# Model groups for multi-model experiments on NLG datasets.
MODEL_GROUPS = {
    "3b_ensemble": ["dolly-3b", "incite-3b", "pythia-2.8b", "gemma-2b"],
    "7b_ensemble": [
        "llama-2-7b",
        "mistral-7b",
        "vicuna-7b",
        "gemma-7b",
        "nous-capybara",
    ],
}
MODEL_GROUPS_OLD = {
    "3b_ensemble": ["dolly-3b", "incite-3b", "pythia-2.8b", "gemma-2b"],
    "7b_ensemble": [
        "llama-2-7b",
        "mistral-7b",
        "vicuna-7b",
        "gemma-7b",
        "nous-capybara",
    ],
    "ensemble_1": ["llama-2-7b", "mistral-7b", "pythia-2.8b"],
    "ensemble_2": ["gemma-7b", "llama-2-7b", "nous-capybara"],
    "ensemble_3": ["llama-2-7b", "pythia-2.8b", "vicuna-7b"],
    "ensemble_4": ["dolly-3b", "llama-2-7b", "nous-capybara"],
    "ensemble_5": ["gemma-2b", "incite-3b", "mistral-7b"],
    "ensemble_6": ["dolly-3b", "gemma-7b", "llama-2-7b"],
    "ensemble_7": ["dolly-3b", "mistral-7b", "nous-capybara"],
    "ensemble_8": ["gemma-2b", "gemma-7b", "llama-2-7b"],
    "ensemble_9": ["gemma-2b", "incite-3b", "nous-capybara"],
    "ensemble_10": ["mistral-7b", "pythia-2.8b", "vicuna-7b"],
    "ensemble_11": ["gemma-2b", "llama-2-7b", "pythia-2.8b", "vicuna-7b"],
    "ensemble_12": ["gemma-2b", "gemma-7b", "mistral-7b", "pythia-2.8b"],
    "ensemble_13": ["dolly-3b", "gemma-2b", "llama-2-7b", "vicuna-7b"],
    "ensemble_14": ["gemma-7b", "mistral-7b", "nous-capybara", "pythia-2.8b"],
    "ensemble_15": ["dolly-3b", "gemma-7b", "llama-2-7b", "mistral-7b"],
    "ensemble_16": ["dolly-3b", "gemma-2b", "nous-capybara", "pythia-2.8b"],
    "ensemble_17": ["dolly-3b", "gemma-7b", "mistral-7b", "pythia-2.8b"],
    "ensemble_18": ["gemma-2b", "gemma-7b", "mistral-7b", "vicuna-7b"],
    "ensemble_19": ["gemma-2b", "gemma-7b", "llama-2-7b", "mistral-7b"],
    "ensemble_20": ["gemma-2b", "gemma-7b", "nous-capybara", "pythia-2.8b"],
    "ensemble_21": ["gemma-2b", "incite-3b", "llama-2-7b", "pythia-2.8b", "vicuna-7b"],
    "ensemble_22": [
        "dolly-3b",
        "incite-3b",
        "llama-2-7b",
        "nous-capybara",
        "pythia-2.8b",
    ],
    "ensemble_23": ["dolly-3b", "gemma-7b", "incite-3b", "llama-2-7b", "pythia-2.8b"],
    "ensemble_24": [
        "gemma-7b",
        "mistral-7b",
        "nous-capybara",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_25": ["dolly-3b", "gemma-2b", "gemma-7b", "mistral-7b", "vicuna-7b"],
    "ensemble_26": [
        "dolly-3b",
        "incite-3b",
        "llama-2-7b",
        "nous-capybara",
        "vicuna-7b",
    ],
    "ensemble_27": ["dolly-3b", "gemma-2b", "gemma-7b", "incite-3b", "pythia-2.8b"],
    "ensemble_28": [
        "gemma-2b",
        "incite-3b",
        "llama-2-7b",
        "nous-capybara",
        "vicuna-7b",
    ],
    "ensemble_29": [
        "dolly-3b",
        "gemma-7b",
        "incite-3b",
        "nous-capybara",
        "pythia-2.8b",
    ],
    "ensemble_30": [
        "dolly-3b",
        "gemma-2b",
        "llama-2-7b",
        "nous-capybara",
        "pythia-2.8b",
    ],
    "ensemble_31": [
        "dolly-3b",
        "gemma-2b",
        "incite-3b",
        "llama-2-7b",
        "mistral-7b",
        "pythia-2.8b",
    ],
    "ensemble_32": [
        "gemma-2b",
        "gemma-7b",
        "incite-3b",
        "mistral-7b",
        "nous-capybara",
        "vicuna-7b",
    ],
    "ensemble_33": [
        "dolly-3b",
        "incite-3b",
        "mistral-7b",
        "nous-capybara",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_34": [
        "dolly-3b",
        "gemma-7b",
        "incite-3b",
        "llama-2-7b",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_35": [
        "dolly-3b",
        "gemma-2b",
        "incite-3b",
        "llama-2-7b",
        "mistral-7b",
        "nous-capybara",
    ],
    "ensemble_36": [
        "gemma-2b",
        "gemma-7b",
        "llama-2-7b",
        "mistral-7b",
        "nous-capybara",
        "vicuna-7b",
    ],
    "ensemble_37": [
        "gemma-2b",
        "incite-3b",
        "mistral-7b",
        "nous-capybara",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_38": [
        "gemma-7b",
        "incite-3b",
        "llama-2-7b",
        "mistral-7b",
        "nous-capybara",
        "pythia-2.8b",
    ],
    "ensemble_39": [
        "dolly-3b",
        "gemma-2b",
        "incite-3b",
        "llama-2-7b",
        "nous-capybara",
        "pythia-2.8b",
    ],
    "ensemble_40": [
        "dolly-3b",
        "gemma-7b",
        "incite-3b",
        "nous-capybara",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_41": [
        "dolly-3b",
        "gemma-2b",
        "gemma-7b",
        "incite-3b",
        "llama-2-7b",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_42": [
        "dolly-3b",
        "gemma-2b",
        "incite-3b",
        "llama-2-7b",
        "mistral-7b",
        "nous-capybara",
        "pythia-2.8b",
    ],
    "ensemble_43": [
        "dolly-3b",
        "gemma-2b",
        "gemma-7b",
        "llama-2-7b",
        "mistral-7b",
        "nous-capybara",
        "vicuna-7b",
    ],
    "ensemble_44": [
        "dolly-3b",
        "gemma-2b",
        "gemma-7b",
        "mistral-7b",
        "nous-capybara",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_45": [
        "dolly-3b",
        "gemma-7b",
        "incite-3b",
        "llama-2-7b",
        "nous-capybara",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_46": [
        "dolly-3b",
        "gemma-2b",
        "gemma-7b",
        "incite-3b",
        "llama-2-7b",
        "mistral-7b",
        "pythia-2.8b",
    ],
    "ensemble_47": [
        "gemma-2b",
        "gemma-7b",
        "incite-3b",
        "llama-2-7b",
        "nous-capybara",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_48": [
        "dolly-3b",
        "gemma-2b",
        "gemma-7b",
        "llama-2-7b",
        "mistral-7b",
        "pythia-2.8b",
        "vicuna-7b",
    ],
    "ensemble_49": [
        "dolly-3b",
        "gemma-7b",
        "incite-3b",
        "llama-2-7b",
        "mistral-7b",
        "nous-capybara",
        "vicuna-7b",
    ],
    "ensemble_50": [
        "dolly-3b",
        "gemma-7b",
        "incite-3b",
        "llama-2-7b",
        "mistral-7b",
        "nous-capybara",
        "pythia-2.8b",
    ],
}
