import os
import transformers

from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import (
    load_dataset,
    concatenate_datasets,
    DatasetDict,
    load_from_disk
)
import numpy as np
import functools as ft
from typing import List

from ..model.configs.gam_config import GAMConfig
from ..model.configs.basic import BasicConfig
#from ..model.configs.apikeys import APIKeys
from .. import utils as U

def get_tokenizer(tokenizer_name):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize(
        element,
        tokenizer: transformers.PreTrainedTokenizer,
        context_length: int
    ) -> dict:
    """Tokenizers input and returns their input_ids 

    """
    outputs = tokenizer(
        element["text"],  # need to change accordingly
        padding='max_length',  # Pad all sequences to max_length
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,  # Do not return overflowing tokens
        return_length=True,
    )
    return {"input_ids": outputs["input_ids"]}

def resample_dataset(dataset, weight):
    num_samples = int(len(dataset) * weight)
    indices = np.random.choice(len(dataset), num_samples, replace=True)
    return dataset.select(indices)

def combine_datasets(dataset_dicts, weights:dict=None): # weights e.g. {'train':[1.5,1.0]}
    combined_dict = {}
    
    # Initialize weights if not provided
    for dataset_dict in dataset_dicts:
        for key, dataset in dataset_dict.items():
            if key in combined_dict:
                combined_dict[key] = concatenate_datasets([combined_dict[key], dataset])
            else:
                combined_dict[key] = dataset

    # Apply weights by resampling the datasets
    for key in combined_dict:
        datasets = []
        for idx,dataset in enumerate(dataset_dicts):
            if key in dataset:
                if weights is None or key not in weights or weights[key][idx] == 1.0:
                    datasets.append(dataset[key])
                else:
                    resampled_dataset = resample_dataset(dataset[key], weights[key][idx])
                    datasets.append(resampled_dataset)
        combined_dict[key] = concatenate_datasets(datasets)
    
    return DatasetDict(combined_dict)


def pretokenize_dataset(dataset_name):
    def decorator(dataload_func):
        @ft.wraps(dataload_func)
        def wrapper(tokenizer_name=None, context_length=None, *args, **kwargs):
            
            tokenized_dir = U.pjoin(
                os.environ["DATA_DIR"],
                dataset_name,
                'tokenized',
                tokenizer_name,
                str(context_length)
            )
            tokenizer = get_tokenizer(tokenizer_name)

            try:
                tokenized_datasets = load_from_disk(tokenized_dir)
                print(f"Loaded tokenized dataset {dataset_name} from disk")
            except FileNotFoundError:
                ds = dataload_func(tokenizer_name=tokenizer_name, context_length=context_length, *args, **kwargs)
                ds = ds.shuffle()
                tokenized_datasets = ds.map(
                    lambda x: tokenize(x, tokenizer=tokenizer, context_length=context_length),
                    batched=True, remove_columns=ds["train"].column_names, num_proc=16, batch_size=1000,
                )
                tokenized_datasets.save_to_disk(tokenized_dir)
                print(f"Saved tokenized dataset {dataset_name} to disk")

            return tokenized_datasets
        return wrapper
    return decorator

@pretokenize_dataset('babylm')
def load_babylm(tokenizer_name, context_length):
    return load_dataset('deven367/babylm-100M')

@pretokenize_dataset('tinystories')
def load_tinystories(tokenizer_name, context_length):
    data_files = {
        "train": 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt',
        'valid': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'
    }
    return load_dataset("text", data_files=data_files)

loaders={
    'babylm'      :load_babylm,
    'tinystories' :load_tinystories
}

def load_datasets(cfg: GAMConfig): # weights e.g. {'train':[1.5,1.0]} for two datasets
    """Loads the datasets 

    """
    dataset_dicts = [
        loaders[dataset](
            tokenizer_name=cfg.tokenizer,
            context_length=cfg.context_length
        ) for dataset in cfg.training_data
    ]
    tokenizer=get_tokenizer(cfg.tokenizer)
    return combine_datasets(dataset_dicts, cfg.training_weight),tokenizer
