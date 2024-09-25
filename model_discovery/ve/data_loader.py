import os
import transformers
import logging

logging.basicConfig(level=logging.ERROR)

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
import boto3
import gzip

from ..configs.gam_config import GAMConfig
from .. import utils as U

DEFAULT_NUM_PROC_LOAD =  os.cpu_count()*4 # Configure it based on your system, it can significantly speed up the download of datasets
DEFAULT_NUM_PROC_TOKENIZE =  os.cpu_count()

try:
    hf_key = os.environ.get("HF_KEY",None)
    if hf_key is not None:
        login(hf_key)
    else:
        raise ValueError("HF_KEY is not set, please set it to login to HuggingFace Hub.")
except Exception as e: 
    print("Failed to login to HuggingFace Hub, some datasets may not be available to download: {e}")

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
    for name in dataset_dicts:
        # print(f"Combining dataset: {name}")
        dataset_dict = dataset_dicts[name]
        for key, dataset in dataset_dict.items():
            if key in combined_dict:
                combined_dict[key] = concatenate_datasets([combined_dict[key], dataset])
            else:
                combined_dict[key] = dataset

    # Apply weights by resampling the datasets
    for key in combined_dict:
        datasets = []
        for idx,name in enumerate(dataset_dicts):
            dataset = dataset_dicts[name]
            if key in dataset:
                if weights is None or key not in weights or weights[key][idx] == 1.0:
                    datasets.append(dataset[key])
                else:
                    resampled_dataset = resample_dataset(dataset[key], weights[key][idx])
                    datasets.append(resampled_dataset)
        combined_dict[key] = concatenate_datasets(datasets)
    
    ds = DatasetDict(combined_dict)
    # print('Combined dataset:\n',ds)
    return ds
    
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
                print(f"Loaded tokenized dataset {dataset_name} from {tokenized_dir}")
            except FileNotFoundError:
                ds = dataload_func(tokenizer_name=tokenizer_name, context_length=context_length, *args, **kwargs)
                # ds = ds.shuffle() # no need to shuffle now, hf trainer will shuffle it every epoch, https://discuss.huggingface.co/t/how-to-ensure-the-dataset-is-shuffled-for-each-epoch-using-trainer-and-datasets/4212/7
                if 'train' in ds:
                    remove_columns = ds["train"].column_names
                else: # chunked datasets
                    remove_columns = ds["train_0"].column_names 
                tokenized_datasets = ds.map(
                    lambda x: tokenize(x, tokenizer=tokenizer, context_length=context_length),
                    batched=True, remove_columns=remove_columns, num_proc=DEFAULT_NUM_PROC_TOKENIZE, 
                    batch_size=1000,
                )
                print('Tokenized dataset:\n',tokenized_datasets)
                if 'train' not in tokenized_datasets: # chunked datasets, merge all train_0, train_1, etc. to train
                    K=0
                    while f"train_{K}" in tokenized_datasets:
                        K+=1
                    tokenized_datasets['train'] = concatenate_datasets([tokenized_datasets[f"train_{i}"] for i in range(K)])
                    for i in range(K):
                        del tokenized_datasets[f"train_{i}"]
                    print('Merged Chunked dataset:\n',tokenized_datasets)
                tokenized_datasets.save_to_disk(tokenized_dir)
                print(f"Saved tokenized dataset {dataset_name} to {tokenized_dir}")

            return tokenized_datasets
        return wrapper
    return decorator

@pretokenize_dataset('babylm')
def load_babylm(tokenizer_name, context_length):
    return load_dataset('deven367/babylm-100M', num_proc=DEFAULT_NUM_PROC_LOAD)

@pretokenize_dataset('tinystories')
def load_tinystories(tokenizer_name, context_length):
    data_files = {
        "train": 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt',
        'valid': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'
    }
    return load_dataset("text", data_files=data_files, num_proc=DEFAULT_NUM_PROC_LOAD)

@pretokenize_dataset('wikitext-2')
def load_wikitext2(tokenizer_name, context_length):
    return load_dataset('wikitext','wikitext-2-v1', num_proc=DEFAULT_NUM_PROC_LOAD)


try:
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID",None),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY",None))
    s3 = session.client("s3")
except Exception as e:
    print("Failed to connect to S3, Python-Edu may not be available to download.")

def download_contents_py(blob_id):
    key = f"content/{blob_id}"
    obj = s3.get_object(Bucket="softwareheritage", Key=key)
    with gzip.GzipFile(fileobj=obj['Body']) as fin:
        content = fin.read().decode("utf-8", errors="ignore")
    return {"text": content}

@pretokenize_dataset('python-edu') # directly load from huggingface, pre downloaded texts, may need to remove later
def load_python_edu(tokenizer_name, context_length,splits=['train','eval']): 
    ds = load_dataset("chengjunyan1/smollm-12.5-corpus", "python-edu", num_proc=DEFAULT_NUM_PROC_LOAD)
    ds = DatasetDict({split:ds[split] for split in splits})
    # ds_train = ds['train'].map(download_contents_py, input_columns="blob_id", num_proc=DEFAULT_NUM_PROC_LOAD)
    # ds_test = ds['test'].map(download_contents_py, input_columns="blob_id", num_proc=DEFAULT_NUM_PROC_LOAD)
    # ds_eval = ds['eval'].map(download_contents_py, input_columns="blob_id", num_proc=DEFAULT_NUM_PROC_LOAD)
    # ds = DatasetDict({'train':ds_train, 'test':ds_test, 'eval':ds_eval})
    return ds

@pretokenize_dataset('fineweb-edu-dedup')
def load_fine_web_dedup(tokenizer_name, context_length,splits=['train','eval']):
    ds = load_dataset("chengjunyan1/smollm-12.5-corpus","fineweb-edu-dedup", num_proc=DEFAULT_NUM_PROC_LOAD)
    ds = DatasetDict({split:ds[split] for split in splits})
    return ds

@pretokenize_dataset('cosmopedia-v2')
def load_cosmopedia_v2(tokenizer_name, context_length,splits=['train','eval']):
    ds = load_dataset("chengjunyan1/smollm-12.5-corpus", "cosmopedia-v2", num_proc=DEFAULT_NUM_PROC_LOAD)
    
    def format_text(examples):
        return {
            'text': [
                f"Audience: {audience}\n\nPrompt: {prompt}\n\nResponse: {text}"
                for audience, prompt, text in zip(examples['audience'], examples['prompt'], examples['text'])
            ]
        }

    ds = DatasetDict({split:ds[split] for split in splits})
    for split in splits:
        ds[split] = ds[split].map(
            format_text,
            batched=True,
            batch_size=1000,
            num_proc=DEFAULT_NUM_PROC_LOAD,
            remove_columns=['audience', 'prompt']
        )
    
    # chunk_size = 1_000_000
    # n_chunks = len(ds['train']) // chunk_size + 1
    # for i in range(n_chunks):
    #     ds[f'train_{i}'] = ds['train'].select(range(i*chunk_size, min((i+1)*chunk_size, len(ds['train']))))
    # del ds['train']
    
    # print('Chunked Cosmopedia-v2 dataset:\n', ds)
    return ds

@pretokenize_dataset('open-web-math')
def load_open_web_math(tokenizer_name, context_length,splits=['train','eval']):
    ds = load_dataset("chengjunyan1/smollm-12.5-corpus","open-web-math", num_proc=DEFAULT_NUM_PROC_LOAD)
    ds = DatasetDict({split:ds[split] for split in splits})
    return ds


@pretokenize_dataset('deepmind-math-small')
def load_deepmind_math_small(tokenizer_name, context_length,splits=['train','eval']):
    ds = load_dataset("chengjunyan1/smollm-12.5-corpus", "deepmind-math-small", num_proc=DEFAULT_NUM_PROC_LOAD)
    
    def format_text(examples):
        return {
            'text': [
                f"Question: {question}\n\nAnswer: {answer}"
                for question, answer in zip(examples['question'], examples['answer'])
            ]
        }
    
    ds = DatasetDict({split:ds[split] for split in splits})
    for split in splits:
        ds[split] = ds[split].map(
            format_text,
            batched=True,
            batch_size=1000,
            num_proc=DEFAULT_NUM_PROC_LOAD,
            remove_columns=['question', 'answer']
        )

    return ds


@pretokenize_dataset('stackoverflow-clean')
def load_stackoverflow_clean(tokenizer_name, context_length,splits=['train','eval']):
    ds = load_dataset("chengjunyan1/smollm-12.5-corpus","stackoverflow-clean", num_proc=DEFAULT_NUM_PROC_LOAD)
    ds = ds.rename_column('content', 'text')
    ds = DatasetDict({split:ds[split] for split in splits})
    return ds

loaders={
    'babylm'      :load_babylm,
    'tinystories' :load_tinystories,
    'wikitext2'  :load_wikitext2,
    'python-edu':load_python_edu,
    'fineweb-edu-dedup':load_fine_web_dedup,
    'cosmopedia-v2':load_cosmopedia_v2,
    'open-web-math':load_open_web_math,
    'deepmind-math-small':load_deepmind_math_small,
    'stackoverflow-clean':load_stackoverflow_clean
}

def load_datasets(cfg: GAMConfig): # weights e.g. {'train':[1.5,1.0]} for two datasets
    """Loads the datasets 

    """
    dataset_dicts = {
        dataset: loaders[dataset](
            tokenizer_name=cfg.tokenizer,
            context_length=cfg.context_length
        ) for dataset in cfg.training_data
    } 
    tokenizer=get_tokenizer(cfg.tokenizer)
    dataset=combine_datasets(dataset_dicts, cfg.training_weight)
    # assert 'train' in dataset and 'valid' in dataset, "Dataset must have 'train' and 'valid' keys, and optionally a 'test' key"
    assert 'train' in dataset, "Dataset must have 'train' key"
    return dataset,tokenizer

def load_datasets_args(tokenizer,context_length,training_data,training_weight=None):
    dataset_dicts = {
        dataset: loaders[dataset](
            tokenizer_name=tokenizer,
            context_length=context_length
        ) for dataset in training_data
    }
    tokenizer=get_tokenizer(tokenizer)
    dataset=combine_datasets(dataset_dicts, training_weight)
    # assert 'train' in dataset and 'valid' in dataset, "Dataset must have 'train' and 'valid' keys, and optionally a 'test' key"
    assert 'train' in dataset, "Dataset must have 'train' key"
    return dataset,tokenizer
