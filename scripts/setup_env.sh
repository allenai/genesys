#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --prepare-data-only, -d    Only prepare the pre-training datasets"
    echo "  -h, --help                   Show this help message"
}

# Parse arguments
PREPARE_DATA_ONLY=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prepare-data-only|-d) PREPARE_DATA_ONLY=true ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

if [ "$PREPARE_DATA_ONLY" = false ]; then
    # Set environment variables, export by yourself
    # export MY_OPENAI_KEY=YOUR-KEY
    # export HF_KEY=YOUR-KEY
    # export GITHUB_TOKEN=YOUR-KEY
    # export WANDB_API_KEY=YOUR-KEY
    # export AWS_ACCESS_KEY_ID=YOUR-KEY
    # export AWS_SECRET_ACCESS_KEY=YOUR-KEY

    # Create directories
    mkdir -p ~/model_discovery/data
    mkdir -p ~/model_discovery/ckpt

    # Set directory environment variables, feel free to change these to your desired directories
    export DATA_DIR=~/model_discovery/data
    export CKPT_DIR=~/model_discovery/ckpt

    export HF_DATASETS_TRUST_REMOTE_CODE=1

    # OPTIONAL: append these exports to your .bashrc or .bash_profile for them to be set globally
    echo "export MY_OPENAI_KEY=$MY_OPENAI_KEY" >> ~/.bashrc
    echo "export HF_KEY=$HF_KEY" >> ~/.bashrc
    echo "export GITHUB_TOKEN=$GITHUB_TOKEN" >> ~/.bashrc
    echo "export WANDB_API_KEY=$WANDB_API" >> ~/.bashrc
    echo "export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" >> ~/.bashrc
    echo "export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> ~/.bashrc
    echo "export DATA_DIR=~/model_discovery/data" >> ~/.bashrc
    echo "export CKPT_DIR=~/model_discovery/ckpt" >> ~/.bashrc
    source ~/.bashrc

    # Installing the LATEST customized LM evaluation harness
    echo "Preparing the environment"
    # pip uninstall lm_eval # uninstall current installation first
    pip install -r requirements.txt

    # Uninstalling peft
    # pip uninstall peft

fi

# Download BabyLM evaluation data
echo "Preparing the BabyLM evaluation data"
if [ -d "$DATA_DIR/blimp_filtered" ] && [ -d "$DATA_DIR/supplement_filtered" ]; then
    echo "BabyLM Dataset already downloaded"
else
    cd $DATA_DIR
    wget https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/66358ec34664da20a0ed6acc/?zip=evaluation_data 
    unzip 'index.html?zip=evaluation_data'
    rm 'index.html?zip=evaluation_data'
fi

# Prepare Datasets
echo "Preparing the pre-training datasets"
python -c "
import sys
sys.path.append('..')
from model_discovery.ve.data_loader import load_datasets
from model_discovery.configs.gam_config import GAMConfig_14M

config = GAMConfig_14M() # dataset setting should be the same across all model scales, so just use the 14M setting to initialize the datasets
load_datasets(config)
"
