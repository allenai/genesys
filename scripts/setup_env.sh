# Set environment variables, export by yourself
# export MY_OPENAI_KEY=YOUR-KEY
# export HF_KEY=YOUR-KEY
# export GITHUB_TOKEN=YOUR-KEY
# export WANDB_API_KEY=YOUR-KEY


# Create directories
mkdir -p ~/model_discovery/data
mkdir -p ~/model_discovery/ckpt

# Set directory environment variables, feel free to change these to your desired directories
export DATA_DIR=~/model_discovery/data
export CKPT_DIR=~/model_discovery/ckpt

# OPTIONAL: append these exports to your .bashrc or .bash_profile for them to be set globally
echo "export MY_OPENAI_KEY=$MY_OPENAI_KEY" >> ~/.bashrc
echo "export HF_KEY=$HF_KEY" >> ~/.bashrc
echo "export GITHUB_TOKEN=$GITHUB_TOKEN" >> ~/.bashrc
echo "export WANDB_API_KEY=$WANDB_API" >> ~/.bashrc
echo "export DATA_DIR=~/model_discovery/data" >> ~/.bashrc
echo "export CKPT_DIR=~/model_discovery/ckpt" >> ~/.bashrc
source ~/.bashrc

# Installing the LATEST customized LM evaluation harness
pip uninstall lm_eval # uninstall current installation first
pip install -r requirements_linux.txt

# Uninstalling peft
pip uninstall peft

# Download BabyLM evaluation data
if [ -d "$DATA_DIR/blimp_filtered" ] && [ -d "$DATA_DIR/supplement_filtered" ]; then
    echo "BabyLM Dataset already downloaded"
    exit 0
else
    cd $DATA_DIR
    wget https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/66358ec34664da20a0ed6acc/?zip=evaluation_data 
    unzip 'index.html?zip=evaluation_data'
    rm 'index.html?zip=evaluation_data'
fi