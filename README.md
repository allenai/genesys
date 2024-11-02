# Genesys

Utils and code for model discovery experiments. 


# How to use Genesys CLI/GUI


1. Set up  first (see below)
2. Run `genesys <command> [args]` to use the CLI
3. Run `genesys gui` to start the GUI


# Setting up 

1. Clone the repo (clone with token for internal) and cd into it
2. Make the virtual env and install the requirements 

```shell
conda create -n modis python=3.12
conda activate modis
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
genesys setup
```
Hint: use ```bash scripts/setup_env.sh -d``` to prepare datasets only

Note: You need to install exec_utils in some way; You may need to manually install some dependencies if they are missing during runtime, then go back to install requirements.txt, the requirements.txt is self-conflict a little bit. 

3. Export the environment variables and API keys

```shell
export MY_OPENAI_KEY=YOURKEY
export TOGETHER_API_KEY=YOURKEY
export ANTHROPIC_API_KEY=YOURKEY
export HF_KEY=YOURKEY
export HF_HUB_KEY=YOURKEY
export GITHUB_TOKEN=YOURKEY
export WANDB_API_KEY=YOURKEY
export AWS_SECRET_ACCESS_KEY=YOURKEY
export AWS_ACCESS_KEY_ID=YOURKEY
export S2_API_KEY=YOURKEY
export DATA_DIR=~/model_discovery/data # assume you are using the home dir 
export CKPT_DIR=~/model_discovery/ckpt # assume you are using the home dir 
export DB_KEY_PATH=~/model_discovery/secrets/db_key.json # provide yours
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PINECONE_API_KEY=YOURKEY
export COHERE_API_KEY=YOURKEY
export PERPLEXITY_API_KEY=YOURKEY
export MATHPIX_API_ID=YOURKEY # optional
```

4. Test with the gui
```
genesys gui
```


# Hints

1. Better separate the design nodes and verification nodes, design checkers need to use GPUs, so may cause conflicts. It is recommended to deploy few design nodes and many verification nodes as design nodes are mostly bounded by CPU and API rate limits. 


<!-- 
### Build search library

Download `library_files.zip`[] , unzip it and put it under `model/library`. It should be like this: 
```
model/
    library/
        files/
            htmls/
            htmls2/
            htmlsp/
            pdfs/
            pdfs2/
            pdfsp/
```
 -->

### eval 

Install the custmoized lm_eval: https://github.com/chengjunyan1/lm-evaluation-harness/tree/main

You must export DATA_DIR first, then download evaluation data in DATA_DIR, e.g.:
```
{DATA_DIR}/blimp_filtered/adjunct_island.jsonl
```
The download link for babyLM evaluation data: https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/66358ec34664da20a0ed6acc/?zip=evaluation_data 

Notice that everytime you change your DATA_DIR, you may need to reinstall it, and remember DO NOT INSTALL peft

Supported tasks: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks, specially, babyLM tasks are: "blimp_filtered","blimp_supplement"

### create beaker image (ai2 internal) 
You can run 
```
sh create_beaker.sh 
```
to create a beaker image that allows you to run beaker batch jobs. You
can run a batch job by doing the following: 
```bash 
beaker experiment create etc/beaker/train_example.yaml

```
which shows how to use the built image in beaker to run an example
training job. 

# Current discovery system 

To build a discovery system, you can do the following: 
```python
from model_discovery import BuildSystem 


system = BuildSystem() 
system("discovery me a new model") 
```
The implementation is in `model_discovery/system.py`, which loads a `designer` and `reviewer` agent (by default) from the agent specification files in `etc/agent_spec` (this can be modified as needed and additional agents can be added). 




# Model Discovery Algorithm

Assumptions:
1. Cost of verification >> sampling where in the sampling process, the cost of implementation > proposal
2. The verification process can asymptotically reflect the distance to the optimum or alternatively the improvements
3. High-quality samples (designs chosen to verify) can increase the convergence of the evolution process
4. LLM thinking depth correlated to total output lengths in a dialog for producing one sample
5. LLM agent can asymptotically produce a high-quality sample after a dialog with probability p
...
