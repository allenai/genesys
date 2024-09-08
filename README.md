# Model Disovery 

Utils and code for model discovery experiments. 

# Setting up 


We recommend setting up a conda environment as below, remember to edit the `setup.sh` file to include the necessary environment variables before running the setup script.

Environment variables needed: 
```shell
export MY_OPENAI_KEY=YOURKEY
export ANTHROPIC_API_KEY=YOURKEY
export HF_KEY=YOURKEY
export HF_HUB_KEY=YOURKEY
export GITHUB_TOKEN=YOURKEY
export WANDB_API_KEY=YOURKEY
export AWS_SECRET_ACCESS_KEY=YOURKEY # Get it and ID from AWS, Security Credentials of your account
export AWS_ACCESS_KEY_ID=YOURKEY 
export S2_API_KEY=YOURKEY # Optional, it allows higher rate limit for the S2 API
export DATA_DIR=~/model_discovery/data # Change it to your data dir
export CKPT_DIR=~/model_discovery/ckpt # Change it to your checkpoint dir
export HF_DATASETS_TRUST_REMOTE_CODE=1
```
You can also set them in setup.sh then follow the instructions below to set up the environment.

```shell
conda create -n modis python=3.12
conda activate modis
3pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # for cuda 11.8
bash scripts/setup.sh # remember to set up the environment variables in setup.sh
```

Hint: use ```bash scripts/setup.sh -d``` to prepare datasets only

### python dependencies 
Currently, the agent portion relies on a private agent repo [**here**](https://github.com/allenai/exec_utils) (*soon to be made public and renamed*). This can be installed as below (requires github token):
```shell
pip install git+https://{TOKEN}@github.com/allenai/exec_utils
pip install -r requirements.txt # use `requirements_linux.txt` for linux
```

This will install all agent associated requirements. You will also need to incorporate one or more of the following API
keys to access the underlying models: 
```shell

export MY_OPENAI_KEY=XXXXXXXXXXXXX
export TOGETHER_API_KEY=XXXXXXXXXXXXX
export HF_KEY=XXXXXXXXXXXXX
export GITHUB_TOKEN=XXXXXXXXXXXXX
```

To check that the installation works correctly, you can try to the following: 
```python
from exec_utils import BuildModel

llm = BuildModel()
llm("what is your name?")
```

### training data variables 
Other library specific environment variables 
```
export DATA_DIR=/path/to/data/dir
```
to specify where to dump data when running training. 

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

# Agent front-end 

You can run: 
```bash

sh run_demo.sh
```


# Pseudo Code Draft

Designer agent:

```
for i in max_attemp:
    while P<threshold_P:
        # self-refine
        P,new_design ~ designer()
    for j in max_attemp_checker:
        pass = checker(new_design)
        if pass:
            break
        else:
            new_design,P ~ debugger()
    if pass:
        review=reviewer(new_design)
        if review>threshold:
            break
        else:
            new_design ~ designer()
```

GAB Flow Execution:

Transformer
 - MHA
   - RoPE
   - Dot-Product Attn
 - FFN
   - Gated

P=Transformer.MHA

X=Input Tensor
Z={} # intermediate variates
P=root
while True:
    for line in **get_unit**(P).source:
        if is_gab_unit(line): # in this case, its calling a child
            P=line.get_path()
            break
        else:
            X,Z=execute(line)
    P=P.next()
    if not P:
        break

        




#### Notes


1. The reviewer sometimes stuck there, giving the same reviews
2. Fix Format Checker
3. Seeds are important, make some manual seeds
4. Introduce AIS prompts maybe if it helps
5. *Interactive debugging*
   1. Capture the prints during the execution of GAB code
   2. Also maybe prompt the code gen process
6. Experiment of kwargs passing:
   1. e.g _forward define actual kwargs, but forward pass **Z
   2. Will Z still preserve all args?


