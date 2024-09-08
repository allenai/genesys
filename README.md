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



Message(id='msg_01Nvch8XBwzwrrMBETHpch49', content=[TextBlock(text="Thank you for providing the seed design and references. I'll now propose a design improvement for the GPT2 model, focusing on modifying one of its GAUs.", type='text'), ToolUseBlock(id='toolu_01PpbFTQ6NDwJJCfj1cRDvFS', input={'selection': 'RotaryPositionalEmbeddings', 'modelname': 'GPT2-RoPE', 'proposal': "\n# GPT2 with Enhanced Rotary Position Embeddings (GPT2-RoPE)\n\n## Motivation\n\nThe current GPT2 model uses a basic implementation of Rotary Positional Embeddings (RoPE). While this is already an improvement over traditional positional encodings, there's room for enhancement, particularly in handling longer sequences and improving the model's ability to capture relative positions more effectively.\n\n## Problem Analysis\n\n1. **Limited sequence length handling**: The current implementation may not scale well to very long sequences, which is increasingly important in modern language models.\n2. **Lack of flexibility**: The existing RoPE implementation doesn't allow for easy adjustment of its properties, such as the rate of position information decay.\n3. **Potential for improved performance**: While RoPE is effective, recent research suggests that more sophisticated implementations can lead to better model performance, especially in tasks requiring understanding of long-range dependencies.\n\n## Core Idea and Philosophy\n\nWe propose to enhance the RotaryPositionalEmbeddings GAU by incorporating ideas from the RoFormer paper and the more recent developments in rotary embeddings. The core philosophy is to create a more flexible and powerful positional embedding system that can:\n\n1. Handle longer sequences more effectively\n2. Allow for fine-tuning of the position encoding properties\n3. Potentially improve the model's performance on tasks requiring long-range understanding\n\n## Design Plan\n\nWe will modify the RotaryPositionalEmbeddings GAU as follows:\n\n1. **Implement a more sophisticated RoPE class**: Based on the RoFormer paper and recent developments, we'll create a new RotaryEmbedding class that offers more flexibility and features.\n\n2. **Add support for learned frequencies**: This will allow the model to potentially learn better position representations for specific tasks or domains.\n\n3. **Implement interpolation factor**: This will enable the model to handle sequences longer than those seen during training more effectively.\n\n4. **Add support for XPos (Extrapolatable Position Embeddings)**: This will further enhance the model's ability to handle longer sequences and improve its understanding of relative positions.\n\n5. **Implement caching mechanism**: To improve efficiency, especially for autoregressive generation tasks.\n\n6. **Add support for different frequency types**: This will allow the embeddings to be optimized for different types of data (e.g., language, images).\n\nHere's a high-level overview of the proposed changes:\n\n```python\nclass RotaryEmbedding(nn.Module):\n    def __init__(\n        self,\n        dim,\n        custom_freqs = None,\n        freqs_for = 'lang',\n        theta = 10000,\n        max_freq = 10,\n        num_freqs = 1,\n        learned_freq = False,\n        use_xpos = False,\n        xpos_scale_base = 512,\n        interpolate_factor = 1.,\n        theta_rescale_factor = 1.,\n        seq_before_head_dim = False,\n        cache_if_possible = True\n    ):\n        super().__init__()\n        # Implementation details...\n\n    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):\n        # Implementation details...\n\n    def rotate_queries_and_keys(self, q, k, seq_dim = None):\n        # Implementation details...\n\n    def get_scale(self, t, seq_len = None, offset = 0):\n        # Implementation details...\n\n    def forward(self, t, seq_len = None, offset = 0):\n        # Implementation details...\n```\n\nThis new implementation will replace the current RotaryPositionalEmbeddings GAU in the GPT2 model.\n\n## Conclusion\n\nBy enhancing the RotaryPositionalEmbeddings GAU with these improvements, we expect to see several benefits:\n\n1. **Improved handling of long sequences**: The model should be able to maintain performance even on sequences longer than those seen during training.\n2. **Better capture of relative positions**: The enhanced RoPE implementation should allow the model to better understand and utilize relative position information.\n3. **Increased flexibility**: The new implementation allows for fine-tuning of various aspects of the positional embeddings, which can be optimized for specific tasks or domains.\n4. **Potential performance improvements**: These enhancements may lead to improved performance on tasks requiring long-range understanding and more nuanced use of positional information.\n\nThis modification maintains the overall structure of the GPT2 model while significantly enhancing its ability to handle and utilize positional information, potentially leading to improved performance across a wide range of natural language processing tasks.\n\n## References\n\n1. Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv preprint arXiv:2104.09864.\n2. Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. arXiv preprint arXiv:2401.04088.\n"}, name='GUE_DESIGN_PROPOSAL_format', type='tool_use')], model='claude-3-5-sonnet-20240620', role='assistant', stop_reason='tool_use', stop_sequence=None, type='message', usage=Usage(input_tokens=16630, output_tokens=1325))