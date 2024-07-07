# model_discovery 

Utils and code for model discovery experiments. 

# Setting up 

We recommend setting up a conda environment as below
```shell
conda create -n model_discovery python=3.10
conda activate model_discovery 
```

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




*TODO*: add additional requirements for training portion of code. 

# Current discovery system 

To build a discovery system, you can do the following: 
```python
from model_discovery import BuildSystem 


system = BuildSystem() 
system("discovery me a new model") 
```
The implementation is in `model_discovery/system.py`, which loads a `designer` and `reviewer` agent (by default) from the agent specification files in `etc/agent_spec` (this can be modified as needed and additional agents can be added). 

# Agent front-end 

**todo** 
