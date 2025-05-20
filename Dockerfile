# FROM ghcr.io/allenai/cuda:11.8-cudnn8-dev-ubuntu20.04	
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# [1] Clone the repo, assume its under your home directory ~
ARG GITHUB_TOKEN
RUN git clone https://${GITHUB_TOKEN}@github.com/allenai/model_discovery.git /home/model_discovery

# [2] Create a virtual env with pytorch, move to the repo, and install genesys cli

# conda create -n genesys python=3.12 -y && \
RUN conda create -n genesys python=3.12 -y
# conda activate genesys && \
SHELL ["conda", "run", "-n", "genesys", "/bin/bash", "-c"]
# cd ~/model_discovery && \
WORKDIR /home/model_discovery
# conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
RUN conda install pytorch==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install -e .
RUN pip install -e .

# [3] Setup LLM API keys
COPY secrets/set_secrets.sh /home/model_discovery/secrets/set_secrets.sh
RUN /bin/bash /home/model_discovery/secrets/set_secrets.sh

# [4] Setup a firebase backend, and store the secret json in DB_KEY_PATH, this is required for the distributed search
RUN mkdir -p /home/model_discovery/secrets
ENV DB_KEY_PATH=/home/model_discovery/secrets/db_key.json
COPY secrets/db_key.json ${DB_KEY_PATH}
# its for more gurantees
COPY model_discovery/secrets.py /home/model_discovery/model_discovery/secrets.py

# [5] Copy ckpt data
RUN mkdir -p /home/model_discovery/ckpt
COPY ckpt/ /home/model_discovery/ckpt/

# [6] Setup, notice that you may need to install exec_utils manually before it, if its not public yet
RUN pip install git+https://${GITHUB_TOKEN}@github.com/allenai/exec_utils.git
# skip data prep, mount from beaker
RUN bash scripts/setup_requirements.sh
# Install optional dependencies
RUN pip install -r requirements_optional.txt


# [7] Deploy the GUI
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# [8] Set entrypoint: 1. activate genesys, 2. run the gui by genesys gui
ENTRYPOINT ["conda", "run", "-n", "genesys", "/bin/bash", "-c"]
CMD ["genesys", "gui"]

# Usages
# docker build -t genesys-demo .
# docker run -p 8501:8501 genesys-demo 

