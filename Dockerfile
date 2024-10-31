FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04


# # Install base tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    make \
    sudo \
    wget \
    iputils-ping \
    unzip 

# # This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# # puts the right NVIDIA things in the right place (that THOR requires).
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    && echo "32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787 Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" \
        | sha256sum --check \
    && bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENTRYPOINT ["bash", "-l"]


# Clone the GitHub repository to home directory
ARG GITHUB_TOKEN
# Use command substitution with shell form of RUN
RUN GITHUB_TOKEN=$(beaker secret read GITHUB_TOKEN) && \
    git clone https://${GITHUB_TOKEN}@github.com/allenai/model_discovery.git /home/model_discovery
WORKDIR /home/model_discovery


# Setup
RUN conda create -n genesys python=3.12
RUN conda activate genesys
RUN conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

# Setup ENV variables

# ENV DATA_DIR=~/model_discovery/data
# ENV CKPT_DIR=~/model_discovery/ckpt
# ENV GAB_PATH=~/model_discovery/model/gab.py
# ENV DB_KEY_PATH=~/model_discovery/secrets/db_key.json
# ENV HF_DATASETS_TRUST_REMOTE_CODE=1

# # Setup secrets
# RUN mkdir ~/model_discovery/secrets
# RUN touch ${DB_KEY_PATH}
# RUN echo ${DB_KEY_JSON} > ${DB_KEY_PATH}

# Install the package
RUN pip install -e .
RUN genesys setup


# Install optional dependencies
RUN pip install -r requirements_optional.txt




# #### old setups

# WORKDIR /stage/allennlp

# ARG GITHUB

# ### SPECIFIC TO MY SETTING 
# COPY requirements_linux.txt .
# COPY etc/ etc/
# COPY model_discovery/ model_discovery/
# COPY _runs/ _runs/
# COPY data/ data/
# COPY run.sh run.sh

# RUN pip install setuptools==69.5.1
# RUN pip install git+https://$GITHUB@github.com/allenai/exec_utils
# RUN pip install -r requirements_linux.txt
