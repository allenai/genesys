# Docker file for genesys demo

FROM ghcr.io/allenai/cuda:11.8-cudnn8-dev-ubuntu20.04	
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:11.8.0-base-ubuntu20.04


# FROM ubuntu:20.04

# Optional: Not needed if using ai2 docker image

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     git \
#     wget \
#     && rm -rf /var/lib/apt/lists/*
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
#     && rm Miniconda3-latest-Linux-x86_64.sh 
# # RUN . ~/.bashrc
# ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# [1] Clone the repo, assume its under your home directory ~
ARG GITHUB_TOKEN
RUN git clone https://${GITHUB_TOKEN}@github.com/allenai/model_discovery.git /root/model_discovery

# [2] Create a virtual env with pytorch, move to the repo, and install genesys cli

# conda create -n genesys python=3.12 -y && \
# RUN conda create -n genesys python=3.12 -y
# SHELL ["conda", "run", "-n", "genesys", "/bin/bash", "-c"]
# RUN conda install -c conda-forge ittapi intel-openmp tbb -y
# ~ should be /root/ in the image
WORKDIR /root/model_discovery
# RUN conda install pytorch==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# RUN conda install pytorch==2.4.1 cpuonly -c pytorch -y
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .

# [4] Setup a firebase backend, and store the secret json in DB_KEY_PATH, this is required for the distributed search
ENV DB_KEY_PATH=/root/model_discovery/secrets/db_key.json
RUN mkdir -p /root/model_discovery/secrets
COPY secrets/db_key.json ${DB_KEY_PATH}
# its for more gurantees
COPY model_discovery/secrets.py /root/model_discovery/model_discovery/secrets.py

# [5] Setup, notice that you may need to install exec_utils manually before it, if its not public yet
RUN pip install git+https://${GITHUB_TOKEN}@github.com/allenai/exec_utils.git
# skip data prep, mount from beaker
ENV DATA_DIR=/root/model_discovery/data
ENV CKPT_DIR=/root/model_discovery/ckpt
RUN mkdir -p ${DATA_DIR} ${CKPT_DIR}
# RUN bash scripts/setup_requirements.sh
RUN pip install paperswithcode-client>=0.3.1 
RUN pip uninstall lm_eval -y 
RUN pip install -r requirements.txt
RUN pip install hf_xet

# [6] Copy ckpt data
COPY ckpt/evo_exp_full_a ${CKPT_DIR}/evo_exp_full_a
COPY ckpt/RESULTS ${CKPT_DIR}/RESULTS
COPY ckpt/.setting.json ${CKPT_DIR}/.setting.json
COPY data/ ${DATA_DIR}/

# [7] Deploy the GUI
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# [8] Set entrypoint: 1. activate genesys, 2. run the gui by genesys gui
# ENTRYPOINT ["conda", "run", "-n", "genesys", "/bin/bash", "-c"]
# CMD ["streamlit run bin/app.py --server.port=8501 --server.address=0.0.0.0"]
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["genesys gui"]


# Usages
# docker build --build-arg GITHUB_TOKEN="your_actual_github_pat_here" -t genesys-demo .
# docker run -p 8502:8501 genesys-demo 

# docker tag genesys-demo us-central1-docker.pkg.dev/model-discovery/genesys/genesys-demo-gcp:latest
# docker push us-central1-docker.pkg.dev/model-discovery/genesys/genesys-demo-gcp:latest


