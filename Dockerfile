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
COPY . /root/genesys

# [2] Create a virtual env with pytorch, move to the repo, and install genesys cli

# conda create -n genesys python=3.12 -y && \
# RUN conda create -n genesys python=3.12 -y
# SHELL ["conda", "run", "-n", "genesys", "/bin/bash", "-c"]
# RUN conda install -c conda-forge ittapi intel-openmp tbb -y
# ~ should be /root/ in the image
WORKDIR /root/genesys
# RUN conda install pytorch==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# RUN conda install pytorch==2.4.1 cpuonly -c pytorch -y
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .

# [3] Setup, notice that you may need to install exec_utils manually before it, if its not public yet
RUN pip install git+https://${GITHUB_TOKEN}@github.com/allenai/exec_utils.git
# skip data prep, mount from beaker
ENV DATA_DIR=/root/genesys/data
ENV CKPT_DIR=/root/genesys/ckpt
RUN mkdir -p ${DATA_DIR} ${CKPT_DIR}
# RUN bash scripts/setup_requirements.sh
RUN pip install paperswithcode-client>=0.3.1 
RUN pip uninstall lm_eval -y 
RUN pip install -r requirements.txt
RUN pip install hf_xet

# [4] Copy ckpt data and secrets
COPY model_discovery/secrets.py /root/genesys/model_discovery/secrets.py
# COPY ckpt/ ${CKPT_DIR}/
# COPY data/ ${DATA_DIR}/
RUN python scripts/demo_data_download.py

# [5] Deploy the GUI
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# [6] Set entrypoint: 1. activate genesys, 2. run the gui by genesys gui
# ENTRYPOINT ["conda", "run", "-n", "genesys", "/bin/bash", "-c"]
# CMD ["streamlit run bin/app.py --server.port=8501 --server.address=0.0.0.0"]
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["genesys gui"]


# Usages
# docker build -t genesys-demo .
# docker run -p 8502:8501 genesys-demo 

# Permissions
# gcloud auth login
# gcloud auth configure-docker us-central1-docker.pkg.dev

# Push to GCP
# docker tag genesys-demo us-central1-docker.pkg.dev/model-discovery/genesys/genesys-demo-gcp:latest
# docker push us-central1-docker.pkg.dev/model-discovery/genesys/genesys-demo-gcp:latest


