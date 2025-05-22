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



RUN mkdir -p /root
#  !!! Notice that the ~ in the image is /root, so dont set dir to ~/ !!!
ENV DATA_DIR=/root/genesys/data
ENV CKPT_DIR=/root/genesys/ckpt
RUN mkdir -p ${DATA_DIR} ${CKPT_DIR}

WORKDIR /root/genesys



RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu


ARG GITHUB_TOKEN
RUN pip install git+https://${GITHUB_TOKEN}@github.com/allenai/exec_utils.git
RUN pip install paperswithcode-client>=0.3.1 
RUN pip uninstall lm_eval -y 
RUN pip install hf_xet

COPY ./requirements.txt /root/genesys/requirements.txt
RUN pip install -r requirements.txt

COPY ./scripts/demo_data_download.py /root/genesys/scripts/demo_data_download.py
RUN python scripts/demo_data_download.py

COPY . /root/genesys
RUN pip install -e .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["genesys gui"]


###### Usages ######

# Build the image
# docker build -build-arg GITHUB_TOKEN=your_github_token -t genesys-demo .
# docker run -p 8502:8501 genesys-demo 

# Permissions
# gcloud auth login
# gcloud auth configure-docker us-central1-docker.pkg.dev

# Push to GCP
# docker tag genesys-demo us-central1-docker.pkg.dev/model-discovery/genesys/genesys-demo-gcp:latest
# docker push us-central1-docker.pkg.dev/model-discovery/genesys/genesys-demo-gcp:latest


