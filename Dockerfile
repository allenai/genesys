FROM ghcr.io/allenai/cuda:11.8-cudnn8-dev-ubuntu20.04	


# # Install base tools.
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     git \
#     make \
#     sudo \
#     wget \
#     iputils-ping \
#     unzip 

# # This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# # puts the right NVIDIA things in the right place (that THOR requires).
# ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
#     && echo "32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787 Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" \
#         | sha256sum --check \
#     && bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/miniconda3 \
#     && rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH

# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


ENTRYPOINT ["conda", "run", "-n", "genesys"]
CMD ["/bin/bash", "-l"]


# Setup ENV variables
RUN mkdir -p /home/data /home/ckpt /home/secrets
ENV DATA_DIR=/home/data
ENV CKPT_DIR=/home/ckpt
ENV DB_KEY_PATH=/home/secrets/db_key.json
COPY secrets/db_key.json ${DB_KEY_PATH}

# write the secret to the path
# ARG FIREBASE_KEY
# RUN echo "${FIREBASE_KEY}" > ${DB_KEY_PATH}

# Setup
RUN conda create -n genesys python=3.12 -y
SHELL ["conda", "run", "-n", "genesys", "/bin/bash", "-c"]
RUN conda install pytorch==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

ARG GITHUB_TOKEN
RUN git clone https://${GITHUB_TOKEN}@github.com/allenai/model_discovery.git /home/model_discovery
WORKDIR /home/model_discovery

# Install the package
RUN pip install -e .
# skip data prep, mount from beaker
RUN genesys setup --skip-data-prep 


# Install optional dependencies
RUN pip install -r requirements_optional.txt



# docker build --build-arg GITHUB_TOKEN=$GITHUB_TOKEN -t genesys-i1 .
# docker run -it --gpus all -it genesys-i1 genesys node -D


# beaker image create --name genesys-i1 genesys-i1



