# Docker file for genesys demo

FROM ghcr.io/allenai/cuda:11.8-cudnn8-dev-ubuntu20.04

RUN mkdir -p /root
#  !!! Notice that the ~ in the image is /root, so dont set dir to ~/ !!!
ENV DATA_DIR=/root/genesys/data
ENV CKPT_DIR=/root/genesys/ckpt
RUN mkdir -p ${DATA_DIR} ${CKPT_DIR}

WORKDIR /root/genesys

RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

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
# CHANGED FOR SKIFF2 / CLOUD RUN:
# Cloud Run injects the port to listen on via $PORT (default 8080). Streamlit
# defaults to 8501 and ignores $PORT, so we pass it explicitly. The ${PORT:-8501}
# fallback keeps `docker run`/docker-compose local dev working unchanged.
CMD ["genesys gui --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
