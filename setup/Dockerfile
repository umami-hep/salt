FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

## ensure locale is set during build
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git h5utils wget curl

# install requirements
#COPY setup/requirements.txt .
COPY pyproject.toml .
RUN python --version
RUN pip install -e .
