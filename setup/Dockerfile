FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# ensure locale is set during build
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

# add some packages
RUN apt-get update
RUN apt-get install -y git h5utils wget curl vim

# check and update python and pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

# install package
COPY salt/ salt/
COPY pyproject.toml .
RUN python -m pip install -e .
