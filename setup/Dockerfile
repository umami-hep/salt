FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

## ensure locale is set during build
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y git h5utils wget curl

# install requirements
RUN python --version
RUN pip3 --version
COPY pyproject.toml .
RUN mkdir salt
RUN pip3 install --upgrade pip
RUN pip3 install -e .
