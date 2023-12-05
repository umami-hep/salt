# base image
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# local and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /salt

# add some packages
RUN apt-get update && apt-get install -y git h5utils wget vim build-essential jq

# update python pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

# install dependencies (make use of caching)
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# add some other packages to the image, instead of as a package dependency
RUN python -m pip install puma-hep umami-preprocessing

# copy and install package
COPY . .
RUN python -m pip install -e .

