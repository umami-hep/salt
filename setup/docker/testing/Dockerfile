# Using base image from pytorch
FROM ${BASE_IMAGE}

# local and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive

# add some packages
RUN apt-get update && \
    apt-get install -y git h5utils wget vim nano build-essential jq

# update python pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

# install dependencies (make use of caching)
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# add some other packages to the image, instead of as a package dependency
RUN python -m pip install puma-hep umami-preprocessing

# Flash attention sometimes has issues in a requirements file
RUN python -m pip install wheel packaging ninja
RUN python -m pip install flash-attn==2.5.7

# Create folder and copy salt
COPY . /salt
WORKDIR /salt

# Install salt
RUN cd /salt && python -m pip install .
