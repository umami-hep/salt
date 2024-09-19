# Base image created in previous stage
FROM ${BASE_IMAGE}

# Create folder and copy salt
COPY . /salt
WORKDIR /salt

# Install salt
RUN cd /salt && python -m pip install .

