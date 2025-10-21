# Start with a base image that includes Miniconda and latest Python
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Create a new conda environment with older Python version - 3.9
RUN conda create -n py39 python=3.9 -y

# Make RUN commands use the new environment
SHELL ["/bin/bash", "-c"]

# Verify Python version
RUN python --version

# Set default command to run bash with the environment activated
CMD ["/bin/bash"]
