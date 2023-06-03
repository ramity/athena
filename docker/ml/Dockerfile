FROM python:3.10-slim

# Update distro
RUN apt update
RUN apt upgrade -y

# Install pip deps
RUN pip install --upgrade pip
RUN pip install torch
RUN pip install torchvision
RUN pip install jupyterlab
RUN pip install numpy
RUN pip install matplotlib
RUN pip install ipympl

WORKDIR /var/notebooks

# Start jupyter lab
ENTRYPOINT jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root
