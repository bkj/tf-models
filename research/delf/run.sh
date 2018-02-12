#!/bin/bash

# run.sh

# --
# Install

conda create -n delf pip python=2.7
source activate delf

URL="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl"
pip install --ignore-installed --upgrade $URL

cd ../slim && pip install -e . && cd ../delf
cd .. && export PYTHONPATH=$PYTHONPATH:`pwd` && cd ./delf


protoc delf/protos/*.proto --python_out=.
pip install -e .
python -c 'import delf'

# --
# Download models

mkdir parameters && cd parameters
wget http://download.tensorflow.org/models/delf_v1_20171026.tar.gz
tar -xvzf delf_v1_20171026.tar.gz

# --
# Download data

mkdir data && cd data
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz