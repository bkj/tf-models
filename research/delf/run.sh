#!/bin/bash

# run.sh

# --
# Install

conda create -n delf pip python=2.7
source activate delf

URL="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp27-none-linux_x86_64.whl"
pip install --ignore-installed --upgrade $URL
pip install matplotlib scipy scikit-image

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
mkdir oxford5k_images oxford5k_features
tar -xvzf oxbuild_images.tgz -C oxford5k_images/
cd ../

rm list_images.txt
echo data/oxford5k_images/hertford_000056.jpg >> list_images.txt
echo data/oxford5k_images/oxford_000317.jpg >> list_images.txt
find ./queries -type f >> list_images.txt

# --

find ./queries -type f | python delf/python/examples/extract_features.py \
  --config-path ./delf/python/examples/delf_config_example.pbtxt \
  --output-dir ./_results/queries

python delf/python/examples/match_images.py \
  --image_1_path ./_data/queries/holiday0.jpg \
  --image_2_path ./_data/queries/holiday-d.jpg \
  --features_1_path ./_results/queries/holiday0.delf \
  --features_2_path ./_results/queries/holiday-d.delf \
  --output_image matched_images.png

rsub matched_images.png