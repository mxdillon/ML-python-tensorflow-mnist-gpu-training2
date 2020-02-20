#!/usr/bin/env bash

set -e

echo "RUNNING BASHRC"
source /root/.bashrc

echo "INSTALLING TF"
conda install -c anaconda tensorflow-gpu=1.15.0

echo "INSTALLING REQUIREMENTS"
python -m pip install -r ./requirements.txt

echo "RUNNING UNIT TESTS"
pytest -v

echo "RUNNING TRAINING"
python app.py