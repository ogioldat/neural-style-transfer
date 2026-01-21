#!/bin/bash

python -m venv .venv_metal
source .venv_metal/bin/activate

which python

pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal keras
pip install jupyter ipykernel

python -m ipykernel install --user --name tf_metal --display-name "METAL_ENV"
