#!/bin/bash

mkdir -p data

curl -L -o data/conbal.zip\
    https://www.kaggle.com/api/v1/datasets/download/pablomoratodomnguez/conbal \

unzip -o data/conbal.zip -d data/