#!/usr/bin/bash

pip install -r requirements.txt

git clone git@github.com:aspuru-guzik-group/olympus.git
cd olympus
git checkout dev
pip install -e . 


