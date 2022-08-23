#!/usr/bin/env python

import glob

def list_datasets():
    dirs_ = glob.glob('dataset_*/')
    dataset_names = [d.split('/')[-1].split('_')[-1] for d in dirs_]
    print(f'AVAILABLE DATASETS : {dataset_names}')

