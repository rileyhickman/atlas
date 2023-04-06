#!/usr/bin/env python 

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openai

from gptchem.gpt_classifier import GPTClassifier
from gptchem.tuner import Tuner

# set openai key
openai.api_key = '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'

train_smiles = ["CC", "CDDFSS"]
train_labels = [0, 1]
test_smiles = ['CCCC', 'CCCCCCCC']


classifier = GPTClassifier(
    property_name='transition wavelength',
    tuner=Tuner(
        n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False,
    )
)

classifier.fit(train_smiles, train_labels)