#!/usr/bin/env python

import os, sys
import shutil
import pickle
import numpy as np
import pandas as pd

from olympus import __home__
from olympus.datasets import Dataset


def save_pkl_file(data_all_repeats):
	"""save pickle file with results so far"""

	if os.path.isfile('results.pkl'):
		shutil.move('results.pkl', 'bkp-results.pkl')  # overrides existing files

	# store run results to disk
	with open("results.pkl", "wb") as content:
		pickle.dump(data_all_repeats, content)


def load_data_from_pkl_and_continue(N):
	"""load results from pickle file"""

	data_all_repeats = []
	# if no file, then we start from scratch/beginning
	if not os.path.isfile('results.pkl'):
		return data_all_repeats, N

	# else, we load previous results and continue
	with open("results.pkl", "rb") as content:
		data_all_repeats = pickle.load(content)

	missing_N = N - len(data_all_repeats)

	return data_all_repeats, missing_N


dataset = Dataset(kind='lnp3')
df = dataset.data

num_repeats = 40
budget = 50#df.shape[0]


#------------------
# begin experiment
#------------------

data_all_repeats = []

for repeat in range(num_repeats):

	print(f'starting repeat {repeat+1} of {num_repeats}')

	df = df.sample(frac=1).reset_index(drop=True)

	data_all_repeats.append(df)

	save_pkl_file(data_all_repeats)
	
		




