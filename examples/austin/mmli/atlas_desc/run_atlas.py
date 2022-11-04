#!/usr/bin/env python

import os, sys
import shutil
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical
from olympus.datasets import Dataset

from atlas.optimizers.gp.planner import BoTorchPlanner


#----------
# Settings
#----------


use_descriptors = True

dataset_name='mmli_opv'

repeats = 10
budget = 200
random_seed = None # i.e. use a different random seed each time
batch_size = 1

print(f'{dataset_name} atlas')

#------------------
# helper functions
#------------------

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


# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)


for num_repeat in range(missing_repeats):

	dataset = Dataset(kind=dataset_name)

	planner = BoTorchPlanner(
		goal='maximize',
		num_init_design=10,
		init_design_strategy='random',
		use_descriptors=use_descriptors,
		batch_size=batch_size,
	)
	planner.set_param_space(dataset.param_space)

	campaign = Campaign(goal='maximize')
	campaign.set_param_space(dataset.param_space)

	while len(campaign.observations.get_values()) < budget:
		print(f'===============================')
		print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {len(campaign.observations.get_values())+1}')
		print(f'===============================')

		samples = planner.recommend(campaign.observations)

		measurement = []
		for sample in samples:
			measurement.extend(dataset.run(sample, return_paramvector=True))

		campaign.add_observation(samples, measurement)


	data_all_repeats.append(campaign)
	save_pkl_file(data_all_repeats)

	print('Done!')
