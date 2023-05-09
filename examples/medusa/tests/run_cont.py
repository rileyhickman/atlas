#!/usr/bin/env python

import os, sys
import shutil

import numpy as np
import pandas as pd
import pickle

from atlas.optimizers.medusa.planner import MedusaPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical
from olympus.surfaces import Branin, Michalewicz, Levy, Dejong # cont


#------------------
# Helper functions
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


def measure_single_obj(sample, surf_map):

	# non-functional param will always be x0
	# i.e. index of the surface to measure
	si = int( sample.x0[2:] )

	# functional params will always be x1 and x2 
	# NOTE: these should already be floats, just making sure
	X_func = np.array([float(sample.x1), float(sample.x2)]) 

	return surf_map[si].run(X_func)[0][0]


#----------
# Settings
#----------

# 2d surfaces as the \Tilde{f} objective functions
surf_map = {
	0: Branin(),
	1: Dejong(),
	2: Michalewicz(),
	3: Levy(),
}


with_descriptors_func = False
with_descriptors_nonfunc = False
num_desc_nonfunc = 2 
func_param_type = 'continuous'

budget = 100
repeats = 1
random_seed = None # i.e. use a different random seed each time

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)

def set_param_space(func_param_type='continuous'):
	param_space = ParameterSpace()
	if func_param_type == 'continuous':
		# 2 continuous functional parameters
		x1 = ParameterContinuous(name='x1', low=0.0, high=1.0)
		x2 = ParameterContinuous(name='x2', low=0.0, high=1.0)
		# 1 categorical non-functional parameter 
		if with_descriptors_nonfunc:
			descriptors = [[float(i) for _ in range(num_desc_nonfunc)] for i in range(len(surf_map))]
		else:
			descriptors = [None for _ in range(len(surf_map))]
		x0 = ParameterCategorical(
			name='x0',
			options = [f's_{i}' for i in range(len(surf_map))],
			descriptors=descriptors
		)

	elif func_param_type == 'categorical':
		# 2 categorical functional parameters 
		if with_descriptors_func:
			descriptors = [[float(i), float(i)] for i in range(21)]
		else:
			descriptors = [None for _ in range(21)]
		x1 = ParameterCategorical(
			name='x1',
			options=[f'x_{i}' for i in range(21)],
			descriptors=descriptors,
		)
		x2 = ParameterCategorical(
			name='x2',
			options=[f'x_{i}' for i in range(21)],
			descriptors=descriptors,
		)
		# 1 categorical non-functional parameter 
		if with_descriptors_nonfunc:
			descriptors = [[float(i) for _ in range(num_desc_nonfunc)] for i in range(len(surf_map))]
		else:
			descriptors = [None for _ in range(len(surf_map))]
		x0 = ParameterCategorical(
			name='x0',
			options = [f's_{i}' for i in range(len(surf_map))],
			descriptors=descriptors
		)
	param_space.add(x0)
	param_space.add(x1)
	param_space.add(x2)

	return param_space


for num_repeat in range(missing_repeats):

	param_space = set_param_space(func_param_type=func_param_type)

	planner = MedusaPlanner(
		goal='minimize',
		general_parameters=[0],
		batch_size=1,

	)
	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)


	iter=0
	converged = False
	while len(campaign.observations.get_values()) < budget and not converged:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			print(f'===============================')
			print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {iter+1}')
			print(f'===============================')

			measurement = measure_single_obj(sample, surf_map)

			print('sample : ', sample)
			print('measurement : ', measurement)

			campaign.add_observation(sample.to_array(), measurement)
			
			iter+=1

	
	# store the results into a DataFrame
	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	x2_col = campaign.observations.get_params()[:, 2]
	
	# TODO: this will be the overall objective , i.e. f(x1, x2, ... xNg, G)
	obj_col = campaign.observations.get_values(as_array=True)

	# TODO: maybe add the single objectives as well as columns ???

	data = pd.DataFrame({'x0': x0_col, 'x1': x1_col, 'x1': x1_col, 'obj': obj_col})
	data_all_repeats.append(data)

	# save results to disk
	save_pkl_file(data_all_repeats)
