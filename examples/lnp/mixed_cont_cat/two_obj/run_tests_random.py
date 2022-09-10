#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd

from olympus.objects import (
	ParameterContinuous,
	ParameterDiscrete,
	ParameterCategorical,
)
from olympus.campaigns import Campaign, ParameterSpace
from olympus.surfaces import Surface
from olympus.scalarizers import Scalarizer
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.planners import RandomSearch


budget = 243
num_repeats = 20


data_all_repeats = []

for num_repeat in range(num_repeats):

	dataset = Dataset('lnp')

	# load emulator
	emulator = Emulator(dataset='lnp', model='BayesNeuralNet')


	campaign = Campaign()
	campaign.set_param_space(dataset.param_space)
	campaign.set_value_space(dataset.value_space)

	planner = RandomSearch(goal='minimize')
	planner.set_param_space(dataset.param_space)

	scalarizer = Scalarizer(
			kind='Hypervolume',
			value_space=dataset.value_space,
			goals=['min', 'max'],
		)

	for num_iter in range(budget):

		print(f'repeat {num_repeat}\titer {num_iter+1}')

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			sample_arr = sample.to_list()
		
			print('sample arr :', sample_arr)
			measurement = emulator.run(sample_arr, return_paramvector=True)

			campaign.add_and_scalarize(sample.to_array(), measurement, scalarizer)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)



	# store the results into a DataFrame
	param_cols = {}
	for param_ix in range(len(dataset.param_space)):
		param_cols[dataset.param_space[param_ix].name] = campaign.observations.get_params()[:, param_ix]

	value_cols = {}
	for value_ix in range(len(dataset.value_space)):
		value_cols[dataset.value_space[value_ix].name] = campaign.observations.get_values()[:, value_ix]

	cols = {**param_cols, **value_cols}

	data = pd.DataFrame(cols)
	data_all_repeats.append(data)

	pickle.dump(data_all_repeats, open('results/results_random.pkl', 'wb'))

