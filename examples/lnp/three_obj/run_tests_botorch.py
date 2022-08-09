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


from atlas.optimizers.gp.planner import BoTorchPlanner


budget = 160
num_repeats = 20
batch_size = 32

data_all_repeats = []


for num_repeat in range(num_repeats):

	# load olympus lnp dataset
	dataset = Dataset('lnp3')

	# load emulator
	emulator = Emulator(dataset='lnp3', model='BayesNeuralNet')

	campaign = Campaign()
	campaign.set_param_space(dataset.param_space)
	campaign.set_value_space(dataset.value_space)

	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		is_moo=True,
		value_space=dataset.value_space,
		batch_size=batch_size,
		num_init_design=batch_size,
		scalarizer_kind='Hypervolume',
		moo_params={},
		goals=['max', 'max', 'min']
	)
	planner.set_param_space(dataset.param_space)

	scalarizer = Scalarizer(
			kind='Hypervolume',
			value_space=dataset.value_space,
			goals=['max', 'max', 'min']
	)

	# for num_iter in range(budget):
	while len(campaign.observations.get_values()) < budget:

		print(f'repeat {num_repeat}\tnum obs {len(campaign.observations.get_values())}')

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			sample_arr = sample.to_list()
			measurement = emulator.run(sample, return_paramvector=True)
			campaign.add_and_scalarize(sample.to_array(), measurement, scalarizer)


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

	pickle.dump(data_all_repeats, open(f'results/results_botorch_batch_{batch_size}_init.pkl', 'wb'))

