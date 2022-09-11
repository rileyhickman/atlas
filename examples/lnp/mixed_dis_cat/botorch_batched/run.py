#!/usr/bin/env python

import time
import os, sys
import shutil
import pickle
import numpy as np
import pandas as pd

from olympus import __home__
from olympus.datasets import Dataset
from olympus.scalarizers import Scalarizer
from olympus.campaigns import Campaign, ParameterSpace

from olympus.objects import (
	ParameterContinuous,
	ParameterDiscrete,
	ParameterCategorical,
	ParameterVector
)

from atlas.optimizers.gp.planner import BoTorchPlanner


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


def measure(params, df):

	match = df[
		(df['drug_input']==int(float(params[0])))&\
		(df['solid_lipid']==params[1])&\
		(df['solid_lipid_input']==int(float(params[2])))&\
		(df['liquid_lipid_input']==int(float(params[3])))&\
		(df['surfractant_input']==float(params[4]))
	].to_dict('r')
	assert len(match)==1
	match = match[0]
	return ParameterVector().from_dict({
		'drug_loading': match['drug_loading'],
		'encap_efficiency': match['encap_efficiency'],
		'particle_diameter': match['particle_diameter']
	})



"""
Stearic_acid

mol_wt: 284.5
topo_psa: 37.3
rotatable_bond_count: 16
h_bond_donor_count: 1
h_bond_acceptor_count: 2



Compritol_888 (Glyceryl Dibehenate)

mol_wt: 432.7
topo_psa: 98
rotatable_bond_count: 22
h_bond_donor_count: 4
h_bond_acceptor_count: 5





Glyceryl_monostearate

mol_wt: 358.6
topo_psa: 66.8
rotatable_bond_count: 20
h_bond_donor_count: 2
h_bond_acceptor_count: 4


"""




dataset = Dataset(kind='lnp3')
df = dataset.data

param_space = ParameterSpace()
param0 = ParameterDiscrete(
	name='drug_input',
	low=6.0,
	high=48.0,
	options=[6.0,12.0,24.0,48.0],
	#descriptors=[[6.0], [12.0], [24.0], [48.0]]
)
param_space.add(param0)

param1 = ParameterCategorical(
	name='solid_lipid',
	options=["Stearic_acid", "Compritol_888", "Glyceryl_monostearate"],
	descriptors=[
		[284.484, 37.3, 16., 1., 2.],
		[432.7, 98., 22., 4., 5.],
		[358.563, 66.8, 20., 2., 4.]
	]
)
param_space.add(param1)

param2 = ParameterDiscrete(
	name='solid_lipid_input',
	low=72.0,
	high=120.0,
	options=[120.0, 108.0,  96.0,  72.0],
	#descriptors=[[120.0], [108.0],  [96.0],  [72.0]]
)
param_space.add(param2)


param3 = ParameterDiscrete(
	name='liquid_lipid_input',
	low=0.0,
	high=48.0,
	options=[0.0, 12.0, 24.0, 48.0],
	#descriptors=[[0.0], [12.0], [24.0], [48.0]]
)
param_space.add(param3)


param4 = ParameterDiscrete(
	name='surfractant_input',
	low=0.0,
	high=0.01,
	options=[0.0, 0.0025, 0.005, 0.01],
	#descriptors=[[0.0], [0.0025], [0.005], [0.01]]
)
param_space.add(param4)


value_space = ParameterSpace()
value0 = ParameterContinuous(name='drug_loading')
value_space.add(value0)
value1 = ParameterContinuous(name='encap_efficiency')
value_space.add(value1)
value2 = ParameterContinuous(name='particle_diameter')
value_space.add(value2)


# print(param_space)

num_repeats = 40
budget = 640#768
num_init_design=32
batch_size = 32


#------------------
# begin experiment
#------------------

data_all_repeats = []

for repeat in range(num_repeats):

	print(f'starting repeat {repeat+1} of {num_repeats}')


	campaign = Campaign()
	campaign.set_param_space(param_space)
	campaign.set_value_space(value_space)

	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		is_moo=True,
		value_space=value_space,
		batch_size=batch_size,
		num_init_design=num_init_design,
		scalarizer_kind='Hypervolume',
		moo_params={},
		goals=['max', 'max', 'min']
	)
	planner.set_param_space(param_space)

	scalarizer = Scalarizer(
			kind='Hypervolume',
			value_space=dataset.value_space,
			goals=['max', 'max', 'min']
	)

	print(planner.problem_type)



	# for num_iter in range(budget):
	while len(campaign.observations.get_values()) < budget:

		print(f'repeat {repeat}\tnum obs {len(campaign.observations.get_values())}')

		samples = planner.recommend(campaign.observations)

		print('samples : ', samples)

		for sample in samples:
			sample_arr = sample.to_list()
			print('sample_arr :', sample_arr)

			measurement = measure(sample_arr, df)


			campaign.add_and_scalarize(sample, measurement, scalarizer)

			print('measurement :',  measurement)

		# check convergence
		bools = [sample.to_list()==['48.0', 'Compritol_888', '120.0', '0.0', '0.005'] for sample in samples]
		print('bools : ', bools)
		if bools.count(True)>0:
			print('found optimum!')
			break

		print('\n')
		print('-'*100)
		print('\n')

	# store the results into a DataFrame
	param_cols = {}
	for param_ix in range(len(param_space)):
		param_cols[dataset.param_space[param_ix].name] = campaign.observations.get_params()[:, param_ix]

	value_cols = {}
	for value_ix in range(len(value_space)):
		value_cols[dataset.value_space[value_ix].name] = campaign.observations.get_values()[:, value_ix]

	cols = {**param_cols, **value_cols}

	data = pd.DataFrame(cols)
	data_all_repeats.append(data)

	pickle.dump(data_all_repeats, open(f'results.pkl', 'wb'))
