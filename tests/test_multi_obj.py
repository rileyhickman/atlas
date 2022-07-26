#!/usr/bin/env python

import pytest
import numpy as np

from olympus.objects import (
	ParameterContinuous,
	ParameterDiscrete,
	ParameterCategorical,
)
from olympus.campaigns import Campaign, ParameterSpace
from olympus.surfaces import Surface
from olympus.scalarizers import Scalarizer

from atlas.optimizers.gp.planner import BoTorchPlanner


def test_continuous_hypervolume():

	moo_surface = Surface(kind='MultFonseca')


	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		is_moo=True,
		value_space=moo_surface.value_space,
		scalarizer_kind='Hypervolume',
		moo_params={},
		goals=['min', 'max']
	)

	scalarizer = Scalarizer(
		kind='Hypervolume',
		value_space=moo_surface.value_space,
		goals=['min', 'max'],
	)

	planner.set_param_space(moo_surface.param_space)

	campaign = Campaign()
	campaign.set_param_space(moo_surface.param_space)
	campaign.set_value_space(moo_surface.value_space)


	BUDGET = 10

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			sample_arr = sample.to_array()
			measurement = moo_surface.run(sample_arr, return_paramvector=True)
			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)


	assert len(campaign.observations.get_params())==BUDGET
	assert len(campaign.observations.get_values())==BUDGET
	assert campaign.observations.get_values().shape[1] == len(moo_surface.value_space)


def test_continuous_parego():

	moo_surface = Surface(kind='MultFonseca')


	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		is_moo=True,
		value_space=moo_surface.value_space,
		scalarizer_kind='Parego',
		moo_params={},
		goals=['min', 'max']
	)

	scalarizer = Scalarizer(
		kind='Parego',
		value_space=moo_surface.value_space,
		goals=['min', 'max'],
	)

	planner.set_param_space(moo_surface.param_space)

	campaign = Campaign()
	campaign.set_param_space(moo_surface.param_space)
	campaign.set_value_space(moo_surface.value_space)


	BUDGET = 10

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			sample_arr = sample.to_array()
			measurement = moo_surface.run(sample_arr, return_paramvector=True)
			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)


	assert len(campaign.observations.get_params())==BUDGET
	assert len(campaign.observations.get_values())==BUDGET
	assert campaign.observations.get_values().shape[1] == len(moo_surface.value_space)


def test_continuous_weighted_sum():

	moo_surface = Surface(kind='MultFonseca')


	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		is_moo=True,
		value_space=moo_surface.value_space,
		scalarizer_kind='WeightedSum',
		moo_params={'weights': [2., 1.]},
		goals=['min', 'max']
	)

	scalarizer = Scalarizer(
		kind='Parego',
		value_space=moo_surface.value_space,
		goals=['min', 'max'],
	)

	planner.set_param_space(moo_surface.param_space)

	campaign = Campaign()
	campaign.set_param_space(moo_surface.param_space)
	campaign.set_value_space(moo_surface.value_space)


	BUDGET = 10

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			sample_arr = sample.to_array()
			measurement = moo_surface.run(sample_arr, return_paramvector=True)
			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)


	assert len(campaign.observations.get_params())==BUDGET
	assert len(campaign.observations.get_values())==BUDGET
	assert campaign.observations.get_values().shape[1] == len(moo_surface.value_space)


def test_continuous_chimera():

	moo_surface = Surface(kind='MultFonseca')


	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		is_moo=True,
		value_space=moo_surface.value_space,
		scalarizer_kind='Chimera',
		moo_params={'absolutes': [True, True], 'tolerances': [0.5, 0.5]},
		goals=['min', 'max']
	)

	scalarizer = Scalarizer(
		kind='Chimera',
		value_space=moo_surface.value_space,
		goals=['min', 'max'],
		absolutes=[True, True],
		tolerances=[0.5, 0.5],
	)

	planner.set_param_space(moo_surface.param_space)

	campaign = Campaign()
	campaign.set_param_space(moo_surface.param_space)
	campaign.set_value_space(moo_surface.value_space)


	BUDGET = 10

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			sample_arr = sample.to_array()
			measurement = moo_surface.run(sample_arr, return_paramvector=True)
			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)


	assert len(campaign.observations.get_params())==BUDGET
	assert len(campaign.observations.get_values())==BUDGET
	assert campaign.observations.get_values().shape[1] == len(moo_surface.value_space)



# def test_categorical():

# 	surface0_kind = 'CatDejong'
# 	surface1_kind = 'CatCamel'
# 	surface0 = Surface(kind=surface_kind0, param_dim=2, num_opts=21)
# 	surface1 = Surface(kind=surface_kind1, param_dim=2, num_opts=21)

# 	value_space = ParameterSpace()
# 	obj0=ParameterContinuous(name='obj0')
# 	obj1=ParameterContinuous(name='obj1')
# 	value_space.add(obj0)
# 	value_space.add(obj1)

# 	campaign = Campaign()
# 	campaign.set_param_space(surface0.param_space)
# 	campaign.set_value_space(value_space)

# 	planner = BoTorchPlanner(
# 		goal='minimize',
# 		feas_strategy='naive-0',
# 		is_moo=True,
# 		value_space=value_space,
# 		scalarizer_kind='Hypervolume',
# 		moo_params={},
# 		goals=['min', 'max']
# 	)

# 	scalarizer = Scalarizer(
# 		kind='Hypervolume',
# 		value_space=value_space,
# 		goals=['min', 'max'],
# 	)

# 	planner.set_param_space(surface.param_space)


# 	BUDGET = 100

# 	for iter in range(BUDGET):

# 		sample = planner.recommend(campaign.observations)
# 		print(f'ITER : {iter}\tSAMPLES : {sample}')
# 		sample_arr = sample.to_array()
# 		measurement0 = surface0.run(sample_arr, return_paramvector=False)
# 		measurement1 = surface1.run(sample_arr, return_paramvector=False)
# 		measurement = ParameterVector()
# 		campaign.add_observation(sample_arr, measurement[0])


if __name__ == '__main__':

	test_continuous_weighted_sum()
	test_continuous_hypervolume()
	test_continuous_chimera()
	test_continuous_parego()
