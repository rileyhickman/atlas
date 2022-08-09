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

from atlas.optimizers.gp.planner import BoTorchPlanner


INIT_DESIGN_STRATEGIES_CONT = ['random', 'sobol', 'lhs']

INIT_DESIGN_STRATEGIES_CAT = ['random']

INIT_DESIGN_STRATEGIES_MIXED = ['random']

@pytest.mark.parametrize('init_design_strategy', INIT_DESIGN_STRATEGIES_CONT)
def test_init_design_cont(init_design_strategy):
	run_continuous(init_design_strategy)


@pytest.mark.parametrize('init_design_strategy', INIT_DESIGN_STRATEGIES_CAT)
def test_init_design_cat(init_design_strategy):
	run_categorical(init_design_strategy)

@pytest.mark.parametrize('init_design_strategy', INIT_DESIGN_STRATEGIES_MIXED)
def test_init_design_mixed(init_design_strategy):
	run_mixed(init_design_strategy)


def run_continuous(init_design_strategy):

	def surface(x):
		return np.sin(8*x[0]) - 2*np.cos(6*x[1]) + np.exp(-2.*x[2])

	param_space = ParameterSpace()
	param_0 = ParameterContinuous(name='param_0', low=0.0, high=1.0)
	param_1 = ParameterContinuous(name='param_1', low=0.0, high=1.0)
	param_2 = ParameterContinuous(name='param_2', low=0.0, high=1.0)
	param_space.add(param_0)
	param_space.add(param_1)
	param_space.add(param_2)

	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		init_design_strategy=init_design_strategy,
		num_init_design=4,
		batch_size=1, 
	)

	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = 10

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface(sample_arr)
			campaign.add_observation(sample_arr, measurement)


	assert len(campaign.observations.get_params())==BUDGET
	assert len(campaign.observations.get_values())==BUDGET



def run_categorical(init_design_strategy):

	surface_kind = 'CatDejong'
	surface = Surface(kind=surface_kind, param_dim=2, num_opts=21)

	campaign = Campaign()
	campaign.set_param_space(surface.param_space)

	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		init_design_strategy=init_design_strategy,
		num_init_design=4, 
		batch_size=1,
	)
	planner.set_param_space(surface.param_space)

	BUDGET = 10

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = np.array(surface.run(sample_arr))
			#print(sample, measurement)
			campaign.add_observation(sample_arr, measurement[0])

	assert len(campaign.observations.get_params())==BUDGET
	assert len(campaign.observations.get_values())==BUDGET


def run_mixed(init_design_strategy):

	param_space = ParameterSpace()
	# add ligand
	param_space.add(
		ParameterCategorical(
			name='cat_index',
			options=[str(i) for i in range(8)],
			descriptors=[None for i in range(8)],        # add descriptors later
		)
	)
	# add temperature
	param_space.add(
		ParameterContinuous(
			name='temperature',
			low=30.,
			high=110.
		)
	)
	# add residence time
	param_space.add(
		ParameterContinuous(
			name='t',
			low=60.,
			high=600.
		)
	)
	# add catalyst loading
	# summit expects this to be in nM
	param_space.add(
		ParameterContinuous(
			name='conc_cat',
			low=0.835/1000,
			high=4.175/1000,
		)
	)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = BoTorchPlanner(
					goal='maximize', 
					feas_strategy='naive-0',
					init_design_strategy=init_design_strategy,
					num_init_design=4, 
					batch_size=1,
				)
	planner.set_param_space(param_space)

	BUDGET = 10

	def mock_yield(x):
		return np.random.uniform()*100

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = mock_yield(sample)
			#print(f'ITER : {iter}\tSAMPLES : {sample}\t MEASUREMENT : {measurement}')
			campaign.add_observation(sample_arr, measurement)

	assert len(campaign.observations.get_params())==BUDGET
	assert len(campaign.observations.get_values())==BUDGET








if __name__ == '__main__':
	#test_continuous()
	test_categorical()
	#test_mixed()
