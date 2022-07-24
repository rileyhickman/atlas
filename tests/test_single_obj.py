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



def test_continuous():

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

	)
	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = 24

	for num_iter in range(BUDGET):

		sample = planner.recommend(campaign.observations)
		print(f'ITER : {num_iter}\tSAMPLES : {sample}')
	
		sample_arr = sample.to_array()
		print(sample_arr)
		measurement = surface(sample_arr)

		campaign.add_observation(sample_arr, measurement)

	assert len(campaign.observations.get_params())==BUDGET
	assert len(campaign.observations.get_values())==BUDGET



if __name__ == '__main__':
	test_continuous()