#!/usr/bin/env python

import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface

from atlas.optimizers.gp.planner import BoTorchPlanner

CONT = {
    "init_design_strategy": [
        "random",
        "sobol",
        "lhs",
    ],  # init design strategues
    "batch_size": [1, 5],  # batch size
    "use_descriptors": [False],  # use descriptors
}

DISC = {
    "init_design_strategy": ["random"],
    "batch_size": [1, 5],
    "use_descriptors": [False],
}

CAT = {
    "init_design_strategy": ["random"],
    "batch_size": [1, 5],
    "use_descriptors": [False, True],
}

MIXED_CAT_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
}

MIXED_DISC_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False],
}


MIXED_CAT_DISC = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
}

MIXED_CAT_DISC_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
}

SCALARIZER_KINDS = ["WeightedSum", "Parego", "Hypervolume", "Chimera"]


# def generate_scalarizer_object(scalarizer_kind, value_space):
#
# 	if scalarizer_kind=='WeightedSum':
# 		moo_params={
# 			'weights': np.random.randint(1, 5, size=len(value_space))
# 		}
# 		scalarizer = Scalarizer(
# 			kind=scalarizer_kind,
# 			value_space=value_space,
# 			moo_params=
# 		)
# 	elif scalarizer_kind=='Parego':
# 		moo_params={}
# 	elif scalarizer_kind=='Hypervolume':
# 		moo_params={}
# 	elif scalarizer_kind=='Chimera':
# 		moo_params={
# 			'absolutes': [False for _ in range(len(value_space))],
# 			'tolerances': np.random.rand(len(value_space))
# 		}
# 	goals = np.random.choice(['min', 'max'], size=len(value_space))
#
#
# 	return None


# def test_continuous_hypervolume():
#
# 	moo_surface = Surface(kind='MultFonseca')
#
# 	planner = BoTorchPlanner(
# 		goal='minimize',
# 		feas_strategy='naive-0',
# 		init_design_strategy='lhs',
# 		is_moo=True,
# 		value_space=moo_surface.value_space,
# 		scalarizer_kind='Hypervolume',
# 		moo_params={},
# 		goals=['min', 'max']
# 	)
#
# 	scalarizer = Scalarizer(
# 		kind='Hypervolume',
# 		value_space=moo_surface.value_space,
# 		goals=['min', 'max'],
# 	)
#
# 	planner.set_param_space(moo_surface.param_space)
#
# 	campaign = Campaign()
# 	campaign.set_param_space(moo_surface.param_space)
# 	campaign.set_value_space(moo_surface.value_space)
#
#
# 	BUDGET = 10
#
# 	while len(campaign.observations.get_values()) < BUDGET:
#
# 		samples = planner.recommend(campaign.observations)
#
# 		for sample in samples:
# 			sample_arr = sample.to_array()
# 			measurement = moo_surface.run(sample_arr, return_paramvector=True)
# 			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)
#
#
# 	assert len(campaign.observations.get_params())==BUDGET
# 	assert len(campaign.observations.get_values())==BUDGET
# 	assert campaign.observations.get_values().shape[1] == len(moo_surface.value_space)
#
#
# def test_continuous_parego():
#
# 	moo_surface = Surface(kind='MultFonseca')
#
#
# 	planner = BoTorchPlanner(
# 		goal='minimize',
# 		feas_strategy='naive-0',
# 		init_design_strategy='lhs',
# 		is_moo=True,
# 		value_space=moo_surface.value_space,
# 		scalarizer_kind='Parego',
# 		moo_params={},
# 		goals=['min', 'max']
# 	)
#
# 	scalarizer = Scalarizer(
# 		kind='Parego',
# 		value_space=moo_surface.value_space,
# 		goals=['min', 'max'],
# 	)
#
# 	planner.set_param_space(moo_surface.param_space)
#
# 	campaign = Campaign()
# 	campaign.set_param_space(moo_surface.param_space)
# 	campaign.set_value_space(moo_surface.value_space)
#
#
# 	BUDGET = 10
#
# 	while len(campaign.observations.get_values()) < BUDGET:
#
# 		samples = planner.recommend(campaign.observations)
#
# 		for sample in samples:
# 			sample_arr = sample.to_array()
# 			measurement = moo_surface.run(sample_arr, return_paramvector=True)
# 			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)
#
#
# 	assert len(campaign.observations.get_params())==BUDGET
# 	assert len(campaign.observations.get_values())==BUDGET
# 	assert campaign.observations.get_values().shape[1] == len(moo_surface.value_space)
#
#
# def test_continuous_weighted_sum():
#
# 	moo_surface = Surface(kind='MultFonseca')
#
#
# 	planner = BoTorchPlanner(
# 		goal='minimize',
# 		feas_strategy='naive-0',
# 		init_design_strategy='lhs',
# 		is_moo=True,
# 		value_space=moo_surface.value_space,
# 		scalarizer_kind='WeightedSum',
# 		moo_params={'weights': [2., 1.]},
# 		goals=['min', 'max']
# 	)
#
# 	scalarizer = Scalarizer(
# 		scalarizer_kind='WeightedSum',
# 		value_space=moo_surface.value_space,
# 		moo_params={'weights': [2., 1.]},
# 		goals=['min', 'max']
# 	)
#
# 	planner.set_param_space(moo_surface.param_space)
#
# 	campaign = Campaign()
# 	campaign.set_param_space(moo_surface.param_space)
# 	campaign.set_value_space(moo_surface.value_space)
#
#
# 	BUDGET = 10
#
# 	while len(campaign.observations.get_values()) < BUDGET:
#
# 		samples = planner.recommend(campaign.observations)
#
# 		for sample in samples:
# 			sample_arr = sample.to_array()
# 			measurement = moo_surface.run(sample_arr, return_paramvector=True)
# 			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)
#
#
# 	assert len(campaign.observations.get_params())==BUDGET
# 	assert len(campaign.observations.get_values())==BUDGET
# 	assert campaign.observations.get_values().shape[1] == len(moo_surface.value_space)
#
#
# def test_continuous_chimera():
#
# 	moo_surface = Surface(kind='MultFonseca')
#
#
# 	planner = BoTorchPlanner(
# 		goal='minimize',
# 		feas_strategy='naive-0',
# 		init_design_strategy='lhs',
# 		is_moo=True,
# 		value_space=moo_surface.value_space,
# 		scalarizer_kind='Chimera',
# 		moo_params={'absolutes': [True, True], 'tolerances': [0.5, 0.5]},
# 		goals=['min', 'max']
# 	)
#
# 	scalarizer = Scalarizer(
# 		kind='Chimera',
# 		value_space=moo_surface.value_space,
# 		goals=['min', 'max'],
# 		absolutes=[True, True],
# 		tolerances=[0.5, 0.5],
# 	)
#
# 	planner.set_param_space(moo_surface.param_space)
#
# 	campaign = Campaign()
# 	campaign.set_param_space(moo_surface.param_space)
# 	campaign.set_value_space(moo_surface.value_space)
#
#
# 	BUDGET = 10
#
# 	while len(campaign.observations.get_values()) < BUDGET:
#
# 		samples = planner.recommend(campaign.observations)
#
# 		for sample in samples:
# 			sample_arr = sample.to_array()
# 			measurement = moo_surface.run(sample_arr, return_paramvector=True)
# 			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)
#
#
# 	assert len(campaign.observations.get_params())==BUDGET
# 	assert len(campaign.observations.get_values())==BUDGET
# 	assert campaign.observations.get_values().shape[1] == len(moo_surface.value_space)
#
#
# def run_continuous(
# 	init_design_strategy, batch_size, use_descriptors, scalarizer, num_init_design=5,
# ):


if __name__ == "__main__":

    # test_continuous_weighted_sum()
    # test_continuous_hypervolume()
    # test_continuous_chimera()
    # test_continuous_parego()

    pass
