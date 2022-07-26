#!/usr/bin/env python

import numpy as np

from olympus.objects import (
	ParameterContinuous,
	ParameterDiscrete,
	ParameterCategorical,
	ParameterVector
)
from olympus.campaigns import Campaign, ParameterSpace
from olympus.surfaces import Surface

from atlas.optimizers.gp.planner import BoTorchPlanner

from atlas.sheets.sheet_manager import SheetManager

def test_continuous_batched():

	BATCH_SIZE = 2
	NUM_INIT_DESIGN = 4

	sheet_config = {
		'sa_filename': None,
		'sheet_name': 'test_atlas',
		'worksheet_name': 'Sheet1',
		'monitor_interval': 5, # interval in seconds
	}
	sheet_manager = SheetManager(sheet_config)

	def surface(x):
		return np.sin(8*x[0]) - 2*np.cos(6*x[1]) + np.exp(-2.*x[2])

	param_space = ParameterSpace()
	param_0 = ParameterContinuous(name='param_0', low=0.0, high=1.0)
	param_1 = ParameterContinuous(name='param_1', low=0.0, high=1.0)
	param_2 = ParameterContinuous(name='param_2', low=0.0, high=1.0)
	param_space.add(param_0)
	param_space.add(param_1)
	param_space.add(param_2)

	value_space = ParameterSpace()
	obj_0 = ParameterContinuous(name='obj_0')
	value_space.add(obj_0)

	planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='naive-0',
		init_design_strategy='lhs',
		num_init_design=NUM_INIT_DESIGN, 
		batch_size=BATCH_SIZE,
	)

	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)
	campaign.set_value_space(value_space)

	BUDGET = 20


	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		df = sheet_manager.df_from_campaign(campaign, samples)
		# write the recommendations to the google sheet
		sheet_manager.write_sheet(df)

		# monitor the google sheet for results (sleeping)
		sheet_manager.monitor_sheet()

		# read in the results 
		df = sheet_manager.read_sheet()

		# add the newest results to the olympus campaign
		for sample_ix, sample in enumerate(samples):
			# sample_arr = sample.to_array()
			# measurement = surface(sample_arr)
			value_names = [v.name for v in value_space]
			row = df.iloc[-BATCH_SIZE+sample_ix, :][value_names].to_dict()
			
			measurement = ParameterVector().from_dict(row)

			print('sample_ix : ', sample_ix )
			print('measurement : ', measurement)

			campaign.add_observation(sample, measurement)





if __name__ == '__main__':

	test_continuous_batched()
	#test_categorical_batched()