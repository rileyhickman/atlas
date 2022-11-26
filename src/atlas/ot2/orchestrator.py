#!/usr/bin/env python

import os, sys
import time
import yaml

from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterVector

from atlas import Logger
from atlas.optimizers.gp.planner import BoTorchPlanner
from atlas.ot2.volume_converter import *
from atlas.ot2.example_protocols.manager import ProtocolManager
from atlas.sheets.sheet_manager import SheetManager



class Orchestrator:
	''' High-level orchestrator for drug formulation experiment 
	'''
	def __init__(self, config_file):
		# load config file
		if os.path.isfile(config_file):
			content = open(f'{config_file}', 'r')
			self.campaign_config = yaml.full_load(content)
		else:
			Logger.log(f'File {config_file} does not exist', 'FATAL')

		# compute molar target concentrations
		self.campaign_config = compute_target_conc(self.campaign_config)

		self.BATCH_SIZE = self.campaign_config['general']['batch_size']
		self.BUDGET = self.campaign_config['general']['budget']
		self.TIMEOUT = self.campaign_config['general']['timeout']


		# get Olympus parameter space and value space
		self.func_param_space, self.full_param_space, self.full_space_instructions = get_param_space(self.campaign_config) 
		self.value_space = get_value_space(self.campaign_config)

		# set campaign
		self.set_campaign()

		# set planner 
		self.set_planner()

		# setup Google sheet manager
		self.instantiate_sheet_manager()



	def set_campaign(self):
		# functional campaign
		self.func_campaign = Campaign()
		self.func_campaign.set_param_space(self.func_param_space)
		self.func_campaign.set_value_space(self.value_space)

		# full campaign
		self.full_campaign = Campaign()
		self.full_campaign.set_param_space(self.full_param_space)
		self.full_campaign.set_value_space(self.value_space)


	def set_planner(self):
		# instantiate planner with functional parameter space
		# check for multiple objectives
		objectives = self.campaign_config['objectives']
		if len(objectives)>1:
			setattr(self, 'is_moo', True)
			# multiple objectives
			goals = [campaign_config['objectives'][f'obj{ix+1}']['goal'] for ix in range(len(objectives))]
			self.planner = BoTorchPlanner(
				goal='minimize',
				batch_size=self.BATCH_SIZE,
				init_design_strategy='random',
				num_init_design=self.BATCH_SIZE,
				is_moo=True,
				value_space=self.func_campaign.value_space,
				scalarizer_kind='Hypervolume',
				moo_params={},
				goals=goals,
			)
		elif len(objectives)==1:
			# single objective
			setattr(self, 'is_moo', False)
			self.planner = BoTorchPlanner(
				goal=''.join(
					(self.campaign_config['objectives']['obj1']['goal'],'imize')
				),
				batch_size=self.BATCH_SIZE,
				init_design_strategy='random',
				num_init_design=self.BATCH_SIZE,
			)
		else:
			Logger.log('You musy provide at least one objective', 'FATAL')

		self.planner.set_param_space(self.func_param_space)


	def instantiate_sheet_manager(self):
		''' Generates instance of Google sheet API manager
		'''
		sheet_config = self.campaign_config['sheets']
		self.sheet_manager = SheetManager(config=sheet_config)

	def instantiate_protocol_manager(self, parameters):
		''' Generates protocol manager instance
		'''
		protocol_config = self.campaign_config['protocol']
		protocol_manager = ProtocolManager(
				protocol_name=protocol_config['name'],
				protocol_parameters=parameters,
			)
		return protocol_manager



	def simulate_protocol(self):
		''' Simulates the OT2 protocol and checks for errors 
		'''
		return None

	def check_ot2_server_connection(self):
		''' Verify connection to OT2 robot server
		'''
		return None


	def orchestrate(self):
		''' Execute the experiment for a given budget of measurements or until a cumulative timeout
		threshold is achieved 
		'''
		if not self.TIMEOUT:
			Logger.log(f'Timeout not set. Experiment will proceed indefinitley for {self.BUDGET} measurements', 'WARNING')
		elif isinstance(self.TIMEOUT, int):
			Logger.log(f'Experiment will proceed for {self.BUDGET} measurements or {self.TIMEOUT} seconds', 'INFO')
		else:
			Logger.log(f'Timeout type not understood. Must provide integer.', 'FATAL')


		# begin experiment

		while len(self.func_campaign.observations.get_values()) < self.BUDGET:

			if self.is_moo:
				obs_for_planner = self.func_campaign.scalarized_observations
			else:
				obs_for_planner = self.func_campaign.observations

			print('\nparams : ', obs_for_planner.get_params())
			print('\nvalues : ', obs_for_planner.get_values())

			func_batch_params = self.planner.recommend(obs_for_planner)

			print('\nfunc_batch_params : ', func_batch_params)

			# convert the samples to the "full" parameter space
			full_batch_params = func_to_full_params(func_batch_params, self.full_space_instructions)

			print('\nfull_batch_params : ', full_batch_params)
		
			# write new params to sheet
			df = self.sheet_manager.df_from_campaign(self.full_campaign, full_batch_params)
			self.sheet_manager.write_sheet(df)

			# convert params to transfer volumes for the OT2
			transfer_volumes = compute_transfer_volumes(self.campaign_config, full_batch_params)

			# TODO: convert the transfer volumes to be recognizable by the protocol (do in volume converter)

			# TODO: generate protocol manager, make OT2 run file, send to OT2 and execute it
			protocol_manager = self.instantiate_protocol_manager(parameters=transfer_volumes)
			protocol_manager.spawn_protocol_file()

			# TODO: send to OT2


			# execute the protocol
			protocol_manager.execute_protocol(simulation=self.campaign_config['protocol']['simulation'])

			# monitor the google sheet for results
			self.sheet_manager.monitor_sheet()

			# read in the results
			df = self.sheet_manager.read_sheet()

			# add the newest results to the Olympus campaigns
			for sample_ix, (func_sample, full_sample) in enumerate(zip(func_batch_params,full_batch_params)):
				value_names = [v.name for v in self.value_space]
				row = df.iloc[-self.BATCH_SIZE+sample_ix, :][value_names].to_dict()
				measurement = ParameterVector().from_dict(row)

				self.func_campaign.add_observation(func_sample, measurement)
				self.full_campaign.add_observation(full_sample, measurement)

		



if __name__ == '__main__':


	runner = Orchestrator(config_file='campaign_config.yaml')

	runner.orchestrate()
















		





