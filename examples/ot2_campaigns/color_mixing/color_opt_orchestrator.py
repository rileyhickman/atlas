#!/usr/bin/env python

import os, sys
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import yaml
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
	ParameterCategorical,
	ParameterContinuous,
	ParameterDiscrete,
	ParameterVector,
)
from olympus.utils.data_transformer import cube_to_simpl

from atlas import Logger
from atlas.optimizers.gp.planner import BoTorchPlanner
from atlas.ot2.example_protocols.manager import ProtocolManager
from atlas.ot2.cv import Camera


class Orchestrator:
	''' High-level orchestrator for the color optimization 
	experiment with the OT-2 robot
	''' 

	def __init__(self, config_file:str):
		Logger.log_welcome()
		if os.path.isfile(config_file):
			content = open(f"{config_file}", "r")
			self.config = yaml.full_load(content)
		else:
			Logger.log(f"File {config_file} does not exist", "FATAL")

		self.BATCH_SIZE = self.config["general"]["batch_size"]
		self.BUDGET = self.config["general"]["budget"]
		self.TIMEOUT = self.config["general"]["timeout"]
		self.TOTAL_VOLUME = self.config['general']['total_volume']

		Logger.log_chapter(
			"Initializing parameter space and campaign",
			line="=",
			style="bold #d9ed92",
		)

		self.func_param_space, self.full_param_space = self.get_param_space(self.config)
		self.value_space = self.get_value_space(self.config)

		self.set_campaign()

		self.set_planner()

		self.camera = Camera(
				grid_dims=(
					self.config['general']['grid_dims']['rows'],
					self.config['general']['grid_dims']['cols']
					),
				save_img_path=self.config['general']['save_img_path'],
		)



	def set_campaign(self):
		# functional campaign
		self.func_campaign = Campaign()
		self.func_campaign.set_param_space(self.func_param_space)
		self.func_campaign.set_value_space(self.value_space)

		# full campaign
		self.full_campaign = Campaign()
		self.full_campaign.set_param_space(self.full_param_space)
		self.full_campaign.set_value_space(self.value_space)


	def get_param_space(self, config):
		''' Generate Olympus param space
		'''
		func_param_space = ParameterSpace()
		full_param_space = ParameterSpace()
		params = config['params']
		param_names = [k for k in params.keys()]

		for component in param_names:
			if isinstance(params[component]['low'],float):
				# we have range, add to functional params
				param_to_add = ParameterContinuous(
					name=component,
					low=params[component]['low'],
					high=params[component]['high']
				)
				func_param_space.add(param_to_add)
				full_param_space.add(param_to_add)
			else:
				param_to_add = ParameterContinuous(
				name=component,
				low=0.0,
				high=1.0,
			)
				full_param_space.add(param_to_add)
		return func_param_space, full_param_space


	def get_value_space(self, config):
		objectives = config["objs"]
		value_space = ParameterSpace()
		for obj in objectives.keys():
			value_space.add(
				ParameterContinuous(
					name=objectives[obj]["name"]
					)
			)
		return value_space


	def compute_volumes(self, func_params):
		# func_params are just red, yellow, blue
		func_params_arr = [p.to_array() for p in func_params]
		full_params_arr = cube_to_simpl(func_params_arr)

		full_params = [
			ParameterVector().from_array(i, param_space=self.full_param_space) for i in full_params_arr
		]

		v = full_params_arr * self.TOTAL_VOLUME
		volumes = {}
		for ix, param in enumerate(self.full_param_space):
			volumes[param.name] = v[:,ix]

		return volumes, full_params


	def set_planner(self):
		self.planner = BoTorchPlanner(
			goal = self.config['objs']['loss']['goal'],
			batch_size=self.BATCH_SIZE,
			init_design_strategy='random',
			num_init_design=1,
		)
		self.planner.set_param_space(self.func_param_space)
	
	def instantiate_protocol_manager(self, parameters):

		protocol_config=self.config['protocol']
		protocol_manager = ProtocolManager(
			protocol_name=protocol_config['name'],
			protocol_parameters=parameters
		)
		return protocol_manager


	def orchestrate(self):

		if not self.TIMEOUT:
			Logger.log(
				f"Timeout not set. Experiment will proceed indefinitley for {self.BUDGET} measurements",
				"WARNING",
			)
		elif isinstance(self.TIMEOUT, int):
			Logger.log(
				f"Experiment will proceed for {self.BUDGET} measurements or {self.TIMEOUT} seconds",
				"INFO",
			)
		else:
			Logger.log(
				f"Timeout type not understood. Must provide integer.", "FATAL"
			)

		# begin experiment
		Logger.log_chapter(
			"Commencing experiment", line="=", style="bold #d9ed92"
		)
		iteration = 0
		num_batches = 0
		while len(self.func_campaign.observations.get_values()) < self.BUDGET:

			num_obs = len(self.func_campaign.observations.get_values())

			Logger.log_chapter(
				f"STARTING BATCH NUMBER {num_batches+1} ({num_obs}/{self.BUDGET} OBSERVATIONS FOUND)",
				line="-",
				style="cyan",
			)

			func_batch_params = self.planner.recommend(self.func_campaign.observations)

			transfer_volumes, full_batch_params = self.compute_volumes(func_batch_params)

			print('\nPROPOSED PARAMS : ',  full_batch_params)

			print('\nTRANSFER VOLUMES : ', transfer_volumes)
			print('\n')

			# add iteration number to parameters 
			transfer_volumes['iteration'] = iteration
			
			protocol_manager = self.instantiate_protocol_manager(
				parameters=transfer_volumes
			)
			protocol_manager.spawn_protocol_file()


			# execute the color mixing i.e. sample prep
			protocol_manager.execute_protocol(
				simulation=self.config["protocol"]["simulation"]
			)

			# when the sample prep has finished, take a picture with the webcam 
			# and measure the loss function value

			loss, _ = self.camera.make_measurement(
				iteration=iteration, 
				target_rgb = Camera.hex_to_rgb(
						self.config['general']['target_hexcode']
				),
				save_img=True,
			)
			# loss = np.random.uniform(1, 1000)

			print('\nLOSS : ', loss)
			print('\n')

			# add to Olympus campaigns
			self.func_campaign.add_observation(func_batch_params[0], loss)
			self.full_campaign.add_observation(full_batch_params[0], loss)
		  
			# save to disk
			pickle.dump(
				[self.func_campaign,self.full_campaign],
				open(self.config['general']['save_img_path']+'results.pkl','wb')
			)

			iteration += 1
			num_batches+=1





if __name__ == '__main__':

	runner = Orchestrator(config_file='color_opt_config.yaml')

	runner.orchestrate()



	














