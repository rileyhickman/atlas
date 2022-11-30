#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import os, sys
import subprocess
import json
from datetime import datetime
import numpy as np

from opentrons import protocol_api

from atlas import Logger
from atlas.ot2.example_protocols import get_protocols_list, ProtocolLoader
from atlas.ot2.example_protocols.abstract_protocol import AbstractProtocol

class ProtocolManager:
	def __init__(self, protocol_name:str, protocol_parameters: Dict[str, np.ndarray]):
		self.protocol_name = protocol_name

		# make sure all values in protocol parametrs dictionary are lists (json serializable)
		self.protocol_parameters = {}
		for key, val in protocol_parameters.items():
			if isinstance(val, np.ndarray):
				self.protocol_parameters[key] = list(val)
			else:
				self.protocol_parameters[key] = val

		self.protocol_module = self.load_protocol(self.protocol_name)
		self.date = datetime.now()

		self.metadata_dict = {
				'protocolName': self.protocol_name,
				'dateModified': f'{self.date.day}/{self.date.month}/{self.date.year}',
				'author':'Soteria Therapeutics',
				'apiLevel': '2.10'
			}


	def load_protocol(self, protocol_name:str) -> AbstractProtocol:
		protocol_name_class = ProtocolLoader.file_to_class(protocol_name)
		protocols_list_class = get_protocols_list()
		if not protocol_name_class in protocols_list_class:
			msg = f'Protocol {protocol_name} not found. Please choose from available options: {protocols_list_filename}'
			Logger.log(msg, 'FATAL')
		else:
			msg = f'Importing protocol {protocol_name}'
			Logger.log(msg, 'INFO')
			module = ProtocolLoader.import_protocol(protocol_name_class)

			return module

	def spawn_protocol_file(self):

		s = '#!/usr/bin/env python\n'
		s += 'from opentrons import protocol_api\n'
		s += 'from atlas.ot2.example_protocols.manager import ProtocolManager\n\n'
		s += 'metadata = '+json.dumps(self.metadata_dict)+'\n\n'

		# write dictionary of parameters for protocol instance
		s += 'protocol_parameters = '+json.dumps(self.protocol_parameters)+'\n\n'

		s += 'def run(protocol: protocol_api.ProtocolContext):\n'
		s += f'\tmanager = ProtocolManager(protocol_name="{self.protocol_name}", protocol_parameters=protocol_parameters)\n'
		s += f'\tprotocol_module_instance = manager.protocol_module(parameters=protocol_parameters)\n'
		s += '\tprotocol_module_instance.run(protocol)'

		with open(f'__OT2_file_{self.protocol_name}.py', 'w') as f:
			f.write(s)


	# def copy_to_ot2(self):
	# 	''' Copy the file to the OT2
	# 	'''
	# 	# TODO: implement this
	# 	# https://support.opentrons.com/s/article/Connecting-to-your-OT-2-with-SSH#:~:text=In%20the%20Opentrons%20App%2C%20find,be%20made%20over%20Wi%2DFi.
	# 	# https://support.opentrons.com/s/article/Setting-up-SSH-access-to-your-OT-2

	# 	filename = f'__OT2_file_{self.protocol_name}.py'

	# 	# scp the file to the open

	# 	return None

	def execute_protocol(self, simulation:Optional[bool]=False):
		''' Run the protocol on the OT2 server
		'''
		# https://docs.opentrons.com/v2/new_advanced_running.html

		if simulation:
			filename = f'__OT2_file_{self.protocol_name}.py'
			msg = f'Simulation of OT2 protocol "{self.protocol_name}" requested. Executing simulation with `opentrons_simulate {filename}`'
			Logger.log(msg, 'WARNING')
			result = subprocess.run(
					f'opentrons_simulate {filename}',
					capture_output=True,
					shell=True,
				)
			# TODO: parse the results of this
			# print('\nstdout :', result.stdout)
			# print('\nstderr : ', result.stderr)
			# quit()
			if True:
				Logger.log(f'Simualtion of OT2 protocol "{self.protocol_name}" finished successfully', 'INFO')
			else:
				Logger.log(f'Simulation of OT2 protocol "{self.protocol_name}" failed!', 'FATAL')

		else:
			raise NotImplementedError




if __name__ == '__main__':

	parameters = dict(
		drug = [33.5, 34.9, 36.3, 38.1, 22.8, 24.1, 25.5, 27.4, 67.1, 69.8, 72.6, 76.2, 45.5, 48.3, 51.0, 54.7, 134.1, 139.6, 145.1, 152.5, 91.1, 96.6, 102.1, 109.4, 268.3, 279.3, 290.3, 304.9, 182.2, 193.2, 204.2, 218.9],
	    cholestrol = [82.5, 82.5, 82.5, 82.5, 185.6, 185.6, 185.6, 185.6, 82.5, 82.5, 82.5, 82.5, 185.6, 185.6, 185.6, 185.6, 82.5, 82.5, 82.5, 82.5, 185.6, 185.6, 185.6, 185.6, 82.5, 82.5, 82.5, 82.5, 185.6, 185.6, 185.6, 185.6],
	    pc = [252.8, 246.5, 240.2, 231.8, 42.1, 35.8, 29.5, 21.1, 252.8, 246.5, 240.2, 231.8, 42.1, 35.8, 29.5, 21.1, 252.8, 246.5, 240.2, 231.8, 42.1, 35.8, 29.5, 21.1, 252.8, 246.5, 240.2, 231.8, 42.1, 35.8, 29.5, 21.1],
	    peglipid = [0.0, 30.1, 60.2, 100.4, 0.0, 30.1, 60.2, 100.4, 0.0, 30.1, 60.2, 100.4, 0.0, 30.1, 60.2, 100.4, 0.0, 30.1, 60.2, 100.4, 0.0, 30.1, 60.2, 100.4, 0.0, 30.1, 60.2, 100.4, 0.0, 30.1, 60.2, 100.4],
	    ethanol = [431.1, 406.0, 380.8, 347.3, 549.5, 524.3, 499.2, 465.6, 397.6, 371.1, 344.5, 309.1, 526.7, 500.2, 473.6, 438.3, 330.5, 301.2, 272.0, 232.9, 481.2, 451.9, 422.6, 383.5, 196.4, 161.6, 126.8, 80.4, 390.1, 355.3, 320.5, 274.1],
	)

	manager = ProtocolManager(protocol_name='lnp_small_molecule', protocol_parameters=parameters)
	manager.spawn_protocol_file()
