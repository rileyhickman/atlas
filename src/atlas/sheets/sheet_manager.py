#!/usr/bin/env python

import os
import time
import gspread
import numpy as np
import pandas as pd

from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner

from atlas import Logger


class SheetManager():

	def __init__(self, config):
		self.config  = config

		if not self.config['sa_filename']:
			Logger.log('Resorting to default service account config in ~/.config/gspread/service_account.json', 'WARNING')
			self.sa = gspread.service_account()
		else:
			if not os.path.exists(self.config['sa_filename']):
				Logger.log('You have not provided a valid service account filename', 'FATAL')
			else:
				self.sa = gspread.service_account(self.config['sa_filename'])

		self.sh = self.sa.open(self.config['sheet_name'])
		self.wks = self.sh.worksheet(self.config['worksheet_name'])


	def read_sheet(self):
		return pd.DataFrame(self.wks.get_all_records())


	def write_sheet(self, df):
		self.wks.update([df.columns.values.tolist()] + df.values.tolist())

	def df_from_campaign(self, campaign, samples):
		''' generate a dataframe from the olympus campaign and
		unmeasured recommendations

		Args:
			campaign (obj): olympus campaign object
			samples (list): list of olympus ParameterVector objects representing the
				umeasured recommendation
		'''

		param_names = [p.name for p in campaign.param_space]
		value_names = [v.name for v in campaign.value_space]
		params = np.array(campaign.observations.get_params())
		values = np.array(campaign.observations.get_values())
		if len(values.shape)==1:
			values = values.reshape(-1, 1)

		data_dict = {p: [] for p in param_names+value_names}
		for param_ix, param in enumerate(param_names):
			if not params.size==0:
				data_dict[param] = params[:, param_ix]
			else:
				pass
		for value_ix, value in enumerate(value_names):
			if not values.size==0:
				data_dict[value] = values[:, value_ix]
			else:
				pass

		data_df = pd.DataFrame(data_dict)

		# print(data_df.shape)
		# print(data_df.head())

		# add the unmeasured recommendations lastly - use TODO
		# for the objective values
		samples_dict = {p: [] for p in param_names+value_names}
		for sample in samples:
			for param_name in param_names:
				samples_dict[param_name].append(sample[param_name])
			for value_name in value_names:
				samples_dict[value_name].append('TODO')

		samples_df = pd.DataFrame(samples_dict)


		# print(samples_df.shape)
		# print(samples_df.head())

		return pd.concat((data_df, samples_df), ignore_index=True)


	def monitor_sheet(self):
		''' check if all the TODOs are replaced by floats or ints
		'''
		exp_finished = False
		df = self.read_sheet()
		counts = 0
		for col in df.columns:
			#counts += df[col].value_counts()['TODO']
			counts += len(df[df[col]=='TODO'])
		with Live(
			Spinner('dots12', text=f'[INFO] Waiting for {counts} measurements ', style="green")
		) as live:
			while not exp_finished:
				df = self.read_sheet()
				counts = 0
				for col in df.columns:
					#counts += df[col].value_counts()['TODO']
					counts += len(df[df[col]=='TODO'])
				if counts == 0:
					exp_finished = True
				else:
					# message = f'Waiting on results for {counts} experiments'
					# Logger.log(message, 'INFO')
					time.sleep(self.config['monitor_interval'])









'''
config should contain
--> path to service account json file
--> sheet name
--> worksheet name
-->
'''
