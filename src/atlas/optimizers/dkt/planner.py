#!/usr/bin/env python

import os, sys
import time
import pickle

import numpy as np
import sobol_seq

import torch
from torch import nn
import gpytorch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models import FixedNoiseGP
from botorch.optim.fit import fit_gpytorch_torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model

# RGPE model related imports
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList

# from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed, optimize_acqf_discrete

from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform


from olympus.planners import CustomPlanner, AbstractPlanner
from olympus import ParameterVector


from atlas.utils import Logger

from atlas.networks.dkt.dkt import DKT

from atlas.optimizers.optimizer_utils import Scaler, flip_source_tasks

from atlas.optimizers.gp.utils import (
	cat_param_to_feat,
	propose_randomly,
	forward_normalize,
	reverse_normalize,
	forward_standardize,
	reverse_standardize,
	infer_problem_type,
	project_to_olymp,
	get_bounds,
	get_cat_dims,
	get_fixed_features_list,
)


from atlas.optimizers.gp.gps import ClassificationGP, CategoricalSingleTaskGP

from atlas.optimizers.gp.acqfs import (
	FeasibilityAwareEI, 
	FeasibilityAwareQEI,
	FeasibilityAwareGeneral, 
	get_batch_initial_conditions, 
	create_available_options,
)


class DKTModel(GP, GPyTorchModel):

	# meta-data for botorch
	_num_outputs = 1

	def __init__(self, model, context_x, context_y):
		super().__init__()
		self.model = model
		self.context_x = context_x
		self.context_y = context_y

	def forward(self, x):
		'''
			x shape  (# proposals, q_batch_size, # params)
			mean shape (# proposals, # params)
			covar shape (# proposals, q_batch_size, # params)
		'''
		_, __, likelihood = self.model.forward(self.context_x, self.context_y, x)
		mean = likelihood.mean
		covar = likelihood.lazy_covariance_matrix

		return gpytorch.distributions.MultivariateNormal(mean, covar)





class DKTPlanner(CustomPlanner, Logger):
	''' Wrapper for deep kernel transfer planner in a closed loop
	optimization setting
	'''

	def __init__(
		self,
		goal='minimize',
		kernel_type='matern',
		warm_start=False,
		transformation='identity',
		model_path='./.tmp_models',
		from_disk=False,
		random_seed=100700,
		batch_size=1,
		num_init_design=5,
		num_proposals=500,
		train_tasks=[],
		valid_tasks=None,
		num_restart_candidates=200,
		param_scaling='identity',
		value_scaling='standardization',
		x_dim=None,
		hyperparams={}
		# **kwargs,
		):
		'''
			Args:
		'''
		AbstractPlanner.__init__(**locals())
		Logger.__init__(self, 'DKTPlanner', verbosity=2)
		self.is_trained = False

		self.goal = goal
		self.kernel_type = kernel_type
		self.warm_start = warm_start
		self.transformation = transformation
		self.model_path = model_path
		self.from_disk = from_disk
		self.random_seed = random_seed
		self.batch_size = batch_size
		self.num_init_design = num_init_design
		self.num_proposals = num_proposals
		self._train_tasks = train_tasks
		self._valid_tasks = valid_tasks

		self.num_restart_candidates = num_restart_candidates
		self.param_scaling = param_scaling
		self.value_scaling = value_scaling
		self.hyperparams = hyperparams
		self._x_dim = x_dim

		# TODO: eventually handle this option
		self.is_moo = False

		# set the random seed
		if random_seed is None:
			self.random_seed = np.random.randint(0, 1e7)
		else:
			self.random_seed = random_seed

		# # NOTE: for maximization, we must flip the signs of the
		# source task values before scaling them
		if self.goal == 'maximize':
			self._train_tasks = flip_source_tasks(self._train_tasks)
			self._valid_tasks = flip_source_tasks(self._valid_tasks)


		# instantiate the scaler
		self.scaler = Scaler(
			param_type=self.param_scaling,
			value_type=self.value_scaling,
		)
		self._train_tasks = self.scaler.fit_transform_tasks(self._train_tasks)
		self._valid_tasks = self.scaler.transform_tasks(self._valid_tasks)


	def _set_param_space(self, param_space):
		''' set the Olympus parameter space (not actually really needed)
		'''

		# infer the problem type
		self.problem_type = infer_problem_type(param_space)

		# make attribute that indicates whether or not we are using descriptors
		# for the categorical variables
		if self.problem_type == 'fully_categorical':
			descriptors = []
			for p in self.param_space:
				descriptors.extend(p.descriptors)
			if all(d is None for d in descriptors):
				self.has_descriptors = False
			else:
				self.has_descriptors = True
		else:
			self.has_descriptors = False



	def _cat_param_to_feat(self, param, val):
		''' convert the option selection of a categorical variable to
		a machine readable feature vector
		Args:
			param (object): the categorical olympus parameter
			val (): the value of the chosen categorical option
		'''
		# get the index of the selected value amongst the options
		arg_val = param.options.index(val)
		if np.all([d==None for d in param.descriptors]):
			# no provided descriptors, resort to one-hot encoding
			#feat = np.array([arg_val])
			feat = np.zeros(len(param.options))
			feat[arg_val] += 1.
		else:
			# we have descriptors, use them as the features
			feat = param.descriptors[arg_val]
		return feat

	def _load_model(self):
		'''
		'''
		if self.x_dim is None:
			# infer the param space
			if self.transformation == 'simpl':
				# if we are using simpl transformation, predictions will be on n+1
				# hypercube
				x_dim = len(self.param_space)+1
			else:
				x_dim = len(self.param_space)
		else:
			# override the x_dim (for categorical vars with descriptors, for instance)
			x_dim = self._x_dim


		self.model = DKT(
			x_dim=x_dim,
			y_dim=1,
			from_disk=self.from_disk,
			model_path=self.model_path,
			hyperparams=self.hyperparams,
		)

	def _meta_train(self):
		''' train the model on the source tasks before commencing the
		target optimization
		'''
		if not hasattr(self, 'model'):
			self._load_model()

		if not self.from_disk:
			# need to meta train
			self.log('DKT model has not been meta-trained! Commencing meta-training procedure', 'WARNING')
			start_time = time.time()
			self.model.train(self._train_tasks, self._valid_tasks)
			training_time = time.time()-start_time
			self.log(f'Meta-training procedure complete in {training_time:.2f} seconds', 'INFO')
		else:
			# already meta trained, load from disk
			self.log(f'Neural process model restored! Skipping meta-training procedure', 'INFO')


	def _tell(self, observations):
		''' register all the current observations
		'''
		self._params = observations.get_params() # string encoding for categorical parameters
		self._values = observations.get_values(
			as_array=True, opposite=self.flip_measurements,
		)
		# make the values 2d if they are not already
		if len(np.array(self._values).shape)==1:
			self._values = np.array(self._values).reshape(-1, 1)



	def _ask(self):
		''' Query the planner for new parameter points to measure
		'''
		iteration = len(self._values) # only valid for  batch_size=1

		# if this is the first iteration, train the source models and
		# randomly select the initial point from the param space

		if len(self._values) < self.num_init_design:
			# need to sample randomly

			# meta train the DKT model
			if not hasattr(self, 'model'):
				self._meta_train()

			_, samples = propose_randomly(1, self.param_space)
			sample = samples[0]
			return_params = ParameterVector().from_array(sample, self.param_space)

		else:
			# we have enough observations to exceed the

			# convert the categorical parmas to ohe vars
			target_params = []
			target_values = []
			for sample_ix, (targ_param, targ_value) in enumerate(zip(self._params, self._values)):
				sample_x = []
				for param_ix, (space_true, element) in enumerate(zip(self.param_space, targ_param)):
					if self.param_space[param_ix].type == 'categorical':
						feat = self._cat_param_to_feat(space_true, element)
						sample_x.extend(feat)
					else:
						sample_x.append(np.float(element))
				target_params.append(sample_x)
				target_values.append(targ_value)
			target_params = np.array(target_params)  # (# target obs, # param dim)
			target_values = np.array(target_values)  # (# target obs, 1)

			# compute the stats of the dataset replacing 0. stds with 1.
			#means_x = [np.mean(target_params[:, ix]) for ix in range(target_params.shape[1])]
			#stds_x = np.array( [np.std(target_params[:, ix]) for ix in range(target_params.shape[1])] )
			#stds_x = np.where(stds_x == 0., 1., stds_x)
			means_y = [np.mean(target_values[:, ix]) for ix in range(target_values.shape[1])]
			stds_y = np.array(  [np.std(target_values[:, ix]) for ix in range(target_values.shape[1])] )
			stds_y = np.where(stds_y == 0., 1., stds_y)

			# scale the target data
			#scaled_x = self.transform(target_params, means_x, stds_x)
			# NOTE: do not need to scale the input parameters for categorical spaces
			scaled_x = target_params
			scaled_y = self.transform(target_values, means_y, stds_y)

			train_x = torch.Tensor(scaled_x)
			train_y = torch.Tensor(scaled_y)

			# get the incumbent point --> always the minimum in olympus
			fbest_scaled = torch.amin(
				train_y[-target_values.shape[0]:]
			)
			maximize=False

			# create the model and acquisition function
			dkt_model = DKTModel(self.model, train_x, train_y)

			bounds = get_bounds(self.param_space, has_descriptors=False)
			choices_feat, choices_cat = None, None

			self.ei = CategoricalEI(
				self.param_space,
				dkt_model,
				best_f=fbest_scaled,
				maximize=maximize,

			)

			if self.problem_type == 'fully_continuous':

				results, _ = optimize_acqf(
					acq_function=self.ei,
					bounds=bounds,
					num_restarts=200,
					q=1,
					raw_samples=1000,
					nonlinear_inequality_constraints=None,
					batch_initial_conditions=None,
				)

			elif self.problem_type == 'mixed':
				results, _ = optimize_acqf_mixed(
					acq_function=self.ei,
					bounds=bounds,
					num_restarts=200,
					q=1,
					raw_samples=1000,
				)

			elif self.problem_type == 'fully_categorical':

				# need to generate the full space of potential choices
				choices_feat, choices_cat = create_available_options(self.param_space, self._params)

				# no need to transform anything here
				results, _ = optimize_acqf_discrete(
					acq_function=self.ei,
					q=1,
					max_batch_size=1000,
					choices=choices_feat.float(),
					unique=True,
				)

			results_np = results.detach().numpy().squeeze(0)

			# if not self.problem_type == 'fully_categorical':
			# 	results_np = self.reverse_transform(results_np, means_y, stds_y)

			# print('results_np : ', results_np )

			# project the sample back to Olympus
			sample = project_to_olymp(
				results_np, self.param_space, has_descriptors=False,
				choices_feat=choices_feat, choices_cat=choices_cat,
			)

			return_params = ParameterVector().from_dict(sample, self.param_space)

			# print('results_np : ', results_np )
			# print('sample :', sample)
			# print('return_params : ', return_params)
			#
			# quit()


		return return_params


		# TODO: try normalization here as oppose to standardization
	def transform(self, data, means, stds):
		''' standardize the data
		'''
		return (data - means) / stds

	def reverse_transform(self, data, means, stds):
		''' un-standardize the data
		'''
		return (data * stds) + means




class CategoricalEI(ExpectedImprovement):
	def __init__(
		self,
		param_space,
		model,
		best_f,
		objective=None,
		maximize=False,
		**kwargs,
	) -> None:
		super().__init__(model, best_f, objective, maximize, **kwargs)
		self._param_space = param_space

	def forward(self, X):
		#X = self.round_to_one_hot(X, self._param_space)
		ei = super().forward(X)
		return ei

	@staticmethod
	def round_to_one_hot(X, param_space):
		'''
		Round all categorical variables to a one-hot encoding
		X shape (# raw_samples, # recommendations, # param dimensions)
		'''
		num_experiments = X.shape[1]
		X = X.clone()
		for q in range(num_experiments):
			c = 0
			for param in param_space:
				if param.type == 'categorical':
					num_options = len(param.options)
					selected_options = X[:, q, c : c + num_options].argmax(axis=1)
					X[:, q, c : c + num_options] = 0
					for j, l in zip(range(X.shape[0]), selected_options):
						X[j, q, int(c + l)] = 1
					check = int(X[:, q, c : c + num_options].sum()) == X.shape[0]
					if not check:
						quit()
					c += num_options
				else:
					# continuous or discrete (??) parameter types
					c += 1
		return X







# DEBUGGING
if __name__ == '__main__':

	from botorch.utils.sampling import draw_sobol_samples
	from botorch.utils.transforms import normalize, unnormalize

	from olympus.objects import (
		ParameterContinuous,
		ParameterDiscrete,
		ParameterCategorical,
	)
	from olympus.campaigns import Campaign, ParameterSpace
	from olympus.surfaces import Surface

	from atlas.utils.synthetic_data import trig_factory

	PARAM_TYPE = 'continuous' #'continuous' # 'mixed', 'categorical'

	PLOT = False

	if PARAM_TYPE == 'continuous':

		def surface(x):
			return np.sin(8*x)

		# define the meta-training tasks
		train_tasks = trig_factory(
			num_samples=20,
			as_numpy=True,

			#scale_range=[[7, 9], [7, 9]],
			#scale_range=[[-9, -7], [-9, -7]],
			scale_range=[[-8.5, -7.5], [7.5, 8.5]],
			shift_range=[-0.02, 0.02],
			amplitude_range=[0.2, 1.2],
		)
		valid_tasks = trig_factory(
			num_samples=5,
			as_numpy=True,
			#scale_range=[[7, 9], [7, 9]],
			#scale_range=[[-9, -7], [-9, -7]],
			scale_range=[[-8.5, -7.5], [7.5, 8.5]],
			shift_range=[-0.02, 0.02],
			amplitude_range=[0.2, 1.2],
		)

		param_space = ParameterSpace()
		# add continuous parameter
		param_0 = ParameterContinuous(name='param_0', low=0., high=1.)
		param_space.add(param_0)

		# variables for optimization
		BUDGET = 50


		planner = DKTPlanner(
			goal='minimize',
			train_tasks=train_tasks,
			valid_tasks=valid_tasks,
			model_path='./tmp_models/',
			from_disk=False,
			hyperparams={'model':{
					'epochs': 2500,
				}
			}
		)

		planner.set_param_space(param_space)


		# make the campaign
		campaign = Campaign()
		campaign.set_param_space(param_space)

		#--------------------------------
		# optional plotting instructions
		#--------------------------------
		if PLOT:
			import matplotlib.pyplot as plt
			import seaborn as sns
			# ground truth, acquisitons
			fig, axes = plt.subplots(1, 3, figsize=(15, 4))
			axes = axes.flatten()
			plt.ion()

			# plotting colors
			COLORS = {
				'ground_truth': '#565676',
				'surrogate': '#A76571',
				'acquisiton': '#c38d94',
				'source_task': '#d8dcff',

			}


		#---------------------
		# Begin the experiment
		#---------------------

		for iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			print(f'ITER : {iter}\tSAMPLES : {samples}')


			
			sample_arr = samples.to_array()
			measurement = surface(
				sample_arr.reshape((1, sample_arr.shape[0]))
			)
			campaign.add_observation(sample_arr, measurement[0])

			# optional plotting
			if PLOT:
				# clear axes
				for ax in axes:
					ax.clear()


				domain = np.linspace(0, 1, 100)

				# plot all the source tasks
				for task_ix, task in enumerate(train_tasks):
					if task_ix == len(train_tasks)-1:
						label='source task'
					else:
						label=None
					axes[0].plot(
						task['params'],
						task['values'],
						lw=0.8, alpha=1.0,
						c=COLORS['source_task'],
						label=label,
					)

				# plot the ground truth function
				axes[0].plot(
					domain, surface(domain),
					ls='--', lw=3,
					c=COLORS['ground_truth'],
					label='ground truth',
				)

				# plot the observations
				params = campaign.observations.get_params()
				values = campaign.observations.get_values()

				# scale the parameters


				if iter > 0:
					# plot the predicitons of the deep gp
					context_x = torch.from_numpy(params).float()
					context_y = torch.from_numpy(values).float()
					target_x = torch.from_numpy(domain.reshape(-1, 1)).float()

					mu, sigma, likelihood = planner.model.forward(
						context_x, context_y, target_x,
					)
					lower, upper = likelihood.confidence_region()
					mu = mu.detach().numpy()


					# plot the neural process predition
					axes[1].plot(
						domain, mu,
						c='k',
						lw=4,
						label='Mnemosyne prediciton',
					)
					axes[1].plot(
						domain, mu,
						c=COLORS['surrogate'],
						lw=3,
						label='DKT prediction',
					)

					axes[1].fill_between(
						domain,
						lower.detach().numpy(),
						upper.detach().numpy(),
						color=COLORS['surrogate'],
						alpha=0.2,
					)

					# plot the acquisition function
					# evalauate the acquisition function for each proposal
					acqs = planner.ei(
						torch.from_numpy(domain.reshape((domain.shape[0], 1, 1))).float()
					).detach().numpy()

					axes[2].plot(
						domain, acqs,
						c='k',
						lw=4,
						label='Mnemosyne prediciton',
					)
					axes[2].plot(
						domain, acqs,
						c=COLORS['acquisiton'],
						lw=3,
						label='DKT prediction',
					)




				for obs_ix, (param, val) in enumerate(zip(params, values)):
					axes[0].plot(
						param[0], val[0],
						marker='o', color='k',
						markersize = 10)
					axes[0].plot(
						param[0], val[0],
						marker='o', color=COLORS['ground_truth'],
						markersize = 7)

				if len(values) >= 1:
					# plot the last observation
					axes[0].plot(
						params[-1][0],
						values[-1][0],
						marker = 'D', color='k',
						markersize = 11,
					)
					axes[0].plot(
						params[-1][0],
						values[-1][0],
						marker = 'D', color=COLORS['ground_truth'],
						markersize = 8,
					)
					# plot horizontal line at last observation location
					axes[0].axvline(params[-1][0], lw=2, ls=':', alpha=0.8)
					axes[1].axvline(params[-1][0], lw=2, ls=':', alpha=0.8)



				axes[0].legend()
				axes[1].legend()
				plt.tight_layout()
				#plt.savefig(f'iter_{iter}.png', dpi=300)
				plt.pause(2)

	elif PARAM_TYPE == 'categorical':

		# test directly the 2d synthetic Olympus tests

		#-----------------------------
		# Instantiate CatDejong surface
		#-----------------------------
		SURFACE_KIND = 'CatCamel'
		surface = Surface(kind=SURFACE_KIND, param_dim=2, num_opts=21)

		tasks = pickle.load(open(f'{__datasets__}/catcamel_2D_tasks.pkl', 'rb'))
		train_tasks = tasks#tasks[:4] # to make the planner scale

		campaign = Campaign()
		campaign.set_param_space(surface.param_space)

		planner = DKTPlanner(
			goal='minimize',
			train_tasks=train_tasks,
			valid_tasks=train_tasks,
			model_path='./tmp_models/',
			from_disk=False,
			num_init_design=5,
			hyperparams={'model':{
					'epochs': 30000,#10000,
				}
			},
			x_dim=42,
		)

		planner.set_param_space(surface.param_space)

		BUDGET=50

		#---------------------
		# begin the experiment
		#---------------------

		for iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			print(f'ITER : {iter}\tSAMPLES : {samples}')
			sample = samples[0]
			sample_arr = sample.to_array()
			measurement = np.array(surface.run(sample_arr))
			campaign.add_observation(sample_arr, measurement[0])


	elif PARAM_TYPE == 'mixed':

		from summit.benchmarks import MIT_case1
		from summit.strategies import LHS
		from summit.utils.dataset import DataSet

		# perform the simulated rxn example
		BUDGET = 20
		NUM_RUNS = 10
		GOAL = 'maximize'

		TARGET_IX = 0  # Case 1
		SOURCE_IX = [1, 2, 3, 4]

		# load the tasks
		tasks = pickle.load(open(f'{__datasets__}/dataset_simulated_rxns/tasks_8.pkl', 'rb'))

		train_tasks = [tasks[i][0] for i in SOURCE_IX]
		valid_tasks = [tasks[i][0] for i in SOURCE_IX]


		#-----------------------
		# build olympus objects
		#-----------------------
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

		planner = DKTPlanner(
			goal='minimize',
			train_tasks=train_tasks,
			valid_tasks=valid_tasks,
			model_path='./tmp_models/',
			from_disk=False,
			num_init_design=2,
			hyperparams={'model':{
					'epochs': 15000,
				}
			},
			x_dim=11,
		)

		planner.set_param_space(param_space)


		# begin experiment
		iteration = 0

		while len(campaign.values) < BUDGET:
			print(f'\nITERATION : {iteration}\n')
			# instantiate summit object for evaluation
			exp_pt = MIT_case1(noise_level=1)
			samples = planner.recommend(campaign.observations)
			print(f'SAMPLES : {samples}')
			for sample in samples:
				# turn into dataframe which summit evaluator expects
				columns = ['conc_cat', 't', 'cat_index', 'temperature']
				values= {
					('conc_cat', 'DATA') : sample['conc_cat'],
					('t', 'DATA'): sample['t'],
					('cat_index', 'DATA'): sample['cat_index'],
					('temperature', 'DATA'): sample['temperature'],
				}
				conditions = DataSet([values], columns=columns)

				exp_pt.run_experiments(conditions)

				measurement = exp_pt.data['y'].values[0]
				print('MEASUREMENT : ', measurement)

				#print(exp_pt.data)

				campaign.add_observation(sample, measurement)

			iteration+=1