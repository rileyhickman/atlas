#!/usr/bin/env python


import os, sys
import time
import pickle
from copy import deepcopy
import numpy as np


import torch
import gpytorch
from botorch.models import SingleTaskGP, MixedSingleTaskGP

from botorch.models.kernels.categorical import CategoricalKernel

from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf, optimize_acqf_mixed, optimize_acqf_discrete
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qNoisyExpectedImprovement

import olympus
from olympus.planners import CustomPlanner, AbstractPlanner
from olympus import ParameterVector
from olympus.scalarizers import Scalarizer
from olympus.planners import Planner

from gpytorch.mlls import ExactMarginalLogLikelihood

from atlas import Logger
from atlas.optimizers.acquisition_optimizers.base_optimizer import AcquisitionOptimizer


from atlas.optimizers.utils import (
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

from atlas.optimizers.gps import ClassificationGPMatern, CategoricalSingleTaskGP

from atlas.optimizers.acqfs import (
	FeasibilityAwareEI,
	FeasibilityAwareQEI,
	FeasibilityAwareGeneral,
	get_batch_initial_conditions,
	create_available_options,
)

from atlas.optimizers.base import BasePlanner



class BoTorchPlanner(BasePlanner):
	''' Wrapper for GP-based Bayesiam optimization with BoTorch
	Args:
		goal (str): the optimization goal, "maximize" or "minimize"
		feas_strategy (str): feasibility acqusition function name
		feas_param (float): feasibilty parameter
		batch_size (int): number of samples to measure per batch (will be fixed at 1 for now)
		random_seed (int): the random seed to use
		num_initial_design (int): number of points to sample using the initial
			design strategy
		init_design_strategy (str): the inital design strategy, "random" or "sobol"
		vgp_iters (int): number of training iterations for the variational GP
		vgp_lr (float): learning rate for the variational optimization procedure
		max_jitter (float):
		cla_threshold (float): classification threshold for the predictions of the
			feasibilty surrogate
		known_constraints (callable): callable which takes parameters and returns boolean
			corresponding to the feaibility of that experiment (True-->feasible, False-->infeasible)
		general_parameters (list): list of parameter indices for which we average the objective
			function over
		is_moo (bool): whether or not we have a multiobjective optimization problem
	'''

	def __init__(
		self,
		goal='minimize',
		feas_strategy='naive-0',
		feas_param=0.2,
		batch_size=1,
		random_seed=None,
		num_init_design=5,
		init_design_strategy='random',
		acquisition_optimizer_kind='gradient', # gradient, genetic
		vgp_iters=1000,
		vgp_lr=0.1,
		max_jitter=1e-1,
		cla_threshold=0.5,
		known_constraints=None,
		general_parmeters=None,
		is_moo=False,
		value_space=None,
		scalarizer_kind='Hypervolume',
		moo_params={},
		goals=None,
		**kwargs,
	):
		local_args = {key:val for key, val in locals().items() if key != 'self'}
		super().__init__(**local_args)


	def build_train_regression_gp(self, train_x, train_y):
		''' Build the regression GP model and likelihood
		'''
		# infer the model based on the parameter types
		if self.problem_type == 'fully_continuous':
			model = SingleTaskGP(train_x, train_y)
		elif self.problem_type in ['mixed']:
			# TODO: implement a method to retrieve the categorical dimensions
			cat_dims = get_cat_dims(self.param_space)
			model = MixedSingleTaskGP(train_x, train_y, cat_dims=cat_dims)
		elif self.problem_type == 'fully_categorical':
			if self.has_descriptors:
				# we have some descriptors, use the Matern kernel
				model = SingleTaskGP(train_x, train_y)
			else:
				# if we have no descriptors, use a Categorical kernel
				# based on the HammingDistance
				model = CategoricalSingleTaskGP(train_x, train_y)
		elif self.problem_type == 'mixed_dis_cat':
			if self.has_descriptors:
				# we have some descriptors, use the Matern kernel
				model = SingleTaskGP(train_x, train_y)
			else:
				cat_dims = get_cat_dims(self.param_space)
				model = MixedSingleTaskGP(train_x, train_y, cat_dims=cat_dims)

		else:
			raise NotImplementedError

		mll = ExactMarginalLogLikelihood(model.likelihood, model)
		# fit the GP
		start_time = time.time()
		with gpytorch.settings.cholesky_jitter(self.max_jitter):
			fit_gpytorch_model(mll)
		gp_train_time = time.time() - start_time
		print(f' >>> [? epochs] GP trained in {round(gp_train_time,3)} sec ')

		return model


	def _ask(self):
		''' query the planner for a batch of new parameter points to measure
		'''
		# if we have all nan values, just keep randomly sampling
		if np.logical_or(
			len(self._values) < self.num_init_design,
			np.all(np.isnan(self._values))
		):
			# set parameter space for the initial design planner
			self.init_design_planner.set_param_space(self.param_space)

			# sample using initial design strategy (with same batch size)
			return_params = []
			for _ in range(self.batch_size):
				# TODO: this is pretty sloppy - consider standardizing this
				if self.init_design_strategy == 'random':
					self.init_design_planner._tell(iteration=self.num_init_design_completed)
				else:
					self.init_design_planner.tell()
				rec_params = self.init_design_planner.ask()
				if isinstance(rec_params, list):
					return_params.append(rec_params[0])
				elif isinstance(rec_params, ParameterVector):
					return_params.append(rec_params)
				else:
					raise TypeError
				self.num_init_design_completed += 1 # batch_size always 1 for init design planner
		else:
			# use GP surrogate to propose the samples
			# get the scaled parameters and values for both the regression and classification data
			self.train_x_scaled_cla, self.train_y_scaled_cla, self.train_x_scaled_reg, self.train_y_scaled_reg = self.build_train_data()

			use_p_feas_only = False
			# check to see if we are using the naive approaches
			if 'naive-' in self.feas_strategy:
				infeas_ix = torch.where(self.train_y_scaled_cla==1.)[0]
				feas_ix = torch.where(self.train_y_scaled_cla==0.)[0]
				# checking if we have at least one objective function measurement
				#  and at least one infeasible point (i.e. at least one point to replace)
				if np.logical_and(
					self.train_y_scaled_reg.size(0) >= 1,
					infeas_ix.shape[0] >= 1
				):
					if self.feas_strategy == 'naive-replace':
						# NOTE: check to see if we have a trained regression surrogate model
						# if not, wait for the following iteration to make replacements
						if hasattr(self, 'reg_model'):
							# if we have a trained regression model, go ahead and make replacement
							new_train_y_scaled_reg = deepcopy(self.train_y_scaled_cla).double()

							input = self.train_x_scaled_cla[infeas_ix].double()

							posterior = self.reg_model.posterior(X=input)
							pred_mu = posterior.mean.detach()

							new_train_y_scaled_reg[infeas_ix] = pred_mu.squeeze(-1)
							new_train_y_scaled_reg[feas_ix] = self.train_y_scaled_reg.squeeze(-1)

							self.train_x_scaled_reg = deepcopy(self.train_x_scaled_cla).double()
							self.train_y_scaled_reg = new_train_y_scaled_reg.view(self.train_y_scaled_cla.size(0), 1).double()

						else:
							use_p_feas_only = True

					elif self.feas_strategy == 'naive-0':
						new_train_y_scaled_reg = deepcopy(self.train_y_scaled_cla).double()

						worst_obj = torch.amax(self.train_y_scaled_reg[~self.train_y_scaled_reg.isnan()])

						to_replace = torch.ones(infeas_ix.size())*worst_obj

						new_train_y_scaled_reg[infeas_ix] = to_replace.double()
						new_train_y_scaled_reg[feas_ix] = self.train_y_scaled_reg.squeeze()

						self.train_x_scaled_reg = self.train_x_scaled_cla.double()
						self.train_y_scaled_reg = new_train_y_scaled_reg.view(self.train_y_scaled_cla.size(0), 1)

					else:
						raise NotImplementedError
				else:
					# if we are not able to use the naive strategies, propose randomly
					# do nothing at all and use the feasibilty surrogate as the acquisition
					use_p_feas_only = True

			# builds and fits the regression surrogate model
			self.reg_model = self.build_train_regression_gp(self.train_x_scaled_reg, self.train_y_scaled_reg)

			if not 'naive-' in self.feas_strategy:
				# build and train the classification surrogate model
				self.cla_model, self.cla_likelihood = self.build_train_classification_gp(self.train_x_scaled_cla, self.train_y_scaled_cla)

				self.cla_model.eval()
				self.cla_likelihood.eval()

			else:
				self.cla_model, self.cla_likelihood = None, None

			# get the incumbent point
			f_best_argmin = torch.argmin(self.train_y_scaled_reg)

			f_best_scaled = self.train_y_scaled_reg[f_best_argmin][0].float()


			# compute the ratio of infeasible to total points
			infeas_ratio = (torch.sum(self.train_y_scaled_cla) / self.train_x_scaled_cla.size(0)).item()
			# get the approximate max and min of the acquisition function without the feasibility contribution
			acqf_min_max = self.get_aqcf_min_max(self.reg_model, f_best_scaled)


			if self.batch_size == 1:
				self.acqf = FeasibilityAwareEI(
					self.reg_model, self.cla_model, self.cla_likelihood,
					self.param_space, f_best_scaled,
					self.feas_strategy, self.feas_param, infeas_ratio, acqf_min_max,
					)
			elif self.batch_size > 1:
				#self.acqf = FeasibilityAwareQEI(
				self.acqf = FeasibilityAwareEI(
					self.reg_model, self.cla_model, self.cla_likelihood,
					self.param_space, f_best_scaled,
					self.feas_strategy, self.feas_param, infeas_ratio, acqf_min_max
				)


			bounds = get_bounds(self.param_space, self._mins_x, self._maxs_x, self.has_descriptors)



			print('PROBLEM TYPE : ', self.problem_type)

			#-------------------------------
			# optimize acquisition function
			#-------------------------------

			acquisition_optimizer = AcquisitionOptimizer(
				self.acquisition_optimizer_kind,
				self.param_space,
				self.acqf,
				bounds,
				self.known_constraints,
				self.batch_size,
				self.feas_strategy,
				self.fca_constraint,
				self.has_descriptors,
				self._params,
				self._mins_x,
				self._maxs_x,
			)
			return_params = acquisition_optimizer.optimize()

		return return_params


	def get_aqcf_min_max(self, reg_model, f_best_scaled, num_samples=2000):
		''' computes the min and max value of the acquisition function without
		the feasibility contribution. These values will be used to approximately
		normalize the acquisition function
		'''
		if self.batch_size == 1:
			acqf = ExpectedImprovement(reg_model, f_best_scaled, objective=None, maximize=False)
		elif self.batch_size > 1:
			acqf = qExpectedImprovement(reg_model, f_best_scaled, objective=None, maximize=False)
		samples, _ = propose_randomly(num_samples, self.param_space)
		if not self.problem_type=='fully_categorical' and not self.has_descriptors:
			# we dont scale the parameters if we have a one-hot-encoded representation
			samples = forward_normalize(samples, self._mins_x, self._maxs_x)

		acqf_vals = acqf(
			torch.tensor(samples).view(samples.shape[0], 1, samples.shape[-1]).double()
		)
		min_ = torch.amin(acqf_vals).item()
		max_ = torch.amax(acqf_vals).item()

		if np.abs( max_ - min_ ) < 1e-6:
			max_ = 1.0
			min_ = 0.0

		return min_, max_



# TODO:
'''
- test botorch planner with
	- continuous-valued 2d surfaces from the paper
	- categorical-valued 2d surfaces form paper
	- perovskite design example
	- drug design example
- implement feasibility aware strategies
	- (N) replace nan values with worst merit seen so far
	- (N) replace nan values with the objective prediction at current point
	- (FCA) feasibility constrained strategy with various parameters for t
	- (FIA) feasibility interpolated strategy with various parameters for t
	-
- edge cases/improvements
	- only train the VGP feasibility surrogate when we have at least one infeasible point
	- only train the regression surrogate when we have at least one feasible point (objective
		function measurement)
	- blah blah blah
'''


#==============
# DEBUGGING
#==============

if __name__ == '__main__':

	PARAM_TYPE = 'mixed_general'

	PLOT = False

	NUM_RUNS = 40
	SEED = 100700

	from olympus.objects import (
		ParameterContinuous,
		ParameterDiscrete,
		ParameterCategorical,
	)
	from olympus.campaigns import Campaign, ParameterSpace
	from olympus.surfaces import Surface

	# optional plotting instructions
	if PLOT:
		import matplotlib.pyplot as plt
		import seaborn as sns
		# ground truth, acquisitions
		fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
		axes = axes.flatten()
		plt.ion()

		# plotting colors
		COLORS = {
			'ground_truth': '#565676',
			'reg_surrogate': '#A76571',
			'cla_surrogate': '#c38d94',
			'acqf': '#c97ea7',
		}


	def surface(x):
		return np.sin(8*x)

	if PARAM_TYPE == 'continuous_nan':

		#	np.random.seed(SEED)

		def constrained_surface(sample_arr):
			if 0.1 < sample_arr[0] < 0.35:
				return np.nan
			elif 0.7 < sample_arr[0] < 0.9:
				return np.nan
			else:
				return np.sin(8*sample_arr[0])

		from olympus.planners import Planner
		param_space = ParameterSpace()
		param_0 = ParameterContinuous(name='param_0', low=0.0, high=1.0)
		param_space.add(param_0)

		#planner = Planner(kind='RandomSearch')
		planner = BoTorchPlanner(
			goal='minimize',
			feas_strategy='fca',
			feas_param=0.5,
		)
		planner.set_param_space(param_space)

		campaign = Campaign()
		campaign.set_param_space(param_space)

		BUDGET = 24

		for num_iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			for sample in samples:
				sample_arr = sample.to_array()
				measurement = constrained_surface(sample_arr)
				campaign.add_observation(sample_arr, measurement)
			print(f'ITER : {num_iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}')


			if PLOT and num_iter > 4:
				# clear axes
				for ax in axes:
					ax.clear()

				# define the domain
				domain = np.linspace(0, 1, 100)
				# normalize the domaim
				domain_norm = forward_normalize(domain, planner._mins_x, planner._maxs_x)

				torch_domain_reg = torch.tensor(domain_norm.reshape(-1, 1)).double()
				torch_domain_cla = torch.tensor(domain_norm).float()
				torch_domain_acqf = torch.tensor(domain_norm).float().view(domain_norm.shape[0], 1, 1)


				surf_vals = np.array([constrained_surface(np.array([x])) for x in domain])

				# shade the constrained regions
				axes[0].fill_between(
					domain,
					np.amin(surface(domain)),
					np.amax(surface(domain)),
					where=np.isnan(surf_vals),
					alpha=0.3,
					color='gray'
				)


				# plot the ground truth function
				axes[0].plot(
					domain, surface(domain),
					ls='--', lw=3,
					c=COLORS['ground_truth'],
					label='ground truth',
				)
				params = campaign.observations.get_params()
				values  = campaign.observations.get_values()

				for obs_ix, (param, val) in enumerate(zip(params, values)):
					if np.isnan(val):
						axes[0].plot(
							param[0], -1,
							marker='x', color='k',
							markersize = 10)
					else:
						axes[0].plot(
							param[0], val,
							marker='o', color='k',
							markersize = 10)
						axes[0].plot(
							param[0], val,
							marker='o', color=COLORS['ground_truth'],
							markersize = 7)

				if len(values) >= 1:
					# plot the last observation
					axes[0].plot(
						params[-1][0],
						values[-1],
						marker = 'D', color='k',
						markersize = 11,
					)
					axes[0].plot(
						params[-1][0],
						values[-1],
						marker = 'D', color=COLORS['ground_truth'],
						markersize = 8,
					)
					# plot horizontal line at last observation location
					axes[0].axvline(params[-1][0], lw=2, ls=':', alpha=0.8)
					#axes[1].axvline(params[-1][0], lw=2, ls=':', alpha=0.8)




				# plot the regression surrogate and classification surrogate

				# predict the regression surrogate
				reg_surr = planner.reg_model(torch_domain_reg)

				reg_mu = reg_surr.mean.detach().numpy()
				reg_std = np.sqrt(reg_surr.variance.detach().numpy())

				# unstandardize the reg surrogate predictions
				reg_mu = reverse_standardize(reg_mu, planner._means_y, planner._stds_y)

				axes[1].plot(domain, reg_mu, lw=3, color=COLORS['reg_surrogate'], label='reg surrogate')
				axes[1].fill_between(
					domain,
					reg_mu+1.96*reg_std,
					reg_mu-1.96*reg_std,
					alpha=0.2,
					color=COLORS['reg_surrogate'],
				)

				# predict the classification surrogate, P(feasible|X)
				p_feas = planner.cla_likelihood(planner.cla_model(torch_domain_cla)).mean.detach().numpy()

				axes[2].plot(domain, p_feas, lw=3, c=COLORS['cla_surrogate'], label = 'cla surrogate')


				# plot the acquisition function
				acqf = planner.acqf(torch_domain_acqf).detach().numpy()

				# # unstandardize the acquisition function
				# acqf = reverse_standardize(acqf, planner._means_y, planner._stds_y)


				axes[3].plot(domain, acqf, lw=3, c=COLORS['acqf'], label='acquisition function')


				axes[0].set_ylabel('measurement')
				axes[1].set_ylabel('regression surrogate value')
				axes[2].set_ylabel(r'$P(feasible|x)$')
				axes[2].set_xlabel('parameter')
				axes[3].set_ylabel('feasibility-aware acquisition value')
				axes[3].set_xlabel('parameter')


				axes[2].axhline(0.5, lw=3, alpha=0.4, ls=':', c='k')
				axes[2].set_ylim(0, 1)


				plt.tight_layout()
				plt.pause(2.0)


	elif PARAM_TYPE == 'categorical_nan':

		sys.path.append('../benchmarks_unknown/')

		from benchmark_functions import CatDejongConstr as BenchmarkSurface
		#from benchmark_functions import CatMichalewiczConstr as BenchmarkSurface

		surface = BenchmarkSurface()

		param_space = ParameterSpace()

		x0 = ParameterCategorical(
			name='x0',
			options=[f'x_{i}' for i in range(21)],
			#descriptors=[None for _ in range(21)],
			descriptors=[[float(i), float(i)] for i in range(21)]
		)
		param_space.add(x0)

		x1 = ParameterCategorical(
			name='x1',
			options=[f'x_{i}' for i in range(21)],
			#descriptors=[None for _ in range(21)],
			descriptors=[[float(i), float(i)] for i in range(21)]
		)
		param_space.add(x1)

		planner = BoTorchPlanner(
			goal='minimize',
			feas_strategy='fca',
			feas_param=0.5,
		)
		planner.set_param_space(param_space)

		campaign = Campaign()
		campaign.set_param_space(param_space)

		OPT = surface.best
		print('OPTIMUM : ', OPT)

		BUDGET = 442

		for iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			sample = samples[0].to_dict()
			measurement = surface.eval_merit(sample)

			print(f'ITER : {iter}\tSAMPLES : {samples}\tMEASUREMENT : {measurement["obj"]}')
			campaign.add_observation(samples[0].to_array(), measurement['obj'])

			if (sample['x0'], sample['x1']) == OPT:
				print(f'FOUND OPTIMUM AFTER {iter+1} ITERATIONS!')
				break


	elif PARAM_TYPE == 'continuous_moo_nan':

		def surface1(x):
			return np.sin(8*x)

		def surface2(x):
			return 0.6*np.cos(10*x) + 0.8

		def constrained_surface1(sample_arr):
			if 0.1 < sample_arr[0] < 0.35:
				return np.nan
			elif 0.7 < sample_arr[0] < 0.9:
				return np.nan
			else:
				return surface1(sample_arr[0])

		def constrained_surface2(sample_arr):
			if 0.1 < sample_arr[0] < 0.35:
				return np.nan
			elif 0.7 < sample_arr[0] < 0.9:
				return np.nan
			else:
				return surface2(sample_arr[0])

		from olympus.planners import Planner
		param_space = ParameterSpace()
		param_0 = ParameterContinuous(name='param_0', low=0.0, high=1.0)
		param_space.add(param_0)


		tolerances = [-0.1, 0.75]
		absolutes = [True, True]
		goals = ['min', 'min']

		#planner = Planner(kind='RandomSearch')
		planner = BoTorchPlanner(
			goal='minimize',
			feas_strategy='fwa',
			feas_param=0.5,
			is_moo=True,
			tolerances=tolerances,
			absolutes=absolutes,
			goals=goals,
		)
		planner.set_param_space(param_space)

		campaign1 = Campaign()
		campaign1.set_param_space(param_space)

		campaign2 = Campaign()
		campaign2.set_param_space(param_space)

		BUDGET = 24

		# domain = np.linspace(0, 1, 500)
		# import matplotlib.pyplot as plt
		# import seaborn as sns

		# fig, ax = plt.subplots(1, 1)
		# ax.plot(domain, surface1(domain))
		# ax.plot(domain, surface2(domain))
		# plt.tight_layout()
		# plt.show()

		observations = []

		# begin the experiment
		for num_iter in range(BUDGET):

			samples = planner.recommend([campaign1.observations, campaign2.observations])
			for sample in samples:
				sample_arr = sample.to_array()
				measurement1 = constrained_surface1(sample_arr)
				measurement2 = constrained_surface2(sample_arr)

				campaign1.add_observation(sample_arr, measurement1)
				campaign2.add_observation(sample_arr, measurement2)

			print(f'ITER : {num_iter}\tSAMPLES : {samples}\t MEASUREMENT 1 : {measurement1}\t MEASUREMENT 2: {measurement2}')


	elif PARAM_TYPE == 'categorical_moo_nan':

		sys.path.append('../benchmarks_unknown/')

		from benchmark_functions import CatDejongConstr
		from benchmark_functions import CatMichalewiczConstr

		surface1 = CatDejongConstr()
		surface2 = CatDejongConstr() # is flipped later
		#surface2 = CatMichalewiczConstr()

		param_space = ParameterSpace()

		x0 = ParameterCategorical(
			name='x0',
			options=[f'x_{i}' for i in range(21)],
			descriptors=[None for _ in range(21)],
			#descriptors=[[float(i), float(i)] for i in range(21)]
		)
		param_space.add(x0)

		x1 = ParameterCategorical(
			name='x1',
			options=[f'x_{i}' for i in range(21)],
			descriptors=[None for _ in range(21)],
			#descriptors=[[float(i), float(i)] for i in range(21)]
		)
		param_space.add(x1)

		tolerances = [20., 0.6]
		absolutes = [True, False]
		goals = ['min', 'min']

		planner = BoTorchPlanner(
			goal='minimize',
			feas_strategy='fwa',
			feas_param=0.5,
			is_moo=True,
			tolerances=tolerances,
			absolutes=absolutes,
			goals=goals
		)
		planner.set_param_space(param_space)

		campaign1 = Campaign()
		campaign1.set_param_space(param_space)

		campaign2 = Campaign()
		campaign2.set_param_space(param_space)

		BUDGET = 442

		for iter in range(BUDGET):

			samples = planner.recommend([campaign1.observations, campaign2.observations])
			for sample in samples:
				measurement1 = surface1.eval_merit(sample.to_dict())
				measurement2 = surface2.eval_merit(sample.to_dict())

			campaign1.add_observation(samples[0].to_array(), measurement1['obj'])
			campaign2.add_observation(samples[0].to_array(), -measurement2['obj'])

			# samples = planner.recommend(campaign.observations)
			# sample = samples[0].to_dict()
			# measurement = surface.eval_merit(sample)

			print(f'ITER : {iter}\tSAMPLES : {samples}\tMEASUREMENT : {measurement1["obj"]}\tMEASUREMENT : {measurement2["obj"]}')


	elif PARAM_TYPE == 'continuous':
		param_space = ParameterSpace()
		param_0 = ParameterContinuous(name='param_0', low=0.0, high=1.0)
		param_space.add(param_0)

		planner = BoTorchPlanner(goal='minimize')
		planner.set_param_space(param_space)

		campaign = Campaign()
		campaign.set_param_space(param_space)

		BUDGET = 24


		for num_iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			print(f'ITER : {num_iter}\tSAMPLES : {samples}')
			for sample in samples:
				sample_arr = sample.to_array()
				measurement = surface(
					sample_arr.reshape((1, sample_arr.shape[0]))
				)
				campaign.add_observation(sample_arr, measurement[0])

	elif PARAM_TYPE == 'continuous_nan_constr':

		def constrained_surface(sample_arr):
			if 0.1 < sample_arr[0] < 0.35:
				return np.nan
			elif 0.7 < sample_arr[0] < 0.9:
				return np.nan
			else:
				return np.sin(8*sample_arr[0])

		def surface(x):
			return np.sin(8*x)

		param_space = ParameterSpace()
		param_0 = ParameterContinuous(name='param_0', low=0.0, high=1.0)
		param_space.add(param_0)

		def known_constraints(x):
			return x*0.5 #torch.rand(x.shape[0]) - 0.05

		planner = BoTorchPlanner(
			goal='minimize',
			known_constraints=known_constraints,
		)

		planner.set_param_space(param_space)

		campaign = Campaign()
		campaign.set_param_space(param_space)

		BUDGET = 24


		for num_iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			print(f'ITER : {num_iter}\tSAMPLES : {samples}')
			for sample in samples:
				sample_arr = sample.to_array()
				measurement = constrained_surface(sample_arr)
				#measurement = surface(sample_arr)
				campaign.add_observation(sample_arr, measurement)


	elif PARAM_TYPE == 'categorical':

		surface_kind = 'CatDejong'
		surface = Surface(kind=surface_kind, param_dim=2, num_opts=21)

		campaign = Campaign()
		campaign.set_param_space(surface.param_space)

		planner = BoTorchPlanner(goal='minimize')
		planner.set_param_space(surface.param_space)

		OPT = ['x10', 'x10']

		BUDGET = 442

		for iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			print(f'ITER : {iter}\tSAMPLES : {samples}')
			sample = samples[0]
			sample_arr = sample.to_array()
			measurement = np.array(surface.run(sample_arr))
			campaign.add_observation(sample_arr, measurement[0])

			if [sample_arr[0], sample_arr[1]] == OPT:
				print(f'FOUND OPTIMUM AFTER {iter+1} ITERATIONS!')
				break


	elif PARAM_TYPE == 'suzuki':

		from olympus.emulators import Emulator
		from olympus.datasets import Dataset
		from olympus.planners import Planner
		from olympus import Database

		all_campaigns = []

		# load the Olympus emulator
		dataset = Dataset(kind='suzuki_i')
		emul = Emulator(dataset=dataset, model='BayesNeuralNet')



		for i in range(NUM_RUNS):
			planner = BoTorchPlanner(goal='maximize')
			planner.set_param_space(dataset.param_space)

			print(dataset.param_space)

			campaign = Campaign()
			campaign.set_param_space(dataset.param_space)

			BUDGET = 25

			for iter in range(BUDGET):

				samples = planner.recommend(campaign.observations)
				print(samples)
				# sample_arr = samples.to_array()
				measurement = emul.run(samples)
				print(measurement)
				print(f'ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement[0][0]}')
				campaign.add_observation(sample_arr, measurement[0][0])

			all_campaigns.append(campaign)

		pickle.dump(all_campaigns, open('results/suzuki_botorch.pkl', 'wb'))


	elif PARAM_TYPE == 'mixed':

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

		planner = BoTorchPlanner(goal='maximize')
		planner.set_param_space(param_space)


		BUDGET = 24

		def mock_yield(x):
			return np.random.uniform()*100

		for iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			sample_arr = samples.to_array()
			measurement = mock_yield(samples)
			print(f'ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}')
			campaign.add_observation(sample_arr, measurement)


	elif PARAM_TYPE == 'mixed_general':

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
			general_parmeters=[0],
		)
		planner.set_param_space(param_space)


		BUDGET = 24

		def mock_yield(x):
			return np.random.uniform()*100

		for iter in range(BUDGET):

			samples = planner.recommend(campaign.observations)
			sample_arr = samples.to_array()
			measurement = mock_yield(samples)
			print(f'ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}')
			campaign.add_observation(sample_arr, measurement)


	elif PARAM_TYPE == 'suzuki_random':

		from olympus.emulators import Emulator
		from olympus.datasets import Dataset
		from olympus.planners import Planner
		from olympus import Database

		all_campaigns = []

		# load the Olympus emulator
		emul = Emulator(dataset='suzuki', model='BayesNeuralNet')

		dataset = Dataset(kind='suzuki')

		for i in range(NUM_RUNS):

			planner = Planner(kind='RandomSearch')
			planner.set_param_space(dataset.param_space)

			campaign = Campaign()
			campaign.set_param_space(dataset.param_space)

			BUDGET = 25

			for iter in range(BUDGET):

				samples = planner.recommend(campaign.observations)
				sample_arr = samples[0].to_array()
				measurement = emul.run(sample_arr)
				print(f'ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement[0][0]}')
				campaign.add_observation(sample_arr, measurement[0][0])

			all_campaigns.append(campaign)

		pickle.dump(all_campaigns, open('results/suzuki_random.pkl', 'wb'))


	elif PARAM_TYPE == 'perovskites':
		# load in the perovskites dataset
		lookup_df = pickle.load(open('datasets_emulators/perovskites/perovskites.pkl', 'rb'))

		# make a function for measuring the perovskite bandgap
		def measure(param):
			''' lookup the HSEO6 bandgap for given perovskite component
			'''
			match = lookup_df.loc[
							(lookup_df.organic == param['organic']) &
							(lookup_df.anion == param['anion']) &
							(lookup_df.cation == param['cation'])
						]
			assert len(match)==1
			bandgap = match.loc[:, 'hse06'].to_numpy()[0]
			return bandgap

		all_campaigns = []

		for i in range(NUM_RUNS):

			# build the experiment
			organic_options = lookup_df.organic.unique().tolist()
			anion_options = lookup_df.anion.unique().tolist()
			cation_options = lookup_df.cation.unique().tolist()

			# make the parameter space
			param_space = ParameterSpace()

			organic_param = ParameterCategorical(
				name='organic',
				options=organic_options,
				descriptors=[None for _ in organic_options],
			)
			param_space.add(organic_param)

			anion_param = ParameterCategorical(
				name='anion',
				options=anion_options,
				descriptors=[None for _ in anion_options],
			)
			param_space.add(anion_param)

			cation_param = ParameterCategorical(
				name='cation',
				options=cation_options,
				descriptors=[None for _ in cation_options],
			)
			param_space.add(cation_param)

			planner = BoTorchPlanner(goal='minimize')
			planner.set_param_space(param_space)

			campaign = Campaign()
			campaign.set_param_space(param_space)

			BUDGET = 192

			OPT = ['hydrazinium', 'I', 'Sn'] # value = 1.5249 eV

			for iter in range(BUDGET):
				samples = planner.recommend(campaign.observations)
				measurement = measure(samples[0])
				print(f'ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}')
				campaign.add_observation(samples[0], measurement)

				# check for convergence
				if [samples[0]['organic'], samples[0]['anion'], samples[0]['cation']] == OPT:
					print(f'FOUND OPTIMUM AFTER {iter+1} ITERATIONS!')
					break
			all_campaigns.append(campaign)

		pickle.dump(all_campaigns, open('results/perovskites_botorch_naive.pkl', 'wb'))


	elif PARAM_TYPE == 'perovskites_random':
		# load in the perovskites dataset
		lookup_df = pickle.load(open('datasets_emulators/perovskites/perovskites.pkl', 'rb'))

		from olympus.planners import Planner

		# make a function for measuring the perovskite bandgap
		def measure(param):
			''' lookup the HSEO6 bandgap for given perovskite component
			'''
			match = lookup_df.loc[
							(lookup_df.organic == param['organic']) &
							(lookup_df.anion == param['anion']) &
							(lookup_df.cation == param['cation'])
						]
			assert len(match)==1
			bandgap = match.loc[:, 'hse06'].to_numpy()[0]
			return bandgap

		all_campaigns = []

		for i in range(NUM_RUNS):

			# build the experiment
			organic_options = lookup_df.organic.unique().tolist()
			anion_options = lookup_df.anion.unique().tolist()
			cation_options = lookup_df.cation.unique().tolist()

			# make the parameter space
			param_space = ParameterSpace()

			organic_param = ParameterCategorical(
				name='organic',
				options=organic_options,
				descriptors=[None for _ in organic_options],
			)
			param_space.add(organic_param)

			anion_param = ParameterCategorical(
				name='anion',
				options=anion_options,
				descriptors=[None for _ in anion_options],
			)
			param_space.add(anion_param)

			cation_param = ParameterCategorical(
				name='cation',
				options=cation_options,
				descriptors=[None for _ in cation_options],
			)
			param_space.add(cation_param)


			planner = Planner(kind='RandomSearch')
			planner.set_param_space(param_space)

			campaign = Campaign()
			campaign.set_param_space(param_space)

			BUDGET = 192

			OPT = ['hydrazinium', 'I', 'Sn'] # value = 1.5249 eV

			for iter in range(BUDGET):
				samples = planner.recommend(campaign.observations)
				measurement = measure(samples[0])
				print(f'ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}')
				campaign.add_observation(samples[0], measurement)

				# check for convergence
				if [samples[0]['organic'], samples[0]['anion'], samples[0]['cation']] == OPT:
					print(f'FOUND OPTIMUM AFTER {iter+1} ITERATIONS!')
					break
			all_campaigns.append(campaign)

		pickle.dump(all_campaigns, open('results/perovskites_random.pkl', 'wb'))


	elif PARAM_TYPE == 'perovskites_descriptors':

		# load in the perovskites dataset
		lookup_df = pickle.load(open('datasets_emulators/perovskites/perovskites.pkl', 'rb'))

		# make a function for measuring the perovskite bandgap
		def measure(param):
			''' lookup the HSEO6 bandgap for given perovskite component
			'''
			match = lookup_df.loc[
							(lookup_df.organic == param['organic']) &
							(lookup_df.anion == param['anion']) &
							(lookup_df.cation == param['cation'])
						]
			assert len(match)==1
			bandgap = match.loc[:, 'hse06'].to_numpy()[0]
			return bandgap

		def get_descriptors(element, kind):
			''' retrive the descriptors for a given element
			'''
			return lookup_df.loc[(lookup_df[kind]==element)].loc[:, lookup_df.columns.str.startswith(f'{kind}-')].values[0].tolist()

		all_campaigns = []
		for i in range(NUM_RUNS):
			# build the experiment
			organic_options = lookup_df.organic.unique().tolist()
			anion_options = lookup_df.anion.unique().tolist()
			cation_options = lookup_df.cation.unique().tolist()

			# make the parameter space
			param_space = ParameterSpace()

			organic_param = ParameterCategorical(
				name='organic',
				options=organic_options,
				descriptors=[get_descriptors(option, 'organic') for option in organic_options],
			)
			param_space.add(organic_param)

			anion_param = ParameterCategorical(
				name='anion',
				options=anion_options,
				descriptors=[get_descriptors(option, 'anion') for option in anion_options],
			)
			param_space.add(anion_param)

			cation_param = ParameterCategorical(
				name='cation',
				options=cation_options,
				descriptors=[get_descriptors(option, 'cation') for option in cation_options],
			)
			param_space.add(cation_param)

			planner = BoTorchPlanner(goal='minimize', num_init_design=10)
			planner.set_param_space(param_space)

			campaign = Campaign()
			campaign.set_param_space(param_space)

			BUDGET = 192

			OPT = ['hydrazinium', 'I', 'Sn'] # value = 1.5249 eV

			for iter in range(BUDGET):
				samples = planner.recommend(campaign.observations)
				measurement = measure(samples[0])
				print(f'ITER : {iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}')
				campaign.add_observation(samples[0], measurement)

				# check for convergence
				if [samples[0]['organic'], samples[0]['anion'], samples[0]['cation']] == OPT:
					print(f'FOUND OPTIMUM AFTER {iter+1} ITERATIONS!')
					break
			all_campaigns.append(campaign)

		pickle.dump(all_campaigns, open('results/perovskites_botorch_descriptors.pkl', 'wb'))
