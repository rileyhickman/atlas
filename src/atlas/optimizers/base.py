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



class BasePlanner(CustomPlanner):

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
		vgp_iters=2000,
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
		AbstractPlanner.__init__(**locals())
		self.goal = goal
		self.feas_strategy = feas_strategy
		self.feas_param = feas_param
		self.batch_size = batch_size
		if random_seed is None:
			self.random_seed = np.random.randint(0, int(10e6))
		else:
			self.random_seed = random_seed
		np.random.seed(self.random_seed)
		self.num_init_design = num_init_design
		self.init_design_strategy = init_design_strategy
		self.acquisition_optimizer_kind = acquisition_optimizer_kind
		self.vgp_iters = vgp_iters
		self.vgp_lr = vgp_lr
		self.max_jitter = max_jitter
		self.cla_threshold = cla_threshold
		self.known_constraints = known_constraints
		self.general_parmeters = general_parmeters
		self.is_moo = is_moo
		self.value_space = value_space
		self.scalarizer_kind = scalarizer_kind
		self.moo_params = moo_params
		self.goals = goals

		# check multiobjective stuff
		if self.is_moo:
			if self.goals is None:
				message = f'You must individual goals for multiobjective optimization'
				Logger.log(message, 'FATAL')

			if self.goal == 'maximize':
				message = 'Overall goal must be set to minimization for multiobjective optimization. Updating ...'
				Logger.log(message, 'WARNING')
				self.goal = 'minimize'

			self.scalarizer = Scalarizer(
				kind=self.scalarizer_kind, value_space=self.value_space, goals=self.goals, **self.moo_params
			)


		# treat the inital design arguments
		if self.init_design_strategy == 'random':
			self.init_design_planner = olympus.planners.RandomSearch(goal=self.goal)
		elif self.init_design_strategy == 'sobol':
			self.init_design_planner = olympus.planners.Sobol(goal=self.goal, budget=self.num_init_design)
		elif self.init_design_strategy == 'lhs':
			self.init_design_planner = olympus.planners.LatinHypercube(goal=self.goal, budget=self.num_init_design)
		else:
			message = f'Initial design strategy {self.init_design_strategy} not implemented'
			Logger.log(message, 'FATAL')

		self.num_init_design_completed = 0


	def _set_param_space(self, param_space):
		''' set the Olympus parameter space (not actually really needed)
		'''

		# infer the problem type
		self.problem_type = infer_problem_type(self.param_space)

		# make attribute that indicates wether or not we are using descriptors for
		# categorical variables
		if self.problem_type == 'fully_categorical':
			descriptors = []
			for p in self.param_space:
				descriptors.extend(p.descriptors)
			if all(d is None for d in descriptors):
				self.has_descriptors = False
			else:
				self.has_descriptors = True
		elif self.problem_type in ['mixed','mixed_dis_cat']:
			descriptors = []
			for p in self.param_space:
				if p.type == 'categorical':
					descriptors.extend(p.descriptors)
			if all(d is None for d in descriptors):
				self.has_descriptors = False
			else:
				self.has_descriptors = True
		else:
			self.has_descriptors = False



	def build_train_classification_gp(self, train_x, train_y):
		''' build the GP classification model and likelihood
		and train the model
		'''
		model = ClassificationGPMatern(train_x, train_y)
		likelihood = gpytorch.likelihoods.BernoulliLikelihood()

		model, likelihood = self.train_vgp(model, likelihood,  train_x, train_y)

		return model, likelihood

	def train_vgp(self, model, likelihood, train_x, train_y):

		model.train()
		likelihood.train()
		optimizer=torch.optim.Adam(model.parameters(), lr=self.vgp_lr)

		mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())


		# TODO: might be better to break into batches here...
		# NOTE: we could also do some sort of cross-validation here for early stopping
		start_time = time.time()
		with gpytorch.settings.cholesky_jitter(self.max_jitter):
			for iter_ in range(self.vgp_iters):
				optimizer.zero_grad()
				output = model(train_x)
				loss = -mll(output, train_y)
				loss.backward()
				optimizer.step()
		vgp_train_time = time.time() - start_time
		print(f' >>> [{self.vgp_iters} epochs] VGP trained in {round(vgp_train_time,3)} sec \t Loss : {round(loss.item(), 3)} ')

		return model, likelihood


	def build_train_data(self):
		''' build the training dataset at each iteration
		'''
		if self.is_moo:
			# parameters should be the same for each objective
			# nans should be in the same locations for each objective
			feas_ix = np.where(~np.isnan(self._values[:, 0]))[0]
			# generate the classification dataset
			params_cla = self._params.copy()
			values_cla = np.where(~np.isnan(self._values[:, 0]), 0., self._values[:, 0])
			train_y_cla = np.where(np.isnan(values_cla), 1., values_cla)
			# generate the regression dataset
			params_reg = self._params[feas_ix].reshape(-1, 1)
			train_y_reg = self._values[feas_ix, :]  # (num_feas_observations, num_objectives)
			# scalarize the data
			train_y_reg = self.scalarizer.scalarize(train_y_reg).reshape(-1, 1) # (num_feas_observations, 1)

		else:
			feas_ix = np.where(~np.isnan(self._values))[0]
			# generate the classification dataset
			params_cla = self._params.copy()
			values_cla = np.where(~np.isnan(self._values), 0., self._values)
			train_y_cla = np.where(np.isnan(values_cla), 1., values_cla)

			# generate the regression dataset
			params_reg = self._params[feas_ix].reshape(-1, 1)
			train_y_reg = self._values[feas_ix].reshape(-1, 1)

		train_x_cla, train_x_reg = [], []

		# adapt the data to

		# adapt the data from olympus form to torch tensors
		for ix in range(self._values.shape[0]):
			sample_x = []
			for param_ix, (space_true, element) in enumerate(zip(self.param_space, params_cla[ix])):
				if self.param_space[param_ix].type == 'categorical':
					feat = cat_param_to_feat(space_true, element)
					sample_x.extend(feat)
				else:
					sample_x.append(float(element))
			train_x_cla.append(sample_x)
			if ix in feas_ix:
				train_x_reg.append(sample_x)

		train_x_cla, train_x_reg = np.array(train_x_cla), np.array(train_x_reg)

		# scale the training data - normalize inputs and standardize outputs
		# TODO: should we scale all the parameters together?
		self._mins_x = np.amin(train_x_cla, axis=0)
		self._maxs_x = np.amax(train_x_cla, axis=0)

		self._means_y, self._stds_y  = np.mean(train_y_reg, axis=0), np.std(train_y_reg, axis=0)
		self._stds_y = np.where(self._stds_y==0.0, 1., self._stds_y)

		if self.problem_type == 'fully_categorical' and not self.has_descriptors:
			# we dont scale the parameters if we have a fully one-hot-encoded representation
			pass
		else:
			# scale the parameters
			train_x_cla = forward_normalize(train_x_cla, self._mins_x, self._maxs_x)
			train_x_reg = forward_normalize(train_x_reg, self._mins_x, self._maxs_x)

		# always forward transform the objectives for the regression problem
		train_y_reg = forward_standardize(train_y_reg, self._means_y, self._stds_y)

		# convert to torch tensors and return
		return (
			torch.tensor(train_x_cla).float(), torch.tensor(train_y_cla).squeeze().float(),
			torch.tensor(train_x_reg).double(), torch.tensor(train_y_reg).double(),
		)


	def reg_surrogate(self, X, return_np=False):
		''' make prediction using regression surrogate model

		Args:
			X (np.ndarray or list): 2d numpy array or nested list with input parameters
		'''

		if not hasattr(self, 'reg_model'):
			msg = 'Optimizer does not yet have regression surrogate model'
			Logger.log(msg, 'FATAL')

		X_proc = []
		# adapt the data from olympus form to torch tensors
		for ix in range(len(X)):
			sample_x = []
			for param_ix, (space_true, element) in enumerate(zip(self.param_space, X[ix])):
				if self.param_space[param_ix].type == 'categorical':
					feat = cat_param_to_feat(space_true, element)
					sample_x.extend(feat)
				else:
					sample_x.append(float(element))
			X_proc.append(sample_x)

		X_proc = torch.tensor(np.array(X_proc)).double()

		if self.problem_type == 'fully_categorical' and not self.has_descriptors:
			# we dont scale the parameters if we have a fully one-hot-encoded representation
			pass
		else:
			# scale the parameters
			X_proc = forward_normalize(X_proc, self._mins_x, self._maxs_x)

		posterior = self.reg_model.posterior(X=X_proc)
		pred_mu, pred_sigma = posterior.mean.detach(), torch.sqrt(posterior.variance.detach())

		# reverse scale the predictions
		pred_mu = reverse_standardize(pred_mu, self._means_y, self._stds_y)

		if self.goal == 'maximize':
			pred_mu = -pred_mu

		if return_np:
			pred_mu, pred_sigma = pred_mu.numpy(), pred_sigma.numpy()

		return pred_mu, pred_sigma


	def cla_surrogate(self, X, return_np=False, normalize=True):

		if not hasattr(self, 'cla_model'):
			msg = 'Optimizer does not yet have classification surrogate model'
			Logger.log(msg, 'FATAL')

		X_proc = []
		# adapt the data from olympus form to torch tensors
		for ix in range(len(X)):
			sample_x = []
			for param_ix, (space_true, element) in enumerate(zip(self.param_space, X[ix])):
				if self.param_space[param_ix].type == 'categorical':
					feat = cat_param_to_feat(space_true, element)
					sample_x.extend(feat)
				else:
					sample_x.append(float(element))
			X_proc.append(sample_x)

		X_proc = torch.tensor(np.array(X_proc)).double()

		if self.problem_type == 'fully_categorical' and not self.has_descriptors:
			# we dont scale the parameters if we have a fully one-hot-encoded representation
			pass
		else:
			# scale the parameters
			X_proc = forward_normalize(X_proc, self._mins_x, self._maxs_x)

		likelihood = self.cla_likelihood(self.cla_model(X_proc.float()))
		mean = likelihood.mean.detach()
		mean = 1.-mean.view(mean.shape[0],1) # switch from p_feas to p_infeas
		if normalize:
			_max =  torch.amax(mean, axis=0)
			_min =  torch.amin(mean, axis=0)
			mean = ( mean - _min ) / (_max - _min)

		if return_np:
			mean = mean.numpy()

		return mean


	def acquisition_function(self, X, return_np, normalize=True):

		X_proc = []
		# adapt the data from olympus form to torch tensors
		for ix in range(len(X)):
			sample_x = []
			for param_ix, (space_true, element) in enumerate(zip(self.param_space, X[ix])):
				if self.param_space[param_ix].type == 'categorical':
					feat = cat_param_to_feat(space_true, element)
					sample_x.extend(feat)
				else:
					sample_x.append(float(element))
			X_proc.append(sample_x)

		X_proc = torch.tensor(np.array(X_proc)).double()

		if self.problem_type == 'fully_categorical' and not self.has_descriptors:
			# we dont scale the parameters if we have a fully one-hot-encoded representation
			pass
		else:
			# scale the parameters
			X_proc = forward_normalize(X_proc, self._mins_x, self._maxs_x)


		acqf_vals = self.acqf(
			X_proc.view(X_proc.shape[0], 1, X_proc.shape[-1])
		).detach()

		acqf_vals = acqf_vals.view(acqf_vals.shape[0], 1)

		if normalize:
			_max =  torch.amax(acqf_vals, axis=0)
			_min =  torch.amin(acqf_vals, axis=0)
			acqf_vals = ( acqf_vals - _min ) / (_max - _min)

		if return_np:
			acqf_vals = acqf_vals.numpy()

		return acqf_vals


	def _tell(self, observations):
		''' unpack the current observations from Olympus
		Args:
			observations (obj): Olympus campaign observations object
		'''
		# elif type(observations) == olympus.campaigns.observations.Observations:
		self._params = observations.get_params(as_array=True) # string encodings of categorical params
		self._values = observations.get_values(as_array=True, opposite=self.flip_measurements)

		# make values 2d if they are not already
		if len(np.array(self._values).shape)==1:
			self._values = np.array(self._values).reshape(-1, 1)




	def fca_constraint(self, X):
		''' Each callable is expected to take a `(num_restarts) x q x d`-dim tensor as an
			input and return a `(num_restarts) x q`-dim tensor with the constraint
			values. The constraints will later be passed to SLSQP. You need to pass in
			`batch_initial_conditions` in this case. Using non-linear inequality
			constraints also requires that `batch_limit` is set to 1, which will be
			done automatically if not specified in `options`.
			>= 0 is a feasible point
			<  0 is an infeasible point
		Args:
			X (torch.tensor):
		'''
		# handle the various potential input tensor sizes (this function can be called from
		# several places, including inside botorch)
		# TODO: this is pretty messy, consider cleaning up
		if len(X.size())==3:
			X = X.squeeze(1)
		if len(X.size())==1:
			X = X.view(1, X.shape[0])
		# squeeze the middle q dimension
		# this expression is >= 0 for a feasible point, < 0 for an infeasible point
		# p_feas should be 1 - P(infeasible|X) which is returned by the classifier
		with gpytorch.settings.cholesky_jitter(1e-1):
			constraint_val = (1 - self.cla_likelihood(self.cla_model(X.float())).mean.unsqueeze(-1).double()) - self.feas_param

		return constraint_val
