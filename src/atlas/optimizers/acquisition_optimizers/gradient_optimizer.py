#!/usr/bin/env python

import numpy as np
import torch 
from botorch.optim import optimize_acqf, optimize_acqf_mixed, optimize_acqf_discrete


from olympus import ParameterVector


from atlas import Logger

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

from atlas.optimizers.gp.acqfs import (
	get_batch_initial_conditions, 
	create_available_options,
)


class GradientOptimizer():

	def __init__(
		self,
		param_space,
		acqf, 
		bounds, 
		known_constraints,
		batch_size,
		feas_strategy,
		fca_constraint,
		has_descriptors,
		params,
		mins_x, 
		maxs_x,
	):
		self.param_space = param_space
		self.problem_type = infer_problem_type(self.param_space)
		self.acqf = acqf 
		self.bounds = bounds
		self.known_constraints = known_constraints
		self.batch_size = batch_size
		self.feas_strategy = feas_strategy
		self.fca_constraint = fca_constraint
		self.has_descriptors = has_descriptors
		self._params = params
		self._mins_x =  mins_x
		self._maxs_x = maxs_x 

		self.choices_feat, self.choices_cat = None, None



	def optimize(self):

		if self.problem_type == 'fully_continuous':
			results = self._optimize_fully_continuous()
		elif self.problem_type == 'mixed':
			results = self._optimize_mixed()
		elif self.problem_type == 'fully_categorical':
			results = self._optimize_fully_categorical()

		return self.postprocess_results(results)


	def _optimize_fully_continuous(self):

		nonlinear_inequality_constraints = []
		if callable(self.known_constraints):
			# we have some known constraints
			nonlinear_inequality_constraints.append(self.known_constraints)
		if self.feas_strategy == 'fca':
			nonlinear_inequality_constraints.append(self.fca_constraint)
			# attempt to get the batch initial conditions
			batch_initial_conditions = get_batch_initial_conditions(
				num_restarts=200, batch_size=self.batch_size, param_space=self.param_space,
				constraint_callable=nonlinear_inequality_constraints,
			)
			if type(batch_initial_conditions) == type(None):
				# if we cant find sufficient inital design points, resort to using the
				# acqusition function only (without the feasibility constraint)
				nonlinear_inequality_constraints = nonlinear_inequality_constraints.pop()
				
				# try again with only the a priori known constraints
				batch_initial_conditions = get_batch_initial_conditions(
					num_restarts=200, batch_size=self.batch_size, param_space=self.param_space,
					constraint_callable=nonlinear_inequality_constraints,
				)

				if type(batch_initial_conditions) == type(None):
					# if we still cannot find initial conditions, there is likey a problem, return to user
					message = 'Could not find inital conditions for constrianed optimization...'
					Logger.log(message, 'FATAL')
				elif type(batch_initial_conditions) == torch.Tensor:
					# weve found sufficient conditions
					pass
			elif type(batch_initial_conditions) == torch.Tensor:
				# we've found initial conditions
				pass
		else:
			# we dont have fca constraints, if we have known constraints, 
			if callable(self.known_constraints):

				batch_initial_conditions = get_batch_initial_conditions(
				num_restarts=200, batch_size=self.batch_size, param_space=self.param_space,
				constraint_callable=nonlinear_inequality_constraints,
				)
				if type(batch_initial_conditions) == type(None):
					# return an error to the user 
					message = 'Could not find inital conditions for constrianed optimization...'
					Logger.log(message, 'FATAL')
		
		if not self.known_constraints and not self.feas_strategy =='fca':
			# we dont have any constraints
			nonlinear_inequality_constraints = None
			batch_initial_conditions = None 


		results, _ = optimize_acqf(
			acq_function=self.acqf,
			bounds=self.bounds,
			num_restarts=200,
			q=self.batch_size,
			raw_samples=1000,
			nonlinear_inequality_constraints=nonlinear_inequality_constraints,
			batch_initial_conditions=batch_initial_conditions,
		)

		return results


	def _optimize_mixed(self):

		fixed_features_list = get_fixed_features_list(self.param_space)

		results, _ = optimize_acqf_mixed(
			acq_function=self.acqf,
			bounds=self.bounds,
			num_restarts=30,
			q=self.batch_size,
			raw_samples=800,
			fixed_features_list=fixed_features_list,
		)

		return results

	def _optimize_fully_categorical(self):


		# need to implement the choices input, which is a
		# (num_choices * d) torch.Tensor of the possible choices
		# need to generate fully cartesian product space of possible
		# choices
		if self.feas_strategy == 'fca':
			# if we have feasibilty constrained acquisition, prepare only
			# the feasible options as availble choices
			constraint_callable = self.fca_constraint
		else:
			constraint_callable = None

		self.choices_feat, self.choices_cat = create_available_options(
			self.param_space, self._params, constraint_callable
		)
		if self.has_descriptors:
			self.choices_feat = forward_normalize(
				choices_feat.detach().numpy(), self._mins_x, self._maxs_x,
			)
			self.choices_feat = torch.tensor(self.choices_feat)

		results, _ = optimize_acqf_discrete(
			acq_function=self.acqf,
			q=self.batch_size,
			max_batch_size=1000,
			choices=self.choices_feat.float(),
			unique=True
				)
		return results


	def postprocess_results(self, results):

		# convert the results form torch tensor to numpy
		#results_np = np.squeeze(results.detach().numpy())
		results_torch  = torch.squeeze(results)

		# TODO: clean this bit up
		if self.problem_type in ['fully_categorical', 'mixed'] and not self.has_descriptors:
			# project the sample back to Olympus format
			samples = []
			results_np = results_torch.detach().numpy()
			if len(results_np.shape) == 1:
				results_np = results_np.reshape(1, -1)
			results_np = reverse_normalize(results_np, self._mins_x, self._maxs_x)
			for sample_ix in range(results_np.shape[0]):
				sample = project_to_olymp(
					results_np[sample_ix], self.param_space,
					has_descriptors=self.has_descriptors,
					choices_feat=self.choices_feat, choices_cat=self.choices_cat,
				)
				samples.append(ParameterVector().from_dict(sample, self.param_space))


		elif self.problem_type in ['fully_categorical', 'mixed'] and self.has_descriptors:

			# if we have descriptors, dont reverse normalize the results (this
			# works better for the lookup)
			# project the sample back to Olympus format
			samples = []
			if len(results_torch.shape) == 1:
				results_torch = results_torch.reshape(1, -1)
			for sample_ix in range(results_torch.shape[0]):
				sample = project_to_olymp(
					results_torch[sample_ix], 
					self.param_space,
					has_descriptors=self.has_descriptors,
					choices_feat=self.choices_feat, choices_cat=self.choices_cat,
				)
				samples.append(ParameterVector().from_dict(sample, self.param_space))

		else:
			# reverse transform the inputs
			results_np = results_torch.detach().numpy()
			if len(results_np.shape) == 1:
				results_np = results_np.reshape(1, -1)
			results_np = reverse_normalize(results_np, self._mins_x, self._maxs_x)

			samples = []
			for sample_ix in range(results_np.shape[0]):
				# project the sample back to Olympus format
				sample = project_to_olymp(
					results_np[sample_ix], 
					self.param_space,
					has_descriptors=self.has_descriptors,
					choices_feat=self.choices_feat, choices_cat=self.choices_cat,
				)
				samples.append(ParameterVector().from_dict(sample, self.param_space))

		return_params = samples

		return return_params




