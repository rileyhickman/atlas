#!/usr/bin/env python

import numpy as np
import torch
from botorch.optim import optimize_acqf, optimize_acqf_mixed, optimize_acqf_discrete


from olympus import ParameterVector


from atlas import Logger

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

from atlas.optimizers.acqfs import (
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
		best_idx = None # only needed for the fully categorical case
		if self.problem_type == 'fully_continuous':
			results = self._optimize_fully_continuous()
		elif self.problem_type == 'mixed':
			results = self._optimize_mixed()
		elif self.problem_type in ['fully_categorical', 'fully_discrete', 'mixed_dis_cat']:
			results, best_idx = self._optimize_fully_categorical()

		return self.postprocess_results(results, best_idx)


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
				self.choices_feat.detach().numpy(), self._mins_x, self._maxs_x,
			)
			self.choices_feat = torch.tensor(self.choices_feat)

		results, best_idx = self._optimize_acqf_discrete(
			acq_function=self.acqf,
			q=self.batch_size,
			max_batch_size=1000,
			choices=self.choices_feat.float(),
			unique=True
				)
		return results, best_idx


	def _optimize_acqf_discrete(self, acq_function, q, max_batch_size, choices, unique, strategy='greedy'):
		# this function assumes 'unique' argument is always set to True
		# strategy can be set to 'greedy' or 'sequential'
		original_choices_batched = torch.clone(choices)
		choices_batched = choices.unsqueeze(-2)
		if q > 1:
			if strategy=='sequential':
				candidate_list, acq_value_list = [], []
				base_X_pending = acq_function.X_pending
				for _ in range(q):
					with torch.no_grad():
						acq_values = torch.cat([acq_function(X_) for X_ in choices_batched.split(max_batch_size)])
						# print(acq_values)
					best_idx = torch.argmax(acq_values)
					candidate_list.append(choices_batched[best_idx])
					acq_value_list.append(acq_values[best_idx])
					# set pending points
					candidates = torch.cat(candidate_list, dim=-2)
					acq_function.set_X_pending(
						torch.cat([base_X_pending, candidates], dim=-2)
						if base_X_pending is not None
						else candidates
					)
					# need to remove choice from choice set if enforcing uniqueness
					if unique:
						choices_batched = torch.cat(
							[choices_batched[:best_idx], choices_batched[best_idx + 1 :]]
						)
				# Reset acq_func to previous X_pending state
				acq_function.set_X_pending(base_X_pending)
				# need to get and return the original indices of the selected candidates
				best_idxs = []
				for candidate in candidate_list: # each candidate is shape (1, num_features)
					bools = [torch.all(candidate[0]==original_choices_batched[i,:]) for i in range(original_choices_batched.shape[0])]
					assert bools.count(True)==1
					best_idxs.append(np.where(bools)[0][0])


				return candidate_list, best_idxs

			elif strategy == 'greedy':
				with torch.no_grad():
					acq_values = torch.cat([acq_function(X_) for X_ in choices_batched.split(max_batch_size)])
				best_idxs = list(torch.argsort(acq_values, descending=True).detach().numpy())[:q]
				# print(best_idxs)
				#
				# quit()

				return [choices[best_idx] for best_idx in best_idxs], best_idxs

		# otherwise we have q=1, just take the argmax acqusition value
		with torch.no_grad():
			acq_values = torch.cat([acq_function(X_) for X_ in choices_batched.split(max_batch_size)])
		best_idx = [torch.argmax(acq_values).detach()]


		return [choices[best_idx]], best_idx


	def postprocess_results(self, results, best_idx=None):
		# expects list as results

		# convert the results form torch tensor to numpy
		#results_np = np.squeeze(results.detach().numpy())
		if isinstance(results, list):
			results_torch = [torch.squeeze(res) for res in results]
		else:
			# TODO: update this
			results_torch = results


		# TODO: clean this bit up
		if self.problem_type in ['fully_categorical', 'mixed', 'mixed_dis_cat'] and not self.has_descriptors:
			# project the sample back to Olympus format
			samples = []
			# if len(results_np.shape) == 1:
			# 	results_np = results_np.reshape(1, -1)
			#results_torch = reverse_normalize(results_torch, self._mins_x, self._maxs_x)
			for sample_ix in range(len(results_torch)):
				sample = project_to_olymp(
					results_torch[sample_ix], self.param_space,
					has_descriptors=self.has_descriptors,
					choices_feat=self.choices_feat, choices_cat=self.choices_cat,
				)
				samples.append(ParameterVector().from_dict(sample, self.param_space))


		elif self.problem_type in ['fully_categorical', 'mixed', 'mixed_dis_cat'] and self.has_descriptors:

			samples = []
			# if len(results_torch.shape) == 1:
			# 	results_torch = results_torch.reshape(1, -1)
			for sample_ix in range(len(results_torch)):
				sample = self.choices_cat[best_idx[sample_ix]]
				olymp_sample = {}
				for elem, name in zip(sample, [p.name for p in self.param_space]):
					olymp_sample[name] = elem
				samples.append(ParameterVector().from_dict(olymp_sample, self.param_space))


		else:
			# reverse transform the inputs
			results_np = results_torch.detach().numpy()
			results_np = reverse_normalize(results_np, self._mins_x, self._maxs_x)
			if len(results_np.shape) == 1:
				results_np = results_np.reshape(1, -1)
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
