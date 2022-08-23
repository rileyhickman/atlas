#!/usr/bin/env python



from atlas import Logger

from atlas.optimizers.acquisition_optimizers.genetic_optimizer import GeneticOptimizer
from atlas.optimizers.acquisition_optimizers.gradient_optimizer import GradientOptimizer

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

class AcquisitionOptimizer():

	def __init__(
		self,
		kind,
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
		self.kind = kind
		self.param_space = param_space
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

		# check kind of acquisition optimization
		if self.kind == 'gradient':
			self.optimizer = GradientOptimizer(
				self.param_space,
				self.acqf,
				self.bounds,
				self.known_constraints,
				self.batch_size,
				self.feas_strategy,
				self.fca_constraint,
				self.has_descriptors,
				self._params,
				self._mins_x,
				self._maxs_x,
			)

		elif self.kind == 'genetic':
			pass 

		else:
			msg = f'Acquisition optimizer kind {self.kind} not known'
			Logger.log(msg, 'FATAL')



	def optimize(self):
		results = self.optimizer.optimize()
		return  results
	

	