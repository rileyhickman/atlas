#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import copy
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from deap import base, creator, tools
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace
from rich.progress import track

from atlas.optimizers.params import Parameters

from atlas import Logger
from atlas.optimizers.acqfs import create_available_options
from atlas.optimizers.acquisition_optimizers.base_optimizer import (
	AcquisitionOptimizer,
)
from atlas.optimizers.params import Parameters
from atlas.optimizers.utils import (
	cat_param_to_feat,
	forward_normalize,
	forward_standardize,
	get_cat_dims,
	get_fixed_features_list,
	infer_problem_type,
	param_vector_to_dict,
	propose_randomly,
	reverse_normalize,
	reverse_standardize,
	gen_partitions,
)


class GeneticGeneralOptimizer(AcquisitionOptimizer):
	def __init__(
		self,
		params_obj: Parameters,
		acquisition_type: str,
		acqf, # needs to be instance of MedusaAcquisiton
		known_constraints: Callable,
		batch_size: int,
		feas_strategy: str,
		fca_constraint: Callable,
		params: torch.Tensor,
		timings_dict: Dict,
		max_Ng: int,
		func_param_space, 
		num_init_evals:int=int(1e3), # number of inital evals for GA
		
		**kwargs: Any, 
	):
		local_args = {
			key: val for key, val in locals().items() if key != "self"
		}
		self.max_Ng = max_Ng
		self.num_init_evals = num_init_evals

		self.params_obj = params_obj
		self.param_space = self.params_obj.param_space
		self.problem_type = infer_problem_type(self.param_space)
		self.bounds = self.params_obj.bounds
		self.batch_size = batch_size
		self.acqf = acqf
		self.timings_dict = timings_dict
		self.acquisition_type = acquisition_type
		

		self.func_param_space = func_param_space
		self.func_problem_type = infer_problem_type(self.func_param_space)

		self.param_space_dim = len(self.param_space)
		self.func_param_space_dim = len(self.func_param_space)

		self.functional_dims = np.logical_not(self.params_obj.exp_general_mask)

		# range of opt domain dimensions for deap 
		self.param_ranges = self._get_param_ranges()

		# parameter space indices that are general parameters
		self.general_parameters = self.params_obj.general_dims



	#------------------------
	# PROPERTIES AND HELPERS
	#------------------------

	@property
	def S(self):
		""" indices of general params """
		return [i for i in range(self.num_S)]

	@property
	def num_S(self):
		"""number of non-functional parameter options """
		S_count = 0
		for param_ix in self.general_parameters:
			S_count+=len(self.param_space[param_ix].options)
		return S_count
 
	@staticmethod
	def _dummy_get_bounds(param_space):
		...
	
	@staticmethod
	def _project_bounds(x, x_low, x_high):
		if x < x_low:
			return x_low
		elif x > x_high:
			return x_high
		else:
			return x
	
	@staticmethod
	def cxDummy(ind1, ind2):
		"""Dummy crossover that does nothing. This is used when we have a single gene in the chromosomes, such that
		crossover would not change the population.
		"""
		return ind1, ind2
	
	@staticmethod
	def param_vectors_to_deap_population(param_vectors):
		population = []
		for param_vector in param_vectors:
			ind = creator.Individual(param_vector)
			population.append(ind)
		return population

	def _wrapped_fca_constraint(self, params):
		# >= 0 is a feasible point --> True
		# < 0 is an infeasible point --> False
		# transform dictionary rep of x to expanded format
		expanded = self.params_obj.param_vectors_to_expanded(
			[ParameterVector().from_dict(params,self.param_space)],
			is_scaled=True,
			return_scaled=False # should already be scaled
		)
		val = self.fca_constraint(
			torch.tensor(expanded).view(expanded.shape[0], 1, expanded.shape[1])
		).detach().numpy()[0][0]

		if val >= 0:
			return True
		else:
			return False
		
	def _get_param_ranges(self):
		param_ranges = []
		counter = 0
		for param in self.param_space:
			if param.type == 'continuous':
				param_ranges.append(self.bounds[1,counter]-self.bounds[0,counter])
				counter+=1
			elif param.type=='discrete':
				param_ranges.append(len(param.options))
				counter+=1
			elif param.type == 'categorical':
				param_ranges.append(len(param.options))
				if self.params_obj.has_descriptors:
					counter+=len(param.descriptors[0])
				else:
					counter+=len(param.options)
		return np.array(param_ranges)

	def indexify(self):
		samples = []
		counter = 0
		for cond, cond_raw in zip(self.batch_initial_conditions, self.raw_conditions):
			sample = []
			counter = 0
			for elem, p in zip(cond_raw, self.func_param_space):
				if p.type == 'continuous':
					sample.append(float(cond[counter]))
					counter+=1
				elif p.type == 'discrete':
					sample.append(float(p.options.index(float(elem))))
					counter+=1
				elif p.type == 'categorical':
					sample.append(float(p.options.index(elem)))
					if self.has_descriptors:
						counter+=len(p.descriptors[0])
					else:
						counter+=len(p.options)
			samples.append(sample)
		return np.array(samples)

	def deindexify(self, x):
		samples = []
		for x_ in x:
			sample = []
			counter = 0
			for elem, p in zip(x_, self.func_param_space):

				if p.type == 'continuous':
					sample.append(float(elem))
					counter+=1
				elif p.type == 'discrete':
					sample.append(float(p.options[int(elem)]))
					counter+=1
				elif p.type == 'categorical':
					sample.extend(
						cat_param_to_feat(
							p, p.options[int(elem)], self.params_obj.has_descriptors,
						)
					)
			samples.append(sample)
		return np.array(samples)


	#-------------
	# EVALUATION
	#-------------

	def dummy_evaluate(self, individual):
		return np.random.uniform(),

	# TODO: outdated
	@staticmethod
	def measure_single_obj(X_func, si, surf_map):
		return surf_map[si].run(X_func)[0][0]

	# TODO: outdated
	def __OLD_evaluate(self, individual):

		G = individual['G']
		X_func = individual['X_func']
		f_x = 0.
		for g_ix, Sg in enumerate(G):
			for si in Sg:
				f_x += self.measure_single_obj(X_func[g_ix], si, self.surf_map)

		return f_x, 

	def acquisition(self, individual: Dict) -> Tuple:
		# print(individual)
		G = individual['G']
		X_func = np.array(individual['X_func'])
		# print('type G : ', type(G))
		# print('type X_func : ',type(X_func))
		# print('shape X_func : ', X_func.shape)

		# de-indexify X_func only before calling the acquisition function
		X_func = self.deindexify(X_func)


		# print('deindex X_func shape : ', X_func.shape)
		# print('deindex X_func : ', X_func)

		# NOTE: dont need to make this a torch tensor yet ... 
		# X_func = torch.tensor(
		# 	X_func.reshape((1, self.batch_size, X_func.shape[1]))
		# )
		# return the negative of the acqf - this is conventionally minimized by
		# deap, but we want to maximize acqf
		return -self.acqf(X_func=X_func, G=G).detach().numpy(),
	

	#----------------------------
	# POPULATION INITIALIZATION
	#----------------------------

	def init_population_best(self, num_inds, max_partitions=20):
		f_xs = []
		Gs = gen_partitions(self.S)
		if len(Gs) > max_partitions:
			Logger.log(f'Max partitions exceeded. Sampling random subset of {max_partitions}', 'WARNING')
			np.random.shuffle(Gs)
			Gs = Gs[:max_partitions]

		for G_ix, G in track(enumerate(Gs), total=len(Gs), description='Evaluating initial candidates...'):
			Ng = len(G)
			X_funcs = []
			for _ in range(Ng):

				# produce the proposals for the functional parameter space only (X_func)
				proposals, raw_proposals = propose_randomly(
					num_proposals=self.num_init_evals, 
					param_space=self.func_param_space,
					has_descriptors=self.params_obj.has_descriptors,
				)
				# print('proposals : ', proposals)
				# print('')
				# print('raw_proposals : ', raw_proposals )
				# quit()

				if self.func_problem_type == 'fully_categorical':
					# use raw proposals (string reps) for fully categorical
					X_funcs.append(raw_proposals)
				elif self.func_problem_type == 'fully_continuous': 
					X_funcs.append(proposals)
				else:
					print('SOMETHING IS WRONG')
					quit()
				
			X_funcs = np.array(X_funcs).swapaxes(0,1)
			for X_func in X_funcs:
				dict_ = {'G': G, 'X_func': list([list(X) for X in X_func]), 'Ng': Ng}
				dict_['f_x'] = self.acquisition(dict_)[0]
				f_xs.append(dict_)

		vals = [d['f_x'] for d in f_xs]
		
		assert len(vals) > num_inds
		sort_idx = np.argsort(vals)[:num_inds]
		pop = [creator.Individual(f_xs[idx]) for idx in sort_idx]

		return pop

	def init_population_random(self, num_inds, max_partitions=20):
		f_xs = []
		Gs = gen_partitions(self.S)
		if len(Gs) > max_partitions:
			Logger.log(f'Max partitions exceeded. Sampling random subset of {max_partitions}', 'WARNING')
			np.random.shuffle(Gs)
			Gs = Gs[:max_partitions]

		for G_ix, G in track(enumerate(Gs), total=len(Gs), description='Evaluating initial candidates...'):
			Ng = len(G)
			X_funcs = []
			for _ in range(Ng):

				proposals, raw_proposals = propose_randomly(
					num_proposals=self.num_init_evals, 
					param_space=self.func_param_space,
					has_descriptors=self.params_obj.has_descriptors,
				)
				if self.func_problem_type == 'fully_categorical':
					# use raw proposals (string reps) for fully categorical
					X_funcs.append(raw_proposals)
				elif self.func_problem_type == 'fully_continuous': 
					X_funcs.append(proposals)
				else:
					print('SOMETHING IS WRONG')
					quit()
				
			X_funcs = np.array(X_funcs).swapaxes(0,1)
			for X_func in X_funcs:
				dict_ = {'G': G, 'X_func': list([list(X) for X in X_func]), 'Ng': Ng}
				f_xs.append(dict_)

			np.random.shuffle(f_xs)
			pop = [creator.Individual(f_x) for f_x in f_xs[:num_inds]]

		return pop
	

	#------------------
	# CUSTOM MUTATIONS
	#------------------

	def custom_mutate_G(self, ind):
		""" mutate non-functional param assignments G
		"""
		
		# determine whether we are making a mutation to G
		mutation_types = ['fusion', 'swap', 'split'] #'fusion', 'split']
		mutated = False
		while not mutated and len(mutation_types)>0:
			mutation_type = np.random.choice(mutation_types)
			#---------------
			# SWAP MUTATION
			#---------------
			if mutation_type == 'swap':
				# swap one general param from one subset to another
				if not len(ind['G'])>1:
					mutation_types.remove(mutation_type)
				else:
					mut_locs = np.random.choice(
						np.arange(len(ind['G'])), size=(2,), replace=False,
					)
					mut_loc_1, mut_loc_2 = mut_locs[0], mut_locs[1]
					mut_ix_1 = np.random.randint(len(ind['G'][mut_loc_1]))
					mut_ix_2 = np.random.randint(len(ind['G'][mut_loc_2]))
					mut_val_1 = copy.deepcopy(ind['G'][mut_loc_1][mut_ix_1])
					mut_val_2 = copy.deepcopy(ind['G'][mut_loc_2][mut_ix_2])                
					# make swap
					ind['G'][mut_loc_1][mut_ix_1] = mut_val_2
					ind['G'][mut_loc_2][mut_ix_2] = mut_val_1
					mutated = True
			#----------------
			# SPLIT MUTATION
			#----------------
			elif mutation_type == 'split':
				# fuse two subsets together at random
				if all([len(Sg)==1 for Sg in ind['G']]):
					mutation_types.remove(mutation_type)
				else:
					Sg_lens = np.array([len(Sg) for Sg in ind['G']])
					split_idxs = np.where(Sg_lens>1)[0]
					split_idx = np.random.choice(split_idxs)
					mut_val = copy.deepcopy(ind['G'][split_idx])
					
					np.random.shuffle(mut_val)
					if len(mut_val)==2:
						num_splits, cutoff = 1, [1]
					else:
						num_splits = np.random.randint(1,len(mut_val)-1)
						cutoff = sorted(np.random.choice(np.arange(1,len(mut_val)-1), size=(num_splits,), replace=False))
					chunks = []
					for ix in range(len(cutoff)):
						if ix==0:
							chunks.append(mut_val[:cutoff[ix]])
						else:
							chunks.append(mut_val[cutoff[ix-1]:cutoff[ix]])
						if ix == len(cutoff)-1:
							chunks.append(mut_val[cutoff[ix]:])
					del ind['G'][split_idx]
					ind['G'].extend(sorted(c) for c in chunks)
					ind['G'] = sorted(ind['G'])
					mutated=True
			#----------------
			# FUSION MUTATION
			#----------------
			elif mutation_type == 'fusion':
				# split one group at random
				# TODO: eventually make sure the split will not exceed the 
				# maximum allowed subsets
				if not len(ind['G'])>1:
					mutation_types.remove(mutation_type)
				else:
					mut_idx = sorted(
						np.random.choice(np.arange(len(ind['G'])), size=(2,), replace=False), reverse=True
					)
					mut_1_val = copy.deepcopy(ind['G'][mut_idx[0]])
					mut_2_val = copy.deepcopy(ind['G'][mut_idx[1]])
					new_Sg = sorted(mut_1_val + mut_2_val)
					for idx in mut_idx:
						del ind['G'][idx]
					ind['G'].append(new_Sg)
					ind['G'] = sorted(ind['G'])
					mutated=True
			else:
				pass
		return ind
		

	def custom_mutate_X_func(self, ind, indpb=0.3, continuous_scale=0.1, discrete_scale=0.1):
		""" mutation for X_func functional parameters
		"""
		
		assert len(ind['X_func']) == ind['Ng']
		G_mut_diff = len(ind['G']) - ind['Ng']
		
		if G_mut_diff < 0:
			# subset(s) were removed, need to reduce number of X_func
			remove_idx = sorted(
				np.random.choice(np.arange(ind['Ng']), size=(abs(G_mut_diff,)), replace=False), reverse=True
			)
			for idx in remove_idx:
				del ind['X_func'][idx]
			ind['Ng'] = len(ind['G']) # reset Ng
			
		elif G_mut_diff > 0:
			# subset(s) were added,  need to inflate the number of X_func
			if len(ind['X_func'])==1:
				duplicate_idx = [0]*G_mut_diff
			else:
				duplicate_idx = sorted(
					np.random.choice(np.arange(ind['Ng']), size=(G_mut_diff,), replace=False), reverse=True
				)
			duplicate_vals = [ind['X_func'][idx] for idx in duplicate_idx]
			ind['X_func'].extend([val for val in duplicate_vals])
			
			ind['Ng'] = len(ind['G']) # reset Ng
		
		# perform a Gaussian mutation
		for X_func_ix in range(len(ind['X_func'])):
			# TODO: could add another probability here...
			for param_ix in range(self.func_param_space_dim):

				if np.random.uniform() < indpb:

					param_type = self.func_param_space[param_ix].type
					
					if param_type == 'continuous':
						# Gaussian perturbation
						bound_low = self.bounds[0, param_ix]
						bound_high = self.bounds[1, param_ix]
						scale = (bound_high - bound_low) * continuous_scale
						ind['X_func'][X_func_ix][param_ix] += np.random.normal(loc=0.0, scale=scale)
						ind['X_func'][X_func_ix][param_ix] = self._project_bounds(
							ind['X_func'][X_func_ix][param_ix], bound_low, bound_high,
						)
						
					elif param_type == 'discrete':
						# add/substract an integer by rounding Gaussian perturbation
						bound_low = 0
						bound_high = len(self.func_param_space[param_ix].options)-1
						# if we have very few discrete variables, just move +/- 1
						if bound_high-bound_low < 10:
							delta = np.random.choice([-1, 1])
							ind['X_func'][X_func_ix][param_ix] += delta
						else:
							scale = (bound_high - bound_low) * discrete_scale
							delta = np.random.normal(loc=0.0, scale=scale)
							ind['X_func'][X_func_ix][param_ix] += np.round(delta, decimals=0)
						ind['X_func'][X_func_ix][param_ix] = self._project_bounds(
							ind['X_func'][X_func_ix][param_ix], bound_low, bound_high,
						)
						
					elif param_type == 'categorical':
						options = self.func_param_space[param_ix].options
						#num_options = float(self.param_ranges[param_ix]) # float so arnage returns doubles

						ind['X_func'][X_func_ix][param_ix] = np.random.choice(options)
					
						
					else:
						raise ValueError()
			
		return ind


	#-----------
	# EVOLUTION
	#-----------

	@staticmethod
	def _evolution(
			population, 
			toolbox, 
			halloffame,
			mutate_G_pb, 
			mutate_X_func_pb,
			mutate_X_func_indpb,
		):

		# size of hall of fame
		hof_size = len(halloffame.items) if halloffame.items else 0
		# Select the next generation individuals (allow for elitism)
		offspring = toolbox.select(population, len(population) - hof_size)
		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))
		
		for mutant in offspring:
			# mutation on G
			if np.random.uniform() < mutate_G_pb:
				toolbox.mutate_G(mutant)
				# if G is mutated, mutate X_func immediately
				toolbox.mutate_X_func(mutant, indpb=mutate_X_func_indpb) 
				del mutant.fitness.values
		
		# option for extra mutation on X_func
		for mutant in offspring:
			# mutation on X_func
			if np.random.uniform() < mutate_X_func_pb:
				toolbox.mutate_X_func(mutant, indpb=mutate_X_func_indpb)
				del mutant.fitness.values

		return offspring



	# TODO: implement constrained evolution method for medusa....


	#-------------------
	# MAIN OPTIMIZATION
	#-------------------

	def _optimize(
			self, 
			num_inds:int=100,
			num_gen:int=100,
			mutate_G_pb:float=0.25, 
			mutate_X_func_pb:float=0.25,
			mutate_X_func_indpb:float=0.25,
			halloffame_frac:float=0.1,
			use_best_init_pop:bool=False, 
			show_progress:bool=True, 
	):
		
		creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) 
		creator.create('Individual', dict, fitness=creator.FitnessMin)

		toolbox = base.Toolbox()
		if use_best_init_pop: 
			# pre-select population by measuring fitness
			toolbox.register('population', self.init_population_best)
		else:
			# use random intial population
			toolbox.register('population', self.init_population_random)

		toolbox.register('evaluate', self.acquisition)

		# mutation/selection opertations operations
		# toolbox.register("mutate_X_func", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
		toolbox.register("select", tools.selTournament, tournsize=3)

		toolbox.register('mutate_G', self.custom_mutate_G)
		toolbox.register('mutate_X_func', self.custom_mutate_X_func, indpb=mutate_X_func_indpb)

		# initialize population 
		population = toolbox.population(num_inds=num_inds)

	
		# Evaluate the entire population
		fitnesses = map(toolbox.evaluate, population)
		for ind, fit in zip(population, fitnesses):
			ind.fitness.values = fit
		
		# create hall of fame
		num_elites = int(round(halloffame_frac * len(population), 0))  # 5% of elite individuals
		halloffame = tools.HallOfFame(num_elites)  # hall of fame with top individuals
		halloffame.update(population)
		
		# register some statistics and create logbook
		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)

		logbook = tools.Logbook()
		logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
		record = stats.compile(population) if stats else {}
		logbook.record(gen=0, nevals=len(population), **record)

		for gen in track(
			range(1, num_gen+1),
			total=num_gen,
			description="Optimizing proposals...",
			transient=False,
		):
			# one generation evolution
			offspring = self._evolution(
				population=population, 
				toolbox=toolbox, 
				halloffame=halloffame,
				mutate_G_pb=mutate_G_pb, 
				mutate_X_func_pb=mutate_X_func_pb,
				mutate_X_func_indpb=mutate_X_func_indpb,
			)
		
			# Evaluate the individual4s with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			
			fitnesses = map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit
				
			# add the best back to population
			offspring.extend(halloffame.items)
			# Update the hall of fame with the generated individuals
			halloffame.update(offspring)

			# The population is entirely replaced by the offspring
			population[:] = offspring    
			
			# Append the current generation statistics to the logbook
			record = stats.compile(population) if stats else {}
			logbook.record(gen=gen, nevals=len(invalid_ind), **record)

		# DEAP cleanup
		del creator.FitnessMin
		del creator.Individual

		# select best recommendations and return them as parameter vectors
		acqf_vals = [ind.fitness for ind in population] #[self.acquisition(x)[0] for x in np.array(population)

		best_idxs = np.argsort(acqf_vals)[:self.batch_size]
	
		# list of dictionaries containing the best recommended batch samples
		# X_func and G 
		best_batch_pop = [population[idx] for idx in best_idxs]

		# print('best_batch_pop : ', best_batch_pop)

		#--------------------
		# acquisition part 2 
		#--------------------
		# select the set of functional parameters to recommend
		# TODO: needs to be extended to batched case

		# deinxdexify the X_funcs
		X_funcs_deindex = self.deindexify(best_batch_pop[0]['X_func'])
		# select the option to measure using variance-based sampling
		select_X_func, select_si = self.acqf.acqf_var(X_funcs_deindex, best_batch_pop[0]['G'])

		# reverse scale the functional parameters
		select_X_func = reverse_normalize(
			select_X_func, 
			self.params_obj._mins_x[self.functional_dims],
			self.params_obj._maxs_x[self.functional_dims],
		)


		# project back to Olympus ParameterVector
		return_params_dict = {}
		func_param_iter = 0
		for param_ix, param in enumerate(self.param_space):

			if param_ix in self.general_parameters:
				# general parameter, get option
				opt = param.options[select_si]
				return_params_dict[param.name] = opt
			else:
				# functional parameter, add the value
				if param.type == 'continuous':
					# project to olympus bounds
					return_params_dict[param.name] = self._project_bounds(
						select_X_func[func_param_iter], param.low, param.high,
					)
				else:
					return_params_dict[param.name] = select_X_func[func_param_iter]

				func_param_iter+=1

		#print(return_params_dict)
		return_params = [ParameterVector().from_dict(return_params_dict, self.param_space)]
		#print(return_params)

		return return_params



#-----------------
# EXTRA FUNCTIONS
#------------------

def collect_results(logbooks):

	min_fitness_vals = []
	num_evals = []

	for logbook in logbooks:
		run_min_fitness_vals = [gen['min'] for gen in logbook]
		run_num_evals = [gen['nevals'] for gen in logbook]
		
		min_fitness_vals.append(run_min_fitness_vals)
		num_evals.append(run_num_evals)
		
	min_fitness_vals = np.array(min_fitness_vals)
	num_evals = np.array(num_evals)

	return min_fitness_vals, num_evals



		