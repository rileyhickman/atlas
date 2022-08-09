#!/usr/bin/env python

import numpy as np
import torch
import itertools



def infer_problem_type(param_space):
	''' infer the parameter space from Olympus. The three possibilities are
	"fully_continuous", "mixed" or "fully_categorical"
	Args:
		param_space (obj): Olympus parameter space object
	'''
	param_types = [p.type for p in param_space]
	if param_types.count('continuous') == len(param_types):
		problem_type = 'fully_continuous'
	elif param_types.count('categorical') == len(param_types):
		problem_type = 'fully_categorical'
	elif np.logical_and(
		'continuous' in param_types,
		'categorical' in param_types
		):
		problem_type = 'mixed'
	return problem_type


def get_cat_dims(param_space):
	dim = 0
	cat_dims = []
	for p in param_space:
		if p.type == 'categorical':
			# this will only work for OHE variables
			d = np.arange(dim, dim+len(p.options))
			cat_dims.extend(list(d))
		else:
			dim+=1

	return cat_dims

def get_fixed_features_list(param_space):
	dim = 0
	fixed_features_list = []
	cat_dims = []
	cat_params = []
	for p in param_space:
		if p.type == 'categorical':
			dims = np.arange(dim, dim+len(p.options))
			cat_dims.extend(dims)
			cat_params.append(p)
		else:
			dim+=1

	param_options = [p.options for p in cat_params]
	cart_product = list(itertools.product(*param_options))
	cart_product = [list(elem) for elem in cart_product]

	current_avail_feat  = []
	current_avail_cat = []
	for elem in cart_product:
		# convert to ohe and add to currently available options
		ohe = []
		#for val, obj in zip(elem, param_space):
		for val, obj in zip(elem, cat_params):
			ohe.append(cat_param_to_feat(obj, val))
		current_avail_feat.append(np.concatenate(ohe))
		current_avail_cat.append(elem)
	
	# make list
	for feat in current_avail_feat:
		fixed_features_list.append(
			{dim_ix:feat[ix] for ix, dim_ix in enumerate(cat_dims)}
		)


	return fixed_features_list




def cat_param_to_feat(param, val):
	''' convert the option selection of a categorical variable (usually encoded
	as a string) to a machine readable feature vector
	Args:
		param (object): the categorical olympus parameter
		val (str): the value of the chosen categorical option
	'''
	# get the index of the selected value amongst the options

	arg_val = param.options.index(val)	
	if np.all([d==None for d in param.descriptors]):
		# no provided descriptors, resort to one-hot encoding
		feat = np.zeros(len(param.options))
		feat[arg_val] += 1.
	else:
		# we have descriptors, use them as the features
		feat = param.descriptors[arg_val]
	return feat


def propose_randomly(num_proposals, param_space):
	''' Randomly generate num_proposals proposals. Returns the numerical
	representation of the proposals as well as the string based representation
	for the categorical variables
	Args:
		num_proposals (int): the number of random proposals to generate
	'''
	proposals = []
	raw_proposals = []
	for propsal_ix in range(num_proposals):
		sample = []
		raw_sample = []
		for param_ix, param in enumerate(param_space):
			if param.type == 'continuous':
				p = np.random.uniform(param.low, param.high, size=None)
				sample.append(p)
				raw_sample.append(p)
			elif param.type == 'discrete':
				num_options = int(((param.high-param.low)/param.stride)+1)
				options = np.linspace(param.low, param.high, num_options)
				p = np.random.choice(options, size=None, replace=False)
				sample.append(p)
				raw_sample.append(p)
			elif param.type == 'categorical':
				options = param.options
				p = np.random.choice(options, size=None, replace=False)
				feat = cat_param_to_feat(param, p)
				sample.extend(feat)  # extend because feat is vector
				raw_sample.append(p)
		proposals.append(sample)
		raw_proposals.append(raw_sample)
	proposals = np.array(proposals)
	raw_proposals = np.array(raw_proposals)

	return proposals, raw_proposals



def forward_standardize(data, means, stds):
	''' forward standardize the data
	'''
	return (data - means) / stds

def reverse_standardize(data, means, stds):
	''' un-standardize the data
	'''
	return (data*stds) + means

def forward_normalize(data, min_, max_):
	''' forward normalize the data
	'''
	ixs = np.where(np.abs(max_-min_)<1e-10)[0]
	if not ixs.size == 0:
		max_[ixs]=np.ones_like(ixs)
		min_[ixs]=np.zeros_like(ixs)
	return (data - min_) / (max_ - min_)

def reverse_normalize(data, min_, max_):
	''' un-normlaize the data
	'''
	ixs = np.where(np.abs(max_-min_)<1e-10)[0]
	if not ixs.size == 0:
		max_[ixs]=np.ones_like(ixs)
		min_[ixs]=np.zeros_like(ixs)
	return data * (max_ - min_) + min_


def project_to_olymp(
	results,
	param_space,
	has_descriptors=False,
	choices_feat=None,
	choices_cat=None,
	):
	''' project an acquisition function result numpy array to an
	Olympus param vector to be returned by the planners _ask method
	'''
	olymp_samples = {}
	if has_descriptors:
		# simply look up the index
		# TODO: this is kind of a hack, will fail in some cases...
		# consider cleaning this up
		bools = [torch.all(results.float()==choices_feat[i, :].float()) for i in range(choices_feat.shape[0])]
		assert bools.count(True)==1
		idx = np.where(bools)[0][0]
		sample = choices_cat[idx]
		for elem, name in zip(sample, [p.name for p in param_space]):
			olymp_samples[name] = elem

		return olymp_samples

	idx_counter = 0
	for param_ix, param in enumerate(param_space):
		if param.type == 'continuous':
			# if continuous, check to see if the proposed param is
			# within bounds, if not, project in
			sample = results[idx_counter]
			if sample > param.high:
				sample = param.high
			elif sample < param.low:
				sample = param.low
			else:
				pass
			idx_counter += 1
		elif param.type == 'categorical':
			if has_descriptors:
				pass
			else:
				# if categorical, scan the one-hot encoded portion
				cat_vec = results[idx_counter:idx_counter+len(param.options)]
				#argmin = get_closest_ohe(cat_vec)
				argmax = np.argmax(cat_vec)
				sample = param.options[argmax]
				idx_counter += len(param.options)
		elif param.type == 'discrete':
			# TODO: discrete params not supported now
			pass
		# add sample to dictionary
		olymp_samples[param.name] = sample

	return olymp_samples

def get_closest_ohe(cat_vec):
	''' return index of closest ohe vector
	'''
	ohe_options = np.eye(cat_vec.shape[0])
	dists = np.sum(np.square(np.subtract(ohe_options,cat_vec)), axis=1)
	return np.argmin(dists)


def get_bounds(param_space, mins_x, maxs_x, has_descriptors):
	''' returns scaled bounds of the parameter space
	torch tensor of shape (# dims, 2) (low and upper bounds)
	'''
	bounds = []
	idx_counter = 0
	for param_ix, param in enumerate(param_space):
		if param.type == 'continuous':
			b = np.array([param.low, param.high])
			b = (b-mins_x[idx_counter]) / (maxs_x[idx_counter]-mins_x[idx_counter])
			bounds.append(b)
			idx_counter += 1
		elif param.type == 'categorical':
			if has_descriptors:
				bounds += [[np.amin(param.descriptors[opt_ix]), np.amax(param.descriptors[opt_ix])] for opt_ix in range(len(param.options))]
			else:
				bounds += [[0, 1] for _ in param.options]
			idx_counter += len(param.options)


	return torch.tensor(np.array(bounds)).T.float()

