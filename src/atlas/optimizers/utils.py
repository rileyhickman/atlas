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
	elif param_types.count('discrete')==len(param_types):
		problem_type = 'fully_discrete'
	elif all([
		'continuous' in param_types,
		'categorical' in param_types,
		'discrete' not in param_types,
		]):
		problem_type = 'mixed'
	elif all([
		'continuous' not in param_types,
		'categorical' in param_types,
		'discrete' in param_types,
		]):
		problem_type = 'mixed_dis_cat'
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

def get_fixed_features_list(param_space, has_descriptors):
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
			ohe.append(cat_param_to_feat(obj, val, has_descriptors))
		current_avail_feat.append(np.concatenate(ohe))
		current_avail_cat.append(elem)

	# make list
	for feat in current_avail_feat:
		fixed_features_list.append(
			{dim_ix:feat[ix] for ix, dim_ix in enumerate(cat_dims)}
		)


	return fixed_features_list




def cat_param_to_feat(param, val, has_descriptors):
	''' convert the option selection of a categorical variable (usually encoded
	as a string) to a machine readable feature vector
	Args:
		param (object): the categorical olympus parameter
		val (str): the value of the chosen categorical option
	'''
	# get the index of the selected value amongst the options

	arg_val = param.options.index(val)
	if not has_descriptors:
		# no provided descriptors, resort to one-hot encoding
		feat = np.zeros(len(param.options))
		feat[arg_val] += 1.
	else:
		# we have descriptors, use them as the features
		feat = param.descriptors[arg_val]
	return feat


def propose_randomly(num_proposals, param_space, has_descriptors):
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
				feat = cat_param_to_feat(param, p, has_descriptors)
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
		if bools.count(True)>=1:
			idx = np.where(bools)[0][0]
		else:
			print('no matches ... ')
			quit()
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



def param_vector_to_dict(param_vector, param_names, param_options, param_types):
    """parse single sample and return a dict"""
    param_dict = {}
    for param_index, param_name in enumerate(param_names):
        param_type = param_types[param_index]

        if param_type == 'continuous':
            param_dict[param_name] = param_vector[param_index]

        elif param_type == 'categorical':
            options = param_options[param_index]
            selected_option_idx = int(param_vector[param_index])
            selected_option = options[selected_option_idx]
            param_dict[param_name] = selected_option

        elif param_type == 'discrete':
            options = param_options[param_index]
            selected_option_idx = int(param_vector[param_index])
            selected_option = options[selected_option_idx]
            param_dict[param_name] = selected_option
    return param_dict


def flip_source_tasks(source_tasks):
    ''' flip the sign of the source tasks if the
    optimization goal is maximization
    '''
    flipped_source_tasks = []
    for task in source_tasks:
        flipped_source_tasks.append(
            {
                'params': task['params'],
                'values': -1*task['values'],
            }
        )

    return flipped_source_tasks


class Scaler:

    SUPP_TYPES = ['standardization', 'normalization', 'identity']

    ''' scaler for source data
    Args:
        type (str): scaling type, supported are standardization or
                    normalization
        data (str): data type, either params or values
    '''
    def __init__(self, param_type, value_type):
        if not param_type in self.SUPP_TYPES:
            raise NotImplementedError
        else:
            self.param_type = param_type

        if not value_type in self.SUPP_TYPES:
            raise NotImplementedError
        else:
            self.value_type = value_type

        self.is_fit = False


    def _compute_stats(self, source_tasks):
        ''' computes the stats for an entire set of source tasks
        '''
        # join source tasks params
        all_source_params = []
        all_source_values = []
        for task in source_tasks:
            all_source_params.append(task['params'])
            all_source_values.append(task['values'])
        all_source_params = np.concatenate(np.array(all_source_params), axis=0)
        all_source_values = np.concatenate(np.array(all_source_values), axis=0)

        # make sure these are 2d
        assert len(all_source_params.shape)==2
        assert len(all_source_values.shape)==2

        # compute stats for parameters
        param_stats = {}
        if self.param_type == 'normalization':
            param_stats['max'] = np.amax(all_source_params, axis=0)
            param_stats['min'] = np.amin(all_source_params, axis=0)
        elif self.param_type == 'standardization':
            # need the mean and the standard deviation
            param_stats['mean'] = np.mean(all_source_params, axis=0)
            std = np.std(all_source_params, axis=0)
            param_stats['std'] = np.where(std == 0., 1., std)
        self.param_stats = param_stats

        # compute stats for values
        value_stats = {}
        if self.value_type == 'normalization':
            value_stats['max'] = np.amax(all_source_values, axis=0)
            value_stats['min'] = np.amin(all_source_values, axis=0)
        elif self.value_type == 'standardization':
            # need the mean and the standard deviation
            value_stats['mean'] = np.mean(all_source_values, axis=0)
            std = np.std(all_source_values, axis=0)
            value_stats['std'] = np.where(std == 0., 1., std)
        self.value_stats = value_stats


    def fit_transform_tasks(self, source_tasks):
        ''' compute stats for a set of source tasks
        '''
        # register the stats
        self._compute_stats(source_tasks)

        transformed_source_tasks = []

        for task in source_tasks:
            trans_task = {}
            # params
            if self.param_type == 'normalization':
                trans_task['params'] = self.normalize(
                    task['params'], self.param_stats['min'], self.param_stats['max'], 'forward'
                )
            elif self.param_type == 'standardization':
                trans_task['params'] = self.standardize(
                    task['params'], self.param_stats['mean'], self.param_stats['std'], 'forward'
                )
            elif self.param_type == 'identity':
                trans_task['params'] = self.identity(task['params'], 'forward')
            # values
            if self.value_type == 'normalization':
                trans_task['values'] = self.normalize(
                    task['values'], self.value_stats['min'], self.value_stats['max'], 'forward'
                )
            elif self.value_type == 'standardization':
                trans_task['values'] = self.standardize(
                    task['values'], self.value_stats['mean'], self.value_stats['std'], 'forward'
                )
            elif self.value_type == 'identity':
                trans_task['values'] = self.identity(task['values'], 'forward')

            transformed_source_tasks.append(trans_task)

        return transformed_source_tasks

    def identity(self, x, direction):
        ''' identity transformation
        '''
        return x

    def standardize(self, x, mean, std, direction):
        ''' standardize the data given parameters
        '''
        if direction == 'forward':
            return (x - mean) / std
        elif direction == 'reverse':
            return x*std + mean


    def normalize(self, x, min, max, direction):
        ''' normalize the data given parameters
        '''
        if direction == 'forward':
            return (x - min) / (max - min)
        elif direction == 'reverse':
            return x*(max - min) + min


    def transform_tasks(self, tasks):
        ''' transform a set of tasks
        '''
        transformed_source_tasks = []
        for task in tasks:
            trans_task = {}
            # params
            trans_task['params'] = self.transform(task['params'], type='params')
            # values
            trans_task['values'] = self.transform(task['values'], type='values')
        transformed_source_tasks.append(trans_task)

        return transformed_source_tasks


    def transform(self, sample, type):
        ''' transforms a sample
        '''
        # make sure this sample is 2d array
        assert len(sample.shape)==2

        if type == 'params':
            if self.param_type == 'normalization':
                return self.normalize(sample, self.param_stats['min'], self.param_stats['max'], 'forward')
            elif self.param_type == 'standardization':
                return self.standardize(sample, self.param_stats['mean'], self.param_stats['std'], 'forward')
            elif self.param_type == 'identity':
                return self.identity(sample, 'forward')
        elif type == 'values':
            if self.value_type == 'normalization':
                return self.normalize(sample, self.value_stats['min'], self.value_stats['max'], 'forward')
            elif self.value_type == 'standardization':
                return self.standardize(sample, self.value_stats['mean'], self.value_stats['std'], 'forward')
            elif self.value_type == 'identity':
                return self.identity(sample, 'forward')



    def inverse_transform(self, sample, type):
        ''' perform inverse transformation
        '''
        # make sure this sample is 2d array
        assert len(sample.shape)==2

        if type == 'params':
            if self.param_type == 'normalization':
                return self.normalize(sample, self.param_stats['min'], self.param_stats['max'], 'forward')
            elif self.param_type == 'standardization':
                return self.standardize(sample, self.param_stats['mean'], self.param_stats['std'], 'forward')
            elif self.param_type == 'identity':
                return self.identity(sample, 'reverse')
        elif type == 'values':
            if self.value_type == 'normalization':
                return self.normalize(sample, self.value_stats['min'], self.value_stats['max'], 'forward')
            elif self.value_type == 'standardization':
                return self.standardize(sample, self.value_stats['mean'], self.value_stats['std'], 'forward')
            elif self.value_type == 'identity':
                return self.identity(sample, 'reverse')
