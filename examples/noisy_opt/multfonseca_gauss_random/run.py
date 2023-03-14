#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd
import torch 
import matplotlib.pyplot as plt
import seaborn as sns


from olympus.surfaces import Surface
from olympus.campaigns import Campaign
from olympus.objects import ParameterContinuous, ParameterCategorical, ParameterVector
from olympus.utils.misc import get_hypervolume
from olympus.planners import RandomSearch

from olympus.utils.misc import get_pareto, get_pareto_set
from olympus.noises import GaussianNoise, GammaNoise


from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated


from atlas.optimizers.qnehvi.planner import qNEHVIPlanner



def add_noise(measurements, noise_models):
    ''' add noises to parameter vector '''
    noisy_param_vects = []
    for measurement in measurements:
        noisy_dict = {}
        for obj_ix, (obj_name, obj_val) in enumerate(measurement.to_dict().items()):

            noisy_obj_val = noise_models[obj_ix]._add_noise(obj_val)
            noisy_dict[obj_name] = noisy_obj_val
        
        noisy_param_vects.append(ParameterVector().from_dict(noisy_dict))

    return noisy_param_vects



#-------------------
# experiment config
#-------------------

budget = 30
num_repeats = 100
noise_model = 'Gaussian'
scale = 0.05

noise_models = [
    GaussianNoise(scale=scale),
    GaussianNoise(scale=scale),
]

surface = Surface(kind='MultFonseca',value_dim=2)

data_all_repeats = []

for repeat in range(num_repeats):

    print(f'\n>>> Starting repeat {repeat+1}/{num_repeats} ...\n')

    planner = RandomSearch()

    planner.set_param_space(surface.param_space)

    # noisy
    campaign = Campaign()
    campaign.set_param_space(surface.param_space)
    campaign.set_value_space(surface.value_space)

    # noiseless 
    noiseless_campaign = Campaign()
    noiseless_campaign.set_param_space(surface.param_space)
    noiseless_campaign.set_value_space(surface.value_space)

    hypervolumes = []

    while len(campaign.observations.get_values()) < budget:

        samples = planner.recommend(campaign.observations)

        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface.run(sample_arr, return_paramvector=True)
            noisy_measurement = add_noise(measurement, noise_models)
            campaign.add_observation(sample_arr, noisy_measurement)
            noiseless_campaign.add_observation(sample_arr, measurement)

            print('SAMPLE : ', sample)
            print('MEASUREMENT : ', measurement)
            print('NOISY MEASUREMENT : ', noisy_measurement)
            print('')

        # compute dominated hypervolume
        volume = get_hypervolume(
                noiseless_campaign.observations.get_values(),
                np.array([1.,1.])
        )
        hypervolumes.append(volume)

        print('HYPERVOLUME : ', volume)


    # store the results into a DataFrame
    param0_col = campaign.observations.get_params()[:, 0]
    param1_col = campaign.observations.get_params()[:, 1]
    obj0_col = campaign.observations.get_values(as_array=True)[:,0]
    obj1_col = campaign.observations.get_values(as_array=True)[:,1]
    noiseless_obj0_col = noiseless_campaign.observations.get_values(as_array=True)[:,0]
    noiseless_obj1_col = noiseless_campaign.observations.get_values(as_array=True)[:,1]
    data = pd.DataFrame({'param0': param0_col, 'param1': param1_col, 'obj0': obj0_col, 'obj1': obj1_col,
                         'noiseless_obj0': noiseless_obj0_col, 'noiseless_obj1': noiseless_obj1_col,
                         'hypervolume': hypervolumes})
    data_all_repeats.append(data)

    pickle.dump(data_all_repeats, open('results.pkl', 'wb'))
