#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.special import rel_entr
import matplotlib.pyplot as plt

from atlas.optimizers.gp.planner import BoTorchPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical


from head import opentrons, AmplitudePhaseDistance
import warnings
warnings.filterwarnings('ignore')


#------------------
# helper functions
#------------------

class Simulator:
    ''' simulator class for Gaussian spectra/distributions'''
    def __init__(self):
        self.domain = np.linspace(-5,5,1000)
        
    def generate(self, mu, sig):
        scale = 1/(np.sqrt(2*np.pi)*sig)
        return scale*np.exp(-np.power(self.domain - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def process_batch(self, Cb, fname):
        out = []
        for c in Cb:
            out.append(self.generate(*c))
        out = np.asarray(out)
        df = pd.DataFrame(out.T, index=self.domain)
        df.to_excel(fname, engine='openpyxl')
        
        return 
    
    def make_target(self, ct):
        return self.domain, self.generate(*ct)
    
    
def APdist(f1,f2,xt):
    ''' distance function (this value is to me maximized,
    i.e. less negative value describes more similar distributions) '''
    da, dp = AmplitudePhaseDistance(f1,f2,xt)
    
    return -(da+dp)

def KLdiv(f1,f2,xt):
    ''' KL divergence
    '''
    assert len(f1)==len(f2)==len(xt)
    upper_ = 1e10
    kl = np.sum(rel_entr(f1,f2))
    if kl > upper_ or kl==np.inf:
        kl = upper_

    return kl

def normalize(f1):
    return (f1-np.amin(f1))/(np.amax(f1)-np.amin(f1))

#---------------------
# experiment settings
#---------------------

target_params = np.array([-2, 0.9])

range_mu = [-5, 5]
range_sigma = [0.3, 3.5]

with_descriptors = False
budget = 50
repeats = 40
random_seed = None
plot=True

# make simulator instance and target
sim = Simulator()
xt, yt = sim.make_target(target_params)

data_all_repeats = []


for num_repeat in range(repeats):

    param_space = ParameterSpace()
    param_space.add(
        ParameterContinuous(
            name='mu', 
            low=range_mu[0],
            high=range_mu[1],
        )
    )
    param_space.add(
        ParameterContinuous(
            name='sigma',
            low=range_sigma[0],
            high=range_sigma[1],

        )
    )

    planner = BoTorchPlanner(
		goal='minimize',
		feas_strategy='fca',
		feas_param=0.2,
        batch_size=1,
        acquisition_type='ei',
        acquisition_optimizer_kind='gradient',

	)
    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    if plot:

        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        plt.ion()

    best_dist = 1e10
          
    for num_iter in range(budget):
        print(f'===============================')
        print(f'   Repeat {num_repeat+1} -- Iteration {num_iter+1}')
        print(f'===============================')

        samples = planner.recommend(campaign.observations)

        sample = samples[0]

        xq, yq = sim.make_target(
            np.array([sample['mu'], sample['sigma']])
        )

        #measurement = -APdist(yt, yq, xt) # take negative, minimize value
        measurement = KLdiv(yt,yq,xt)

        print(f'MU : {round(sample["mu"],4)} SIGMA : {round(sample["sigma"], 4)}\tDIST : {measurement}')

        campaign.add_observation(sample, measurement)

        if plot:
            for ax in axes:
                ax.clear()

            # norm
            axes[0].plot(xt, yt, label='target')
            axes[0].plot(xq, yq, label='current query')
            axes[0].set_title(f'dist : {round(measurement,4)}')
            axes[0].legend()

            if measurement <= best_dist:
                best_yq = yq 
                best_dist = measurement
            
            axes[1].plot(xt, yt, label='target')
            axes[1].plot(xq, best_yq, label='best query')
            axes[1].set_title(f'best dist : {round(best_dist,4)}')
            axes[1].legend()

            plt.tight_layout()
            plt.pause(2.)
        

    # store the results into a DataFrame
    mu_col = campaign.observations.get_params()[:, 0]
    sigma_col = campaign.observations.get_params()[:, 1]
    dist_col = campaign.observations.get_values(as_array=True)

    data = pd.DataFrame({'mu': mu_col, 'sigma': sigma_col, 'dist': dist_col})
    data_all_repeats.append(data)

    pickle.dump(data_all_repeats, open('results.pkl', 'wb'))




    

