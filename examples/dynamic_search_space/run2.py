import os
import pickle
import sys
import time
from copy import deepcopy
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import gpytorch
import numpy as np
import olympus
from olympus import Campaign
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel
from gpytorch.priors import MultivariateNormalPrior

from botorch.fit import fit_gpytorch_model
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_mixed,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace
from olympus.planners import AbstractPlanner, CustomPlanner, Planner
from olympus.scalarizers import Scalarizer
from atlas.optimizers.base import BasePlanner
from atlas.optimizers.dynamic_space.planner import DynamicSSPlanner

from atlas import Logger
from atlas.optimizers.params import Parameters, ParameterContinuous
from atlas.optimizers.utils import (
    cat_param_to_feat,
    forward_normalize,
    forward_standardize,ial
    get_cat_dims,
    get_fixed_features_list,
    infer_problem_type,
    propose_randomly,
    reverse_normalize,
    reverse_standardize,
)

def hm3(x):
    # the hartmann3 function (3D)
    # https://www.sfu.ca/~ssurjano/hart3.html
    # parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array(
        [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ]
    )
    x = x.reshape(x.shape[0], 1, -1)
    B = x - P
    B = B**2
    exponent = A * B
    exponent = np.einsum("ijk->ij", exponent)
    C = np.exp(-exponent)
    hm3 = -np.einsum("i, ki", alpha, C)
    # normalize
    #mean = -0.93
    #std = 0.95
    #hm3 = 1 / std * (hm3 - mean)
    # maximize
    # hm3 = -hm3
    return hm3


import pathlib
from olympus import Surface

PATH: pathlib.Path = pathlib.Path().resolve()
RUNSDIR = os.path.join(PATH, "examples/dynamic_search_space/runs")
MODELS = [
    "RandomSearch",
    #'Gpyopt',
    "Botorch",
    "RGPE",
    "DKT",
    "Dynamic"
]

PLOTPATH: pathlib.Path = "/Users/maozer/VSCodeProjects/atlas/examples/dynamic_search_space/plots"
NUM_RUNS = 1 #max_exp- the max number of experiments
BUDGET = 30 #max_iter- the max number of evaluations per experiment
model_kind = "Dynamic"


# -----------------------------
# Instantiate surface
# -----------------------------
#surface = Surface(kind='DiscreteAckley', param_dim=2)
surface = hm3

campaign = Campaign()
param_space = ParameterSpace()
# add 3 continuous Parameters
param_0 = ParameterContinuous(
    name="param_0", low=0.0, high=1.0
)
param_1 = ParameterContinuous(
    name="param_1", low=0.0, high=1.0
)
param_2 = ParameterContinuous(
    name="param_2", low=0.0, high=1.0
)
param_space.add(param_0)
param_space.add(param_1)
param_space.add(param_2)


campaign.set_param_space(param_space)

np.random.seed(0)
box_len = 0.2
bounds_all = np.array([[0, 1],
                        [0, 1],
                        [0 ,1]])
b_init_center = np.zeros((len(bounds_all), 1))
for i in range(len(bounds_all)):
    b_init_center[i] = np.random.uniform(bounds_all[i][0]+box_len/2,
                                            bounds_all[i][1]-box_len/2)
b_init_lower = b_init_center - box_len/2
b_init_upper = b_init_center + box_len/2
bounds_user = np.asarray([b_init_lower.ravel(), b_init_upper.ravel()]).T

bounds = bounds_user.copy()

init_func_param_space = ParameterSpace()
for i in range (3):
    param = ParameterContinuous(
    name=f"param_{i}", 
    low=float(b_init_lower[i]), 
    high=float(b_init_upper[i])
    )
    init_func_param_space.add(param)



# Generate and normalize input, ouput
temp = [np.random.uniform(x[0], x[1], size=9) for x in bounds]
temp = np.asarray(temp)
temp = temp.T
X_init = list(temp.reshape((9, -1)))

for x in X_init:
    measurement = surface(x.reshape((1, x.shape[0])))
    campaign.add_observation(x, measurement)

planner = DynamicSSPlanner(
    iter_mul=BUDGET,
    epsilon=0.05,
    goal="minimize",
    feas_strategy="naive-0",
    init_design_strategy="lhs",
    num_init_design=9,
    batch_size=1,
    use_descriptors=False
)

planner.set_param_space(campaign.param_space, init_func_param_space)
planner._set_param_space(campaign.param_space)



# start the optimization experiment
iteration = 0
# optimization loop
while len(campaign.values) < BUDGET:
    print("############################")

    print(f"\nITERATION : {iteration+1}\n")

    samples = planner.recommend(campaign.observations)
    print(f"SAMPLES : {samples}")

    for sample in samples:
        sample_arr = sample.to_array()
        # measurement = surface.run(
        #     sample_arr.reshape((1, sample_arr.shape[0]))
        # )
        measurement = surface(sample_arr.reshape((1, sample_arr.shape[0])))

        print(f"func_param_space:{planner.func_param_space}")
        print(F"MEASUREMENT:{measurement}")
        campaign.add_observation(sample_arr, measurement)
        
    iteration += 1
    print("############################")


print(f"run completed")

print(campaign.observations.get_values())



