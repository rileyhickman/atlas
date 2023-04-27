import os
import pickle
import numpy as np

from olympus import Campaign

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
from atlas.optimizers.gp import BoTorchPlanner
from atlas.optimizers.dynamic_space.planner import DynamicSSPlanner

from atlas import Logger
from atlas.optimizers.params import Parameters, ParameterContinuous
from atlas.optimizers.utils import (
    cat_param_to_feat,
    forward_normalize,
    forward_standardize,
    get_cat_dims,
    get_fixed_features_list,
    infer_problem_type,
    propose_randomly,
    reverse_normalize,
    reverse_standardize,
)

import pathlib
import utils.functions as functions


# Declare the model one wants to investigate
bm_function = 'Levy10'

NUM_RUNS = 10 #max_exp- the max number of experiments
BUDGET = 330 #max_iter- the max number of evaluations per experiment

PATH: pathlib.Path = pathlib.Path().resolve()
RUNSDIR = os.path.join(PATH, "examples/dynamic_search_space/runs")
SURFACE_KIND= bm_function
model_kind = "Botorch"



# Load the synthetic functions
if (bm_function == 'Hartman3'):
    myfunction = functions.hartman_3d()
elif (bm_function == 'Hartman6'):
    myfunction = functions.hartman_6d()
elif (bm_function == 'Beale'):
    myfunction = functions.beale()
elif (bm_function == 'Ackley10'):
    myfunction = functions.ackley(10)
elif (bm_function == 'Levy3'):
    myfunction = functions.Levy(3)
elif (bm_function == 'Levy10'):
    myfunction = functions.Levy(10)
elif (bm_function == 'Eggholder'):
    myfunction = functions.egg_holder()
else:
    raise AssertionError("Unexpected value of 'bm_function'!")
func = myfunction.func
n_init_points = 3*myfunction.input_dim


# -----------------------------
# Instantiate surface
# -----------------------------
import time 

run_ix = 0
while run_ix < NUM_RUNS:

    with open("runlog.log", "a") as f:
        f.write(f"now starting run {run_ix}\n")

    print(f"now starting run {run_ix}")
    time.sleep(3)

    # try:
    campaign = Campaign()
    param_space = ParameterSpace()
    for i in range(10):
        param = ParameterContinuous(
        name=f"param_{i}", low=-10, high=10
        )
        param_space.add(param)

    campaign.set_param_space(param_space)

    
    planner = BoTorchPlanner(
        iter_mul=BUDGET,
        epsilon=0.05,
        goal="maximize",
        feas_strategy="naive-0",
        init_design_strategy="lhs",
        num_init_design=n_init_points,
        batch_size=1,
        use_descriptors=False
    )

    planner.param_space = campaign.param_space

    planner._set_param_space(campaign.param_space)

    # start the optimization experiment
    iteration = 0
    # optimization loop
    while len(campaign.values) < BUDGET:
        samples = planner.recommend(campaign.observations)

        for sample in samples:
            sample_arr = sample.to_array()

            measurement = func(sample_arr)

            print(F"MEASUREMENT:{measurement}")

            campaign.add_observation(sample_arr, measurement)

    pickle.dump(
                    {
                        "params": campaign.params,
                        "values": campaign.values,
                    },
                    open(
                        f"runs/run_{model_kind}_{SURFACE_KIND}_29.pkl",
                        "wb",
                    ),
                )
    run_ix +=1
        


    
    


