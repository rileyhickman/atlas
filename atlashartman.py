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
from atlas.optimizers.dynamic_space.planner_all_expandable_dims_at_once_option import Dynamic_Flexible_Planner

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
from olympus import Surface

# Declare the model one wants to investigate
bm_function = 'Hartman3'

NUM_RUNS = 30 #max_exp- the max number of experiments
BUDGET = 39 #max_iter- the max number of evaluations per experiment

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
SURFACE_KIND= bm_function
model_kind = "Dynamic"


import utils.functions as functions

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
bounds_all = np.asarray(myfunction.bounds)
box_len = 0.2*np.max(bounds_all[:, 1] - bounds_all[:, 0]) # Box length to be 20% of the original box
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

    np.random.seed(run_ix)
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
    temp = [np.random.uniform(x[0], x[1], size=n_init_points) for x in bounds]
    temp = np.asarray(temp)
    temp = temp.T
    X_init = list(temp.reshape((n_init_points, -1)))

    for x in X_init:
        measurement = func(x.reshape((1, x.shape[0])))
        campaign.add_observation(x, measurement)

    planner = Dynamic_Flexible_Planner(
        iter_mul=BUDGET,
        epsilon=0.05,
        goal="maximize",
        feas_strategy="naive-0",
        init_design_strategy="lhs",
        expandable_dims=[0,1],
        num_init_design=n_init_points,
        batch_size=1,
        use_descriptors=False
    )

    planner.set_param_space(campaign.param_space, init_func_param_space)
    planner._set_param_space(campaign.param_space)



    # start the optimization experiment
    iteration = 0
    # optimization loop
    while len(campaign.values) < BUDGET:
        samples = planner.recommend(campaign.observations)
        print(f"SAMPLES : {samples}")

        for sample in samples:
            sample_arr = sample.to_array()

            # measurement = surface.run(
            #     sample_arr.reshape((1, sample_arr.shape[0]))
            # )
            measurement = func(sample_arr)

            print(F"MEASUREMENT:{measurement}")

            campaign.add_observation(sample_arr, measurement)

    pickle.dump(
                    {
                        "params": campaign.params,
                        "values": campaign.values,
                    },
                    open(
                        f"atlasruns/bootleghartman/run_{model_kind}_{SURFACE_KIND}_29.pkl",
                        "wb",
                    ),
                )
    run_ix +=1
        


    # except:
    #     print(f"run failed for index {run_ix}")

    #     run_ix +=1
    #     continue
    

    
    


