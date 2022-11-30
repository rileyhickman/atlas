#!/usr/bin/env python

import pickle

import numpy as np
import pandas as pd
from olympus.campaigns import Campaign, ParameterSpace
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)
from olympus.planners import Grid
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface

budget = 243
num_repeats = 50
levels = 5

data_all_repeats = []

for num_repeat in range(num_repeats):

    dataset = Dataset("lnp3")

    # load emulator
    emulator = Emulator(dataset="lnp3", model="BayesNeuralNet")

    campaign = Campaign()
    campaign.set_param_space(dataset.param_space)
    campaign.set_value_space(dataset.value_space)

    planner = Grid(
        goal="minimize", levels=levels, shuffle=True, exceed_budget=True
    )
    planner.set_param_space(dataset.param_space)

    scalarizer = Scalarizer(
        kind="Hypervolume",
        value_space=dataset.value_space,
        goals=["max", "max", "min"],
    )

    for num_iter in range(budget):

        print(f"repeat {num_repeat}\titer {num_iter+1}")

        samples = planner.recommend(campaign.observations)

        for sample in samples:
            sample_arr = sample.to_list()

            print("sample arr :", sample_arr)
            measurement = emulator.run(sample_arr, return_paramvector=True)

            campaign.add_and_scalarize(
                sample.to_array(), measurement, scalarizer
            )

            print("SAMPLE : ", sample)
            print("MEASUREMENT : ", measurement)

    # store the results into a DataFrame
    param_cols = {}
    for param_ix in range(len(dataset.param_space)):
        param_cols[
            dataset.param_space[param_ix].name
        ] = campaign.observations.get_params()[:, param_ix]

    value_cols = {}
    for value_ix in range(len(dataset.value_space)):
        value_cols[
            dataset.value_space[value_ix].name
        ] = campaign.observations.get_values()[:, value_ix]

    cols = {**param_cols, **value_cols}

    data = pd.DataFrame(cols)
    data_all_repeats.append(data)

    pickle.dump(data_all_repeats, open("results.pkl", "wb"))
