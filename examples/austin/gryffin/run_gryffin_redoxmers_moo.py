#!/usr/bin/env python

import os
import pickle
import sys
import shutil
import numpy as np
import olympus
from olympus.campaigns import Campaign
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.evaluators import Evaluator
from olympus.planners import Gryffin, Planner
from olympus.scalarizers import Scalarizer

# --------
# CONFIG
# --------

dataset_name = "redoxmers"
use_descriptors = True
dynamic = True

budget = 10
num_repeats = 10
batch_size = 3
sampling_strategies = np.array([-1, 1, 0.0, 0.5])


# ------------------
# helper functions
# ------------------


def save_pkl_file(data_all_repeats):
    """save pickle file with results so far"""

    if os.path.isfile("results.pkl"):
        shutil.move(
            "results.pkl", "bkp-results.pkl"
        )  # overrides existing files

    # store run results to disk
    with open("results.pkl", "wb") as content:
        pickle.dump(data_all_repeats, content)


def load_data_from_pkl_and_continue(N):
    """load results from pickle file"""

    data_all_repeats = []
    # if no file, then we start from scratch/beginning
    if not os.path.isfile("results.pkl"):
        return data_all_repeats, N

    # else, we load previous results and continue
    with open("results.pkl", "rb") as content:
        data_all_repeats = pickle.load(content)

    missing_N = N - len(data_all_repeats)

    return data_all_repeats, missing_N


# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(
    num_repeats
)


for num_repeat in range(missing_repeats):

    print(f"\nTESTING Gryffin ON {dataset_name} REPEAT {num_repeat} ...\n")

    dataset = Dataset(kind=dataset_name)
    scalarizer = Scalarizer(
        kind="Hypervolume",
        value_space=dataset.value_space,
        goals=["min", "min", "min"],
    )
    planner = Gryffin(
        goal="minimize",
        use_descriptors=use_descriptors,
        auto_desc_gen=dynamic,
        batch_size=batch_size,
        sampling_strategies=sampling_strategies,
    )
    planner.set_param_space(dataset.param_space)

    campaign = Campaign()
    campaign.set_param_space(dataset.param_space)
    campaign.set_value_space(dataset.value_space)
    campaign.set_emulator_specs(dataset)

    evaluator = Evaluator(
        planner=planner,
        emulator=dataset,
        campaign=campaign,
        scalarizer=scalarizer,
    )

    evaluator.optimize(num_iter=budget)

    data_all_repeats.append(campaign)
    save_pkl_file(data_all_repeats)

    print("Done!")
