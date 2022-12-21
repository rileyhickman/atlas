#!/usr/bin/env python

import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterVector,
)
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface

from atlas.optimizers.gp.planner import BoTorchPlanner


CONT = {
    "init_design_strategy": [
        "random",
    ],  # init design strategues
    "batch_size": [1],  # batch size
    "use_descriptors": [False],  # use descriptors
}

CAT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
}


MIXED_CAT_DISC_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
}


@pytest.mark.parametrize("init_design_strategy", CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CONT["use_descriptors"])
def test_init_design_cont(init_design_strategy, batch_size, use_descriptors):
    run_continuous(init_design_strategy, batch_size, use_descriptors)


def known_constraints_cont(params):
    # for 3d unit hypercube param space
    if params['param_0'] > 0.8 or params['param_2'] > 0.9:
        return False
    if params['param_2'] < 0.4 or params['param_1'] < 0.3:
        return False
    return True


def run_continuous(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):
    def surface(x):
        if np.random.uniform()<0.5:
            return np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])
        else:
            return np.nan

    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
    param_2 = ParameterContinuous(name="param_2", low=0.0, high=1.0)
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="fwa",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_type='ei',
        acquisition_optimizer_kind='genetic',
        known_constraints=[known_constraints_cont],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface(sample_arr)
            campaign.add_observation(sample_arr, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_categorical(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):

    surface_kind = "CatDejong"
    surface = Surface(kind=surface_kind, param_dim=2, num_opts=21)

    campaign = Campaign()
    campaign.set_param_space(surface.param_space)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_optimizer_kind='genetic',
    )
    planner.set_param_space(surface.param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = np.array(surface.run(sample_arr))
            # print(sample, measurement)
            campaign.add_observation(sample_arr, measurement[0])

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_disc_cont(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):
    def surface(x):
        if x["param_0"] == "x0":
            factor = 0.1
        elif x["param_0"] == "x1":
            factor = 1.0
        elif x["param_0"] == "x2":
            factor = 10.0

        return (
            np.sin(8.0 * x["param_1"])
            - 2.0 * np.cos(6.0 * x["param_1"])
            + np.exp(-2.0 * x["param_2"])
            + 2.0 * (1.0 / factor)
            + x["param_3"]
        )

    if use_descriptors:
        desc_param_0 = [[float(i), float(i)] for i in range(3)]
        print('here!')
    else:
        desc_param_0 = [None for i in range(3)]

    param_space = ParameterSpace()
    param_0 = ParameterCategorical(
        name="param_0",
        options=["x0", "x1", "x2"],
        descriptors=desc_param_0,
    )
    param_1 = ParameterDiscrete(
        name="param_1",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_2 = ParameterContinuous(
        name="param_2",
        low=0.0,
        high=1.0,
    )
    param_3 = ParameterContinuous(
        name="param_3",
        low=0.0,
        high=1.0,
    )
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)
    param_space.add(param_3)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_optimizer_kind='genetic',
        # acquisition_type='general',
        # general_parameters=[0],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 5

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:

            measurement = surface(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET



if  __name__ == '__main__':

    #run_continuous('random',1,False )
    run_categorical('random',1,False )
    #run_mixed_cat_disc_cont('random',1,False)
