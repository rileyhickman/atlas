#!/usr/bin/env python

import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)
from olympus.surfaces import Surface

from atlas.optimizers.gp.planner import BoTorchPlanner

CONT = {
    "init_design_strategy": [
        "random",
        "sobol",
        "lhs",
    ],  # init design strategues
    "batch_size": [1, 5],  # batch size
    "use_descriptors": [False],  # use descriptors
}

DISC = {
    "init_design_strategy": ["random"],
    "batch_size": [1, 5],
    "use_descriptors": [False],
}

CAT = {
    "init_design_strategy": ["random"],
    "batch_size": [1, 5],
    "use_descriptors": [False, True],
}

MIXED_CAT_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
}

MIXED_DISC_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False],
}


MIXED_CAT_DISC = {
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


@pytest.mark.parametrize("init_design_strategy", DISC["init_design_strategy"])
@pytest.mark.parametrize("batch_size", DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", DISC["use_descriptors"])
def test_init_design_disc(init_design_strategy, batch_size, use_descriptors):
    run_discrete(init_design_strategy, batch_size, use_descriptors)


@pytest.mark.parametrize("init_design_strategy", CAT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CAT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CAT["use_descriptors"])
def test_init_design_cat(init_design_strategy, batch_size, use_descriptors):
    run_categorical(init_design_strategy, batch_size, use_descriptors)


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_CONT["use_descriptors"])
def test_init_design_mixed_cat_cont(
    init_design_strategy, batch_size, use_descriptors
):
    run_mixed_cat_cont(init_design_strategy, batch_size, use_descriptors)


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_DISC_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_DISC_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_DISC_CONT["use_descriptors"])
def test_init_design_mixed_disc_cont(
    init_design_strategy, batch_size, use_descriptors
):
    run_mixed_disc_cont(init_design_strategy, batch_size, use_descriptors)


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_DISC["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_DISC["use_descriptors"])
def test_init_design_mixed_cat_disc(
    init_design_strategy, batch_size, use_descriptors
):
    run_mixed_cat_disc(init_design_strategy, batch_size, use_descriptors)


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_DISC_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC_CONT["batch_size"])
@pytest.mark.parametrize(
    "use_descriptors", MIXED_CAT_DISC_CONT["use_descriptors"]
)
def test_init_design_mixed_cat_disc_cont(
    init_design_strategy, batch_size, use_descriptors
):
    run_mixed_cat_disc_cont(init_design_strategy, batch_size, use_descriptors)


def run_continuous(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):
    def surface(x):
        return np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])

    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
    param_2 = ParameterContinuous(name="param_2", low=0.0, high=1.0)
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
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


def run_discrete(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):
    def surface(x):
        return np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])

    param_space = ParameterSpace()
    param_0 = ParameterDiscrete(
        name="param_0",
        options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    param_1 = ParameterDiscrete(
        name="param_1",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_2 = ParameterDiscrete(
        name="param_2",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=1,
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


def run_mixed_cat_cont(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):

    param_space = ParameterSpace()

    if use_descriptors:
        desc_dummy = [[float(i), float(i)] for i in range(3)]
        desc_cat_index = [[float(i), float(i)] for i in range(4)]
    else:
        desc_dummy = [None for i in range(3)]
        desc_cat_index = [None for i in range(4)]

    # add dummy param
    param_space.add(
        ParameterCategorical(
            name="dummy",
            options=[f"x{i}" for i in range(3)],
            descriptors=desc_dummy,
        )
    )

    # add ligand
    param_space.add(
        ParameterCategorical(
            name="cat_index",
            options=[str(i) for i in range(4)],
            descriptors=desc_cat_index,
        )
    )
    # add temperature
    param_space.add(
        ParameterContinuous(name="temperature", low=30.0, high=110.0)
    )
    # add residence time
    param_space.add(ParameterContinuous(name="t", low=60.0, high=600.0))
    # add catalyst loading
    # summit expects this to be in nM
    param_space.add(
        ParameterContinuous(
            name="conc_cat",
            low=0.835 / 1000,
            high=4.175 / 1000,
        )
    )

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal="maximize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    def mock_yield(x):
        return np.random.uniform() * 100

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = mock_yield(sample)
            # print(f'ITER : {iter}\tSAMPLES : {sample}\t MEASUREMENT : {measurement}')
            campaign.add_observation(sample_arr, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_disc_cont(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):
    def surface(x):
        return np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])

    param_space = ParameterSpace()
    param_0 = ParameterDiscrete(name="param_0", options=[0.0, 0.3, 0.4, 0.9])
    param_1 = ParameterDiscrete(name="param_1", options=[0.0, 0.5, 1.0])
    param_2 = ParameterContinuous(name="param_2", low=15.0, high=20.0)
    param_3 = ParameterContinuous(name="param_3", low=7.0, high=9.0)
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


def run_mixed_cat_disc(
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
        )

    if use_descriptors:
        desc_param_0 = [[float(i), float(i)] for i in range(3)]
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
    param_2 = ParameterDiscrete(
        name="param_2",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:

            measurement = surface(sample)
            campaign.add_observation(sample, measurement)

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
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:

            measurement = surface(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


if __name__ == "__main__":
    # pass
    # run_discrete('random')
    run_continuous("lhs", 1, False)
    # run_categorical_ohe('random')
    # run_categorical_desc('random')
    # run_mixed_cat_disc('random')
    # run_mixed_cat_disc_desc('random')
    # run_mixed_cat_cont('random')
    # run_mixed_cat_cont_desc('random')
    # run_mixed_disc_cont('random')
    # run_mixed_cat_disc_cont('random')
    # run_mixed_cat_disc_cont_desc('random')
