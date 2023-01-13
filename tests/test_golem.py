#!/usr/bin/env python

import numpy as np
import pytest
from olympus.objects import (
    Campaign,
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterSpace,
    ParameterVector,
)

from atlas.optimizers.gp.planner import BoTorchPlanner


class Bertsimas:
    def __init__(self):

        self.xlims = [-1, 3.2]
        self.ylims = [-0.5, 4.4]
        self.minimum = [2.8, 4.0]

    @staticmethod
    def eval(params):
        x0 = params["x0"]
        x1 = params["x1"]
        f = (
            2 * (x0**6)
            - 12.2 * (x0**5)
            + 21.2 * (x0**4)
            + 6.2 * x0
            - 6.4 * (x0**3)
            - 4.7 * (x0**2)
            + x1**6
            - 11 * (x1**5)
            + 43.3 * (x1**4)
            - 10 * x1
            - 74.8 * (x1**3)
            + 56.9 * (x1**2)
            - 4.1 * x0 * x1
            - 0.1 * (x1**2) * (x0**2)
            + 0.4 * (x1**2) * x0
            + 0.4 * (x0**2) * x1
        )
        return f


def run_continuous(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):

    surf = Bertsimas()

    return None
