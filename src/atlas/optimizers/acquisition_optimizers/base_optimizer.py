#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from botorch.acquisition import AcquisitionFunction
from olympus.campaigns import ParameterSpace

from atlas import Logger
from atlas.optimizers.acquisition_optimizers.genetic_optimizer import (
    GeneticOptimizer,
)
from atlas.optimizers.acquisition_optimizers.gradient_optimizer import (
    GradientOptimizer,
)
from atlas.optimizers.params import Parameters
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


class AcquisitionOptimizer:
    def __init__(
        self,
        kind: str,
        params_obj: Parameters,
        acqf: AcquisitionFunction,
        known_constraints: Callable,
        batch_size: int,
        feas_strategy: str,
        fca_constraint: Callable,
        params: torch.Tensor,
    ):
        self.kind = kind
        self.params_obj = params_obj
        self.acqf = acqf
        self.known_constraints = known_constraints
        self.batch_size = batch_size
        self.feas_strategy = feas_strategy
        self.fca_constraint = fca_constraint
        self._params = params

        # check kind of acquisition optimization
        if self.kind == "gradient":
            if self.known_constraints is not None:
                msg = 'Gradient acquisition optimizer does not current support known constraints, please use the Genetic optimizer'
                Logger.log(msg, 'FATAL')

            self.optimizer = GradientOptimizer(
                self.params_obj,
                self.acqf,
                self.known_constraints,
                self.batch_size,
                self.feas_strategy,
                self.fca_constraint,
                self._params,
            )

        elif self.kind == "genetic":
            self.optimizer = GeneticOptimizer(
                self.params_obj,
                self.acqf,
                self.known_constraints,
                self.batch_size,
                self.feas_strategy,
                self.fca_constraint,
                self._params,
            )

        else:
            msg = f"Acquisition optimizer kind {self.kind} not known"
            Logger.log(msg, "FATAL")

    def optimize(self):
        # returns list of parameter vectors with recommendations
        results = self.optimizer.optimize()

        return results
