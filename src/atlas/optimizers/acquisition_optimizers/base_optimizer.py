#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import time
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from olympus.campaigns import ParameterSpace

from atlas import Logger
from atlas.optimizers.acqfs import VarianceBased
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
        acquisition_type: str,
        acqf: AcquisitionFunction,
        known_constraints: Callable,
        batch_size: int,
        feas_strategy: str,
        fca_constraint: Callable,
        params: torch.Tensor,
        timings_dict: Dict,

    ):
        self.kind = kind
        self.params_obj = params_obj
        self.acquisition_type = acquisition_type
        self.acqf = acqf
        self.known_constraints = known_constraints
        self.batch_size = batch_size
        self.feas_strategy = feas_strategy
        self.fca_constraint = fca_constraint
        self._params = params
        self.timings_dict = timings_dict

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

        start_time = time.time()
        # returns list of parameter vectors with recommendations
        results = self.optimizer.optimize()
        self.timings_dict['acquisition_opt'] = time.time()-start_time

        # if we have a general parameter optimization, we use a
        # variance-based sampling procedure to select the next general parameter(s)
        if self.acquisition_type == 'general':

            X_sns_empty, general_raw = self.acqf.generate_X_sns()

            functional_dims = np.logical_not(self.params_obj.exp_general_mask)

            # convert results to expanded tensor
            X_star = torch.tensor(
                self.params_obj.param_vectors_to_expanded(results, return_scaled=True)
            )
            # TODO: careful of batch size
            X_star = torch.unsqueeze(X_star,1)

            X_sns = torch.empty( (X_star.shape[0],) + X_sns_empty.shape )
            for x_ix in range(X_star.shape[0]):
                X_sn  = torch.clone(X_sns_empty)
                X_sn[:, :, functional_dims] = X_star[x_ix, :, functional_dims]
                X_sns[x_ix,:,:,:] = X_sn

            acqf_sn = VarianceBased(reg_model=self.acqf.reg_model)

            # generates list of stdevs, one set for each batch
            for ix, X_sn in enumerate(X_sns):
                sigma = acqf_sn(X_sn)
                select_gen_params = general_raw[ torch.argmax(sigma) ]

                for gen_param_ix in self.params_obj.general_dims:
                    results[ix][self.params_obj.param_space[gen_param_ix].name] = select_gen_params[gen_param_ix]

        return results
