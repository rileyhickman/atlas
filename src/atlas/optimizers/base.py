#!/usr/bin/env python

import os
import pickle
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy as np
import olympus
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
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
from rich.progress import track

from atlas import Logger
from atlas.optimizers.acqfs import (
    FeasibilityAwareEI,
    FeasibilityAwareGeneral,
    FeasibilityAwareQEI,
    create_available_options,
    get_batch_initial_conditions,
)
from atlas.optimizers.params import Parameters
from atlas.optimizers.acquisition_optimizers.base_optimizer import (
    AcquisitionOptimizer,
)
from atlas.optimizers.gps import (
    CategoricalSingleTaskGP,
    ClassificationGPMatern,
)
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


class BasePlanner(CustomPlanner):
    def __init__(
        self,
        goal: str,
        feas_strategy: Optional[str] = "naive-0",
        feas_param: Optional[float] = 0.2,
        batch_size: int = 1,
        random_seed: Optional[int] = None,
        use_descriptors: bool = False,
        num_init_design: int = 5,
        init_design_strategy: str = "random",
        acquisition_type: str = 'ei',  # ei, ucb
        acquisition_optimizer_kind: str = "gradient",  # gradient, genetic
        vgp_iters: int = 2000,
        vgp_lr: float = 0.1,
        max_jitter: float = 1e-1,
        cla_threshold: float = 0.5,
        known_constraints: Optional[List[Callable]] = None,
        general_parameters: Optional[List[int]] = None,
        is_moo: bool = False,
        value_space: Optional[ParameterSpace] = None,
        scalarizer_kind: Optional[str] = "Hypervolume",
        moo_params: Dict[str, Union[str, float, int, bool, List]] = {},
        goals: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        AbstractPlanner.__init__(**locals())
        self.goal = goal
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.batch_size = batch_size
        if random_seed is None:
            self.random_seed = np.random.randint(0, int(10e6))
        else:
            self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.use_descriptors = use_descriptors
        self.num_init_design = num_init_design
        self.init_design_strategy = init_design_strategy
        self.acquisition_type = acquisition_type
        self.acquisition_optimizer_kind = acquisition_optimizer_kind
        self.vgp_iters = vgp_iters
        self.vgp_lr = vgp_lr
        self.max_jitter = max_jitter
        self.cla_threshold = cla_threshold
        self.known_constraints = known_constraints
        self.general_parameters = general_parmeters
        self.is_moo = is_moo
        self.value_space = value_space
        self.scalarizer_kind = scalarizer_kind
        self.moo_params = moo_params
        self.goals = goals

        # check multiobjective stuff
        if self.is_moo:
            if self.goals is None:
                message = f"You must individual goals for multiobjective optimization"
                Logger.log(message, "FATAL")

            if self.goal == "maximize":
                message = "Overall goal must be set to minimization for multiobjective optimization. Updating ..."
                Logger.log(message, "WARNING")
                self.goal = "minimize"

            self.scalarizer = Scalarizer(
                kind=self.scalarizer_kind,
                value_space=self.value_space,
                goals=self.goals,
                **self.moo_params,
            )

        # treat the inital design arguments
        if self.init_design_strategy == "random":
            self.init_design_planner = olympus.planners.RandomSearch(
                goal=self.goal
            )
        elif self.init_design_strategy == "sobol":
            self.init_design_planner = olympus.planners.Sobol(
                goal=self.goal, budget=self.num_init_design
            )
        elif self.init_design_strategy == "lhs":
            self.init_design_planner = olympus.planners.LatinHypercube(
                goal=self.goal, budget=self.num_init_design
            )
        else:
            message = f"Initial design strategy {self.init_design_strategy} not implemented"
            Logger.log(message, "FATAL")

        self.num_init_design_completed = 0



    def _set_param_space(self, param_space: ParameterSpace):
        """set the Olympus parameter space (not actually really needed)"""

        # infer the problem type
        self.problem_type = infer_problem_type(self.param_space)

        # make attribute that indicates wether or not we are using descriptors for
        # categorical variables
        if self.problem_type == "fully_categorical":
            descriptors = []
            for p in self.param_space:
                if not self.use_descriptors:
                    descriptors.extend([None for _ in range(len(p.options))])
                else:
                    descriptors.extend(p.descriptors)
            if all(d is None for d in descriptors):
                self.has_descriptors = False
            else:
                self.has_descriptors = True

        elif self.problem_type in ["mixed_cat_cont", "mixed_cat_dis"]:
            descriptors = []
            for p in self.param_space:
                if p.type == "categorical":
                    if not self.use_descriptors:
                        descriptors.extend(
                            [None for _ in range(len(p.options))]
                        )
                    else:
                        descriptors.extend(p.descriptors)
            if all(d is None for d in descriptors):
                self.has_descriptors = False
            else:
                self.has_descriptors = True

        else:
            self.has_descriptors = False

        # check general parameter types, if we have some
        if self.general_parameters is not None:
            if not all([self.param_space[gen_ix].type in ['discrete', 'categorical'] for ix in self.general_parmeters]):
                msg = 'Only discrete- and categorical-type general parameters are currently supported'
                Logger.log(msg, 'FATAL')


    def build_train_classification_gp(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> Tuple[
        gpytorch.models.ApproximateGP, gpytorch.likelihoods.BernoulliLikelihood
    ]:
        """build the GP classification model and likelihood
        and train the model
        """
        model = ClassificationGPMatern(train_x, train_y)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        model, likelihood = self.train_vgp(model, likelihood, train_x, train_y)

        return model, likelihood

    def train_vgp(
        self,
        model: gpytorch.models.ApproximateGP,
        likelihood: gpytorch.likelihoods.BernoulliLikelihood,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> Tuple[
        gpytorch.models.ApproximateGP, gpytorch.likelihoods.BernoulliLikelihood
    ]:

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.vgp_lr)

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

        # TODO: might be better to break into batches here...
        # NOTE: we could also do some sort of cross-validation here for early stopping
        start_time = time.time()
        with gpytorch.settings.cholesky_jitter(self.max_jitter):
            for iter_ in track(
                range(self.vgp_iters), description="Training variational GP..."
            ):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
        vgp_train_time = time.time() - start_time
        msg = f" Classification surrogate VGP trained in {round(vgp_train_time,3)} sec ({self.vgp_iters} epochs)\t Loss : {round(loss.item(), 3)} "
        Logger.log(msg, "INFO")

        return model, likelihood

    def build_train_data(self) -> Tuple[torch.Tensor, torch.tensor]:
        """build the training dataset at each iteration"""
        if self.is_moo:
            # parameters should be the same for each objective
            # nans should be in the same locations for each objective
            feas_ix = np.where(~np.isnan(self._values[:, 0]))[0]
            # generate the classification dataset
            params_cla = self._params.copy()
            values_cla = np.where(
                ~np.isnan(self._values[:, 0]), 0.0, self._values[:, 0]
            )
            train_y_cla = np.where(np.isnan(values_cla), 1.0, values_cla)
            # generate the regression dataset
            params_reg = self._params[feas_ix].reshape(-1, 1)
            train_y_reg = self._values[
                feas_ix, :
            ]  # (num_feas_observations, num_objectives)
            # scalarize the data
            train_y_reg = self.scalarizer.scalarize(train_y_reg).reshape(
                -1, 1
            )  # (num_feas_observations, 1)

        else:
            feas_ix = np.where(~np.isnan(self._values))[0]
            # generate the classification dataset
            params_cla = self._params.copy()
            values_cla = np.where(~np.isnan(self._values), 0.0, self._values)
            train_y_cla = np.where(np.isnan(values_cla), 1.0, values_cla)

            # generate the regression dataset
            params_reg = self._params[feas_ix].reshape(-1, 1)
            train_y_reg = self._values[feas_ix].reshape(-1, 1)


        train_x_cla, train_x_reg = [], []

        # adapt the data from olympus form to torch tensors
        for ix in range(self._values.shape[0]):
            sample_x = []
            for param_ix, (space_true, element) in enumerate(
                zip(self.param_space, params_cla[ix])
            ):
                if self.param_space[param_ix].type == "categorical":
                    feat = cat_param_to_feat(
                        space_true,
                        element,
                        has_descriptors=self.has_descriptors,
                    )
                    sample_x.extend(feat)
                else:
                    sample_x.append(float(element))
            train_x_cla.append(sample_x)
            if ix in feas_ix:
                train_x_reg.append(sample_x)

        train_x_cla, train_x_reg = np.array(train_x_cla), np.array(train_x_reg)

        # scale the training data - normalize inputs and standardize outputs

        self._means_y, self._stds_y = np.mean(train_y_reg, axis=0), np.std(
            train_y_reg, axis=0
        )
        self._stds_y = np.where(self._stds_y == 0.0, 1.0, self._stds_y)

        if (
            self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            train_x_cla = forward_normalize(
                train_x_cla, self.params_obj._mins_x, self.params_obj._maxs_x
            )
            train_x_reg = forward_normalize(
                train_x_reg, self.params_obj._mins_x, self.params_obj._maxs_x
            )

        # always forward transform the objectives for the regression problem
        train_y_reg = forward_standardize(
            train_y_reg, self._means_y, self._stds_y
        )

        # convert to torch tensors and return
        return (
            torch.tensor(train_x_cla).float(),
            torch.tensor(train_y_cla).squeeze().float(),
            torch.tensor(train_x_reg).double(),
            torch.tensor(train_y_reg).double(),
        )

    def reg_surrogate(
        self,
        X: torch.Tensor,
        return_np: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]
    ]:
        """make prediction using regression surrogate model

        Args:
                X (np.ndarray or list): 2d numpy array or nested list with input parameters
        """

        if not hasattr(self, "reg_model"):
            msg = "Optimizer does not yet have regression surrogate model"
            Logger.log(msg, "FATAL")

        X_proc = []
        # adapt the data from olympus form to torch tensors
        for ix in range(len(X)):
            sample_x = []
            for param_ix, (space_true, element) in enumerate(
                zip(self.param_space, X[ix])
            ):
                if self.param_space[param_ix].type == "categorical":
                    feat = cat_param_to_feat(
                        space_true,
                        element,
                        has_descriptors=self.has_descriptors,
                    )
                    sample_x.extend(feat)
                else:
                    sample_x.append(float(element))
            X_proc.append(sample_x)

        X_proc = torch.tensor(np.array(X_proc)).double()

        if (
            self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            X_proc = forward_normalize(X_proc, self.params_obj._mins_x, self.params_obj._maxs_x)

        posterior = self.reg_model.posterior(X=X_proc)
        pred_mu, pred_sigma = posterior.mean.detach(), torch.sqrt(
            posterior.variance.detach()
        )

        # reverse scale the predictions
        pred_mu = reverse_standardize(pred_mu, self._means_y, self._stds_y)

        if self.goal == "maximize":
            pred_mu = -pred_mu

        if return_np:
            pred_mu, pred_sigma = pred_mu.numpy(), pred_sigma.numpy()

        return pred_mu, pred_sigma

    def cla_surrogate(
        self,
        X: torch.Tensor,
        return_np: bool = False,
        normalize: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:

        if not hasattr(self, "cla_model"):
            msg = "Optimizer does not yet have classification surrogate model"
            Logger.log(msg, "FATAL")

        X_proc = []
        # adapt the data from olympus form to torch tensors
        for ix in range(len(X)):
            sample_x = []
            for param_ix, (space_true, element) in enumerate(
                zip(self.param_space, X[ix])
            ):
                if self.param_space[param_ix].type == "categorical":
                    feat = cat_param_to_feat(
                        space_true,
                        element,
                        has_descriptors=self.has_descriptors,
                    )
                    sample_x.extend(feat)
                else:
                    sample_x.append(float(element))
            X_proc.append(sample_x)

        X_proc = torch.tensor(np.array(X_proc)).double()

        if (
            self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            X_proc = forward_normalize(X_proc, self.params_obj._mins_x, self.params_obj._maxs_x)

        likelihood = self.cla_likelihood(self.cla_model(X_proc.float()))
        mean = likelihood.mean.detach()
        mean = mean.view(mean.shape[0], 1)
        # mean = 1.-mean.view(mean.shape[0],1) # switch from p_feas to p_infeas
        if normalize:
            _max = torch.amax(mean, axis=0)
            _min = torch.amin(mean, axis=0)
            mean = (mean - _min) / (_max - _min)

        if return_np:
            mean = mean.numpy()

        return mean

    def acquisition_function(
        self,
        X: torch.Tensor,
        return_np: bool = True,
        normalize: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:

        X_proc = []
        # adapt the data from olympus form to torch tensors
        for ix in range(len(X)):
            sample_x = []
            for param_ix, (space_true, element) in enumerate(
                zip(self.param_space, X[ix])
            ):
                if self.param_space[param_ix].type == "categorical":
                    feat = cat_param_to_feat(
                        space_true,
                        element,
                        has_descriptors=self.has_descriptors,
                    )
                    sample_x.extend(feat)
                else:
                    sample_x.append(float(element))
            X_proc.append(sample_x)

        X_proc = torch.tensor(np.array(X_proc)).double()

        if (
            self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            X_proc = forward_normalize(X_proc, self.params_obj._mins_x, self.params_obj._maxs_x)

        acqf_vals = self.acqf(
            X_proc.view(X_proc.shape[0], 1, X_proc.shape[-1])
        ).detach()

        acqf_vals = acqf_vals.view(acqf_vals.shape[0], 1)

        if normalize:
            _max = torch.amax(acqf_vals, axis=0)
            _min = torch.amin(acqf_vals, axis=0)
            acqf_vals = (acqf_vals - _min) / (_max - _min)

        if return_np:
            acqf_vals = acqf_vals.numpy()

        return acqf_vals

    def _tell(self, observations: olympus.campaigns.observations.Observations):
        """unpack the current observations from Olympus
        Args:
                observations (obj): Olympus campaign observations object
        """

        # elif type(observations) == olympus.campaigns.observations.Observations:
        self._params = observations.get_params(
            as_array=True
        )  # string encodings of categorical params
        self._values = observations.get_values(
            as_array=True, opposite=self.flip_measurements
        )

        # make values 2d if they are not already
        if len(np.array(self._values).shape) == 1:
            self._values = np.array(self._values).reshape(-1, 1)

        # generate Parameters object
        self.params_obj = Parameters(
            olympus_param_space=self.param_space,
            observations=observations,
            has_descriptors=self.has_descriptors,
            general_parmeters=self.general_parameters,
        )

    def fca_constraint(self, X: torch.Tensor) -> torch.Tensor:
        """Each callable is expected to take a `(num_restarts) x q x d`-dim tensor as an
                input and return a `(num_restarts) x q`-dim tensor with the constraint
                values. The constraints will later be passed to SLSQP. You need to pass in
                `batch_initial_conditions` in this case. Using non-linear inequality
                constraints also requires that `batch_limit` is set to 1, which will be
                done automatically if not specified in `options`.
                >= 0 is a feasible point
                <  0 is an infeasible point
        Args:
                X (torch.tensor):
        """
        # handle the various potential input tensor sizes (this function can be called from
        # several places, including inside botorch)
        # TODO: this is pretty messy, consider cleaning up
        if len(X.size()) == 3:
            X = X.squeeze(1)
        if len(X.size()) == 1:
            X = X.view(1, X.shape[0])
        # squeeze the middle q dimension
        # this expression is >= 0 for a feasible point, < 0 for an infeasible point
        # p_feas should be 1 - P(infeasible|X) which is returned by the classifier
        with gpytorch.settings.cholesky_jitter(1e-1):
            p_infeas = (
                self.cla_likelihood(self.cla_model(X.float()))
                .mean.unsqueeze(-1)
                .double()
            )
            if self.problem_type in ["fully_categorical", "fully_discrete"]:
                _max = torch.amax(p_infeas)
                _min = torch.amin(p_infeas)
                if not torch.abs(_max - _min) > 1e-6:
                    _max = 1.0
                    _min = 0.0
                p_infeas = (p_infeas - _min) / (_max - _min)
            constraint_val = (1.0 - p_infeas) - self.feas_param
            # constraint_val = (1. - self.cla_likelihood(self.cla_model(X.float())).mean.unsqueeze(-1).double()) - self.feas_param

        return constraint_val
