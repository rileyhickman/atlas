#!/usr/bin/env python

import itertools
from copy import deepcopy

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction, ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)

from atlas import Logger
from atlas.optimizers.utils import (
    cat_param_to_feat,
    forward_normalize,
    propose_randomly,
)

class FeasibilityAwareAcquisition:

    def compute_feas_post(self, X):
        """computes the posterior P(infeasible|X)
        Args:
                X (torch.tensor): input tensor with shape (num_samples, q_batch_size, num_dims)
        """
        with gpytorch.settings.cholesky_jitter(1e-1):
            return self.cla_likelihood(
                self.cla_model(X.float().squeeze(1))
            ).mean

    def compute_combined_acqf(self, acqf, p_feas):
        """compute the combined acqusition function"""
        if self.feas_strategy == "fwa":
            return acqf * p_feas
        elif self.feas_strategy == "fca":
            return acqf
        elif self.feas_strategy == "fia":
            return ((1.0 - self.infeas_ratio**self.feas_param) * acqf) + (
                (self.infeas_ratio**self.feas_param) * (p_feas)
            )
        elif "naive-" in self.feas_strategy:
            if self.use_p_feas_only:
                return p_feas
            else:
                return acqf
        else:
            raise NotImplementedError


class VarianceBased(AcquisitionFunction):
    """ Variance-based sampling (active learning)
    """
    def __init__(self, reg_model, **kwargs):
        super().__init__(reg_model, **kwargs)
        self.reg_model = reg_model

    def forward(self, X):
        posterior = self.reg_model.posterior(X.double())
        sigma = posterior.variance.clamp_min(1e-9).sqrt()
        view_shape = sigma.shape[:-2] if sigma.shape[-2] == 1 else sigma.shape[:-1]

        return sigma.view(view_shape)


class FeasibilityAwareVarainceBased(VarianceBased, FeasibilityAwareAcquisition):
    """ Feasibility aware variance-based sampling (active learning)
    """

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(reg_model, **kwargs)
        self.best_f = best_f
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.maximize = maximize


    def forward(self, X):
        acqf = super().forward(X)
        acqf = (acqf - self.acqf_min_max[0]) / (
            self.acqf_min_max[1] - self.acqf_min_max[0] # normalize sigma value
        )

        if not "naive-" in self.feas_strategy:
            p_feas = 1.0 - self.compute_feas_post(X)
        else:
            p_feas = 1.0

        return self.compute_combined_acqf(acqf, p_feas)



class FeasibilityAwareGeneral(AcquisitionFunction, FeasibilityAwareAcquisition):
    """Abstract feasibilty aware general purpose optimization acquisition function.
    """

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        params_obj,
        # general_parameters,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(reg_model, **kwargs)
        self.best_f = best_f
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.params_obj = params_obj
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.maximize = maximize

        self.X_sns_empty, _ = self.generate_X_sns()
        self.functional_dims = np.logical_not(self.params_obj.exp_general_mask)


    def forward(self, X):
        X = X.double()
        best_f = self.best_f.to(X)

        X_sns = torch.empty( (X.shape[0],)+self.X_sns_empty.shape ).double()
        for x_ix in range(X.shape[0]):
            X_sn  = torch.clone(self.X_sns_empty)
            X_sn[:, :, self.functional_dims] = X[x_ix, :, self.functional_dims]
            X_sns[x_ix,:,:,:] = X_sn

        pred_mu_x, pred_sigma_x = [], []

        for X_sn in X_sns:
            posterior = self.reg_model.posterior(X_sn.double())
            mu = posterior.mean
            view_shape = mu.shape[:-2] if mu.shape[-2] == 1 else mu.shape[:-1]
            mu = mu.view(view_shape)
            sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
            pred_mu_x.append(mu)
            pred_sigma_x.append(sigma)

        pred_mu_x = torch.stack(pred_mu_x)
        pred_sigma_x = torch.stack(pred_sigma_x)

        mu_x = torch.mean(pred_mu_x, 0)
        sigma_x = torch.mean(pred_sigma_x, 0)

        u = (mu_x - best_f.expand_as(mu_x)) / sigma_x
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)

        return ei


    def generate_X_sns(self):
        # generate Cartesian product space of the general parameter options
        param_options = []
        for ix in self.params_obj.general_dims:
            param_options.append(self.param_space[ix].options)

        cart_product = list(itertools.product(*param_options))
        cart_product = [list(elem) for elem in cart_product]

        X_sns_empty = torch.empty(size=(len(cart_product), self.params_obj.expanded_dims)).double()
        general_expanded = []
        general_raw  = []
        for elem in cart_product:
            # convert to ohe and add to currently available options
            ohe,raw = [],[]
            for val, obj in zip(elem, self.param_space):
                if obj.type == "categorical":
                    ohe.append(
                        cat_param_to_feat(obj, val, self.params_obj.has_descriptors)
                    )
                    raw.append(val)
                else:
                    ohe.append([val])
            general_expanded.append(np.concatenate(ohe))
            general_raw.append(raw)

        general_expanded = torch.tensor(np.array(general_expanded))

        X_sns_empty[:, self.params_obj.exp_general_mask] = general_expanded
        # forward normalize
        X_sns_empty = forward_normalize(
            X_sns_empty, self.params_obj._mins_x, self.params_obj._maxs_x,
        )
        # TODO: careful of the batch size, will need to change this
        X_sns_empty = torch.unsqueeze(X_sns_empty,1)

        return X_sns_empty, general_raw


class FeasibilityAwareQEI(qExpectedImprovement, FeasibilityAwareAcquisition):
    """Abstract feasibility aware expected improvement acquisition function. Compatible
    with the FIA, FCA and FWA strategies, as well as any of the naive strategies.
    Args:
            reg_model (gpytorch.models.ExactGP): gpytorch regression surrogate GP model
            cla_model (gpytorch.models.ApproximateGP): gpytorch variational GP for fesibility surrogate
            cla_likelihood (gpytorch.likelihoods.BernoulliLikelihood): gpytorch Bernoulli likelihood
            best_f (torch.tensor): incumbent point
            feas_strategy (str): feasibility acqusition function name
            feas_param (float): feasibilty parameter (called "t" in the paper)
            infeas_ratio (float): the quotient of number of infeasible points with total points
            objective ():
            maximize (bool): whether the problem is maximization
    """

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(
            reg_model, best_f, objective=objective, maximize=maximize, **kwargs
        )
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only

    def forward(self, X):
        acqf = super().forward(X)  # get the EI acquisition
        # approximately normalize the EI acquisition function
        acqf = (acqf - self.acqf_min_max[0]) / (
            self.acqf_min_max[1] - self.acqf_min_max[0]
        )
        # p_feas should be 1 - P(feasible|X) because EI is
        # maximized by default
        if not "naive-" in self.feas_strategy:
            p_feas = 1.0 - self.compute_feas_post(X)
        else:
            p_feas = 1.0

        return self.compute_combined_acqf(acqf, p_feas)



class FeasibilityAwareEI(ExpectedImprovement, FeasibilityAwareAcquisition):
    """Abstract feasibility aware expected improvement acquisition function. Compatible
    with the FIA, FCA and FWA strategies, as well as any of the naive strategies.
    Args:
            reg_model (gpytorch.models.ExactGP): gpytorch regression surrogate GP model
            cla_model (gpytorch.models.ApproximateGP): gpytorch variational GP for fesibility surrogate
            cla_likelihood (gpytorch.likelihoods.BernoulliLikelihood): gpytorch Bernoulli likelihood
            best_f (torch.tensor): incumbent point
            feas_strategy (str): feasibility acqusition function name
            feas_param (float): feasibilty parameter (called "t" in the paper)
            infeas_ratio (float): the quotient of number of infeasible points with total points
            objective ():
            maximize (bool): whether the problem is maximization
    """

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(reg_model, best_f, objective, maximize, **kwargs)
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only

    def forward(self, X):

        acqf = super().forward(X)  # get the EI acquisition
        # approximately normalize the EI acquisition function
        acqf = (acqf - self.acqf_min_max[0]) / (
            self.acqf_min_max[1] - self.acqf_min_max[0]
        )
        # p_feas should be 1 - P(infeasible|X) because EI is
        # maximized by default
        if not "naive-" in self.feas_strategy:
            p_feas = 1.0 - self.compute_feas_post(X)
        else:
            p_feas = 1.0

        return self.compute_combined_acqf(acqf, p_feas)


class FeasibilityAwareUCB(UpperConfidenceBound, FeasibilityAwareAcquisition):

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        beta=torch.tensor([0.2]),
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(reg_model, beta, maximize=maximize, **kwargs)
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.best_f = best_f
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.beta = beta
        self.objective = objective
        self.maximize = maximize

    def forward(self, X):

        acqf = super().forward(X)  # get the EI acquisition
        # approximately normalize the EI acquisition function
        acqf = (acqf - self.acqf_min_max[0]) / (
            self.acqf_min_max[1] - self.acqf_min_max[0]
        )
        # p_feas should be 1 - P(infeasible|X) because EI is
        # maximized by default
        if not "naive-" in self.feas_strategy:
            p_feas = 1.0 - self.compute_feas_post(X)
        else:
            p_feas = 1.0

        return self.compute_combined_acqf(acqf, p_feas)

    def compute_feas_post(self, X):
        """computes the posterior P(infeasible|X)
        Args:
                X (torch.tensor): input tensor with shape (num_samples, q_batch_size, num_dims)
        """
        with gpytorch.settings.cholesky_jitter(1e-1):
            return self.cla_likelihood(
                self.cla_model(X.float().squeeze(1))
            ).mean

    def compute_combined_acqf(self, acqf, p_feas):
        """compute the combined acqusition function"""
        if self.feas_strategy == "fwa":
            return acqf * p_feas
        elif self.feas_strategy == "fca":
            return acqf
        elif self.feas_strategy == "fia":
            return ((1.0 - self.infeas_ratio**self.feas_param) * acqf) + (
                (self.infeas_ratio**self.feas_param) * (p_feas)
            )
        elif "naive-" in self.feas_strategy:
            if self.use_p_feas_only:
                return p_feas
            else:
                return acqf
        else:
            raise NotImplementedError



def get_batch_initial_conditions(
    num_restarts,
    batch_size,
    param_space,
    constraint_callable,
    mins_x,
    maxs_x,
    has_descriptors,
    num_chances=15,
    return_raw=False,
    ):
    """generate batches of initial conditions for a
    random restart optimization subject to some constraints. This uses
    rejection sampling, and might get very inefficient for parameter spaces with
    a large infeasible fraction.
    Args:
            num_restarts (int): number of optimization restarts
            batch_size (int): number of samples to recommend per ask/tell call (fixed to 1)
            param_space (obj): Olympus parameter space object for the given problem
            constraint_callable (list): list of callables which specifies the constraint function
            mins_x (np.array): minimum values of each parameter space dimension
            maxs_x (np.array): maximum values of each parameter
            num_chances (int):
    Returns:
            a torch.tensor with shape (num_restarts, batch_size, num_dims)
            of initial optimization conditions
    """
    # take 15*num_restarts points randomly and evaluate the constraint function on all of
    # them, if we have enough, proceed, if not proceed to sequential rejection sampling
    num_raw_samples = 10 * num_restarts
    raw_samples, raw_proposals = propose_randomly(
        num_raw_samples, param_space, has_descriptors
    )
    # forward normalize the randomly generated samples
    raw_samples = forward_normalize(raw_samples, mins_x, maxs_x)

    raw_samples = torch.tensor(raw_samples).view(
        raw_samples.shape[0], batch_size, raw_samples.shape[1]
    )

    if constraint_callable == []:
        # no constraints
        batch_initial_conditions = raw_samples
        batch_initial_conditions_raw = raw_proposals
    else:
        constraint_vals = []
        for constraint in constraint_callable:
            constraint_val = constraint(raw_samples)
            if len(constraint_val.shape) == 1:
                constraint_val = constraint_val.view(constraint_val.shape[0], 1)
            constraint_vals.append(constraint_val)

        if len(constraint_vals) == 2:
            constraint_vals = torch.cat(constraint_vals, dim=1)
            feas_ix = torch.where(torch.all(constraint_vals >= 0, dim=1))[0]
        elif len(constraint_vals) == 1:
            constraint_vals = constraint_vals[0]
            feas_ix = torch.where(constraint_vals >= 0)[0]

        batch_initial_conditions = raw_samples[feas_ix, :, :]
        batch_initial_conditions_raw = raw_proposals[feas_ix, :]

    if batch_initial_conditions.shape[0] >= num_restarts:
        if return_raw:
            return (
                batch_initial_conditions[:num_restarts,:,:],
                batch_initial_conditions_raw[:num_restarts,:]
            )
        return batch_initial_conditions[:num_restarts, :, :]
    elif 0 < batch_initial_conditions.shape[0] < num_restarts:
        print(
            f">>> insufficient samples, resorting to local sampling for {num_chances} iterations..."
        )
        for chance in range(num_chances):
            batch_initial_conditions = sample_around_x(
                batch_initial_conditions, constraint_callable
            )
            print(
                f"chance {chance}\t num feas {batch_initial_conditions.shape[0]}"
            )
            if batch_initial_conditions.shape[0] >= num_restarts:
                return batch_initial_conditions[:num_restarts, :, :]
        return None
    else:
        print(
            ">>> insufficient samples, resorting to unconstrained acquisition"
        )
        return None

    assert len(batch_initial_conditions.size()) == 3

    if return_raw:
        return batch_initial_conditions, batch_initial_conditions_raw
    return batch_initial_conditions



def sample_around_x(raw_samples, constraint_callable):
    """draw samples around points which we already know are feasible by adding
    some Gaussian noise to them
    """
    tiled_raw_samples = raw_samples.tile((10, 1, 1))
    means = deepcopy(tiled_raw_samples)
    stds = torch.ones_like(means) * 0.1
    perturb_samples = tiled_raw_samples + torch.normal(means, stds)
    # # project the values
    # perturb_samples = torch.where(perturb_samples>1., 1., perturb_samples)
    # perturb_samples = torch.where(perturb_samples<0., 0., perturb_samples)
    inputs = torch.cat((raw_samples, perturb_samples))

    constraint_vals = []
    for constraint in constraint_callable:
        constraint_val = constraint(inputs)
        if len(constraint_val.shape) == 1:
            constraint_val = constraint_val.view(constraint_val.shape[0], 1)
        constraint_vals.append(constraint_val)

    if len(constraint_vals) == 2:
        constraint_vals = torch.cat(constraint_vals, dim=1)
        feas_ix = torch.where(torch.all(constraint_vals >= 0))[0]
    elif len(constraint_vals) == 1:
        constraint_vals = torch.tensor(constraint_vals)
        feas_ix = torch.where(constraint_vals >= 0)[0]

    batch_initial_conditions = inputs[feas_ix, :, :]

    return batch_initial_conditions


def create_available_options(
    param_space,
    params,
    constraint_callable,
    normalize,
    mins_x,
    maxs_x,
    has_descriptors,
):
    """build cartesian product space of options, then remove options
    which have already been measured. Returns an (num_options, num_dims)
    torch tensor with all possible options

    If the parameter space is mixed, build and return the Cartesian product space of
    only the categorical and discrete parameters.

    Args:
            param_space (obj): Olympus parameter space object
            params (list): parameters from the current Campaign
            constraint_callable (callable):
            mins_x (np.array): minimum values of each parameter space dimension
            maxs_x (np.array): maximum values of each parameter
    """
    # make sure params are proper data type
    real_params = []
    for param in params:
        p = []
        for ix, elem in enumerate(param):
            if param_space[ix].type == "categorical":
                p.append(elem)
            else:
                p.append(float(elem))
        real_params.append(p)

    # params = [list(elem) for elem in params]
    # get the relevant parmeters -> only categorical and discrete
    relevant_params = [
        p for p in param_space if p.type in ["categorical", "discrete"]
    ]
    param_names = [p.name for p in relevant_params]
    param_options = [p.options for p in relevant_params]

    cart_product = list(itertools.product(*param_options))
    cart_product = [list(elem) for elem in cart_product]

    if len(param_names) == len(param_space):
        # no continuous parameters
        # remove options that we have measured already
        current_avail_feat = []
        current_avail_cat = []

        for elem in cart_product:
            if elem not in real_params:
                # convert to ohe and add to currently available options
                ohe = []
                for val, obj in zip(elem, param_space):
                    if obj.type == "categorical":
                        ohe.append(
                            cat_param_to_feat(obj, val, has_descriptors)
                        )
                    else:
                        ohe.append([val])
                current_avail_feat.append(np.concatenate(ohe))
                current_avail_cat.append(elem)

        current_avail_feat_unconst = np.array(current_avail_feat)
        current_avail_cat_unconst = np.array(current_avail_cat)

        # forward normalize the options before evaluating the c
        if normalize:
            current_avail_feat_unconst = forward_normalize(
                current_avail_feat_unconst, mins_x, maxs_x
            )

        current_avail_feat_unconst = torch.tensor(current_avail_feat_unconst)

        # remove options which are infeasible given the feasibility surrogate model
        # and the threshold
        if constraint_callable is not None:
            # FCA approach, apply feasibility constraint
            constraint_input = current_avail_feat_unconst.view(
                current_avail_feat_unconst.shape[0],
                1,
                current_avail_feat_unconst.shape[1],
            )
            constraint_vals = constraint_callable(constraint_input)
            feas_mask = torch.where(constraint_vals >= 0.0)[0]
            print(
                f"{feas_mask.shape[0]}/{current_avail_feat_unconst.shape[0]} options are feasible"
            )
            if feas_mask.shape[0] == 0:
                msg = "No feasible samples after FCA constraint, resorting back to full space"
                Logger.log(msg, "WARNING")
                # if we have zero feasible samples
                # resort back to the full set of unobserved options
                current_avail_feat = current_avail_feat_unconst
                current_avail_cat = current_avail_cat_unconst
            else:
                current_avail_feat = current_avail_feat_unconst[feas_mask]
                current_avail_cat = current_avail_cat_unconst[
                    feas_mask.detach().numpy()
                ]

        else:
            current_avail_feat = current_avail_feat_unconst
            current_avail_cat = current_avail_cat_unconst

        return current_avail_feat, current_avail_cat

    else:
        # at least one continuous parameter, no need to remove any options
        current_avail_feat = []
        current_avail_cat = []

        for elem in cart_product:
            # convert to ohe and add to currently available options
            ohe = []
            for val, obj in zip(elem, param_space):
                if obj.type == "categorical":
                    ohe.append(cat_param_to_feat(obj, val, has_descriptors))
                else:
                    ohe.append([val])
            current_avail_feat.append(np.concatenate(ohe))
            current_avail_cat.append(elem)

        current_avail_feat_unconst = np.array(current_avail_feat)
        current_avail_cat_unconst = np.array(current_avail_cat)

        # TODO: may need to add the constraint checking here too...
        # forward normalize the options before evaluating the constaints
        # if normalize:
        # 	current_avail_feat_unconst = forward_normalize(current_avail_feat_unconst, mins_x, maxs_x)

        current_avail_feat_unconst = torch.tensor(current_avail_feat_unconst)

        return current_avail_feat_unconst, current_avail_cat_unconst
