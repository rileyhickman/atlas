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
    UpperConfidenceBound,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel
from gpytorch.priors import MultivariateNormalPrior

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
from atlas.optimizers.base import BasePlanner

from atlas import Logger
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


class DynamicSSPlanner(BasePlanner):
    """Wrapper for GP-based Bayesiam optimization with BoTorch
    Args:
            goal (str): the optimization goal, "maximize" or "minimize"
            feas_strategy (str): feasibility acqusition function name
            feas_param (float): feasibilty parameter
            batch_size (int): number of samples to measure per batch (will be fixed at 1 for now)
            random_seed (int): the random seed to use
            num_initial_design (int): number of points to sample using the initial
                    design strategy
            init_design_strategy (str): the inital design strategy, "random" or "sobol"
            vgp_iters (int): number of training iterations for the variational GP
            vgp_lr (float): learning rate for the variational optimization procedure
            max_jitter (float):
            cla_threshold (float): classification threshold for the predictions of the
                    feasibilty surrogate
            known_constraints (callable): callable which takes parameters and returns boolean
                    corresponding to the feaibility of that experiment (True-->feasible, False-->infeasible)
            general_parameters (list): list of parameter indices for which we average the objective
                    function over
            is_moo (bool): whether or not we have a multiobjective optimization problem
    """
    def __init__(
        self,
        epsilon: float = 0.05, # Set the epsilon-accuracy condition
        iter_mul: int = 10, # Set the evaluation budget, i.e. iter_mul x dim
        max_exp: int = 30, # Set the number of experiments
        
        # Set some parameters for b_n and the evaluation budget
        n_iter_i: int = 1,
        iter_adjust:int = 0,
        n_iter:int = None,
        nu = 0.2, 
        sigma = 0.1,

        # List to store important values
        regret_iter:list = [],
        r_optimal_iter:list = [],
        bounds_iter:list = [],
    

        #standard atlas parameters
        goal: str = "maximize",
        batch_size: int = 1,
        random_seed: Optional[int] = None,
        use_descriptors: bool = False,
        num_init_design: int = 5,
        init_design_strategy: str = "random",
        acquisition_type: str = "ucb",  # ei, ucb, variance, general
        acquisition_optimizer_kind: str = "gradient",  # gradient, genetic
        is_moo: bool = False,
        #### below are parameters that are mostly not used
        feas_strategy: Optional[str] = "naive-0",
        feas_param: Optional[float] = 0.2,
        vgp_iters: int = 2000,
        vgp_lr: float = 0.1,
        max_jitter: float = 1e-1,
        la_threshold: float = 0.5,
        known_constraints: Optional[List[Callable]] = None,
        general_parameters: Optional[List[int]] = None,
        scalarizer_kind: Optional[str] = "Hypervolume",
        moo_params: Dict[str, Union[str, float, int, bool, List]] = {},
        goals: Optional[List[str]] = None,
        golem_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        local_args = {
                key: val for key, val in locals().items() if key != "self"
            }

        
        super().__init__(**local_args)

        self.regret = 10*self.epsilon
        

    def build_train_regression_gp(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> gpytorch.models.ExactGP:

        """Build the regression GP model and likelihood"""
        # infer the model based on the parameter types, do multi-start as in the paper
        
        if self.problem_type in [
            "fully_continuous",
            "fully_discrete",
            "mixed_disc_cont",
        ]:
            ls_prior = [0.01, 0.1, 1, 10]
            ml_value = np.zeros(len(ls_prior))
            gp_temp = []
            for idx, ls in enumerate(ls_prior):
                kernel = ScaleKernel(RBFKernel(lengthscale_prior=ls_prior))
                model = SingleTaskGP(train_x, train_y, covar_module=kernel)

                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                # fit the GP
                start_time = time.time()
                with gpytorch.settings.cholesky_jitter(self.max_jitter):
                    fit_gpytorch_model(mll)
                gp_train_time = time.time() - start_time
                Logger.log(
                    f"Regression surrogate GP trained in {round(gp_train_time,3)} sec using {ls} as a lengthscale prior",
                    "INFO",
                )
                output = model(train_x)
                ml_value[idx] = mll(output, train_y)
                gp_temp.append(model)
            gp = gp_temp[np.min(np.argmax(ml_value))]

            return gp

           
        elif self.problem_type == "fully_categorical":
            if self.has_descriptors:
                raise NotImplementedError
            else:
                raise NotImplementedError
        elif "mixed_cat_" in self.problem_type:
            raise NotImplementedError
        else:
            raise NotImplementedError

        return model
    
    def compute_expansion_trigger():
        return
    
    def compute_new_search_space() -> ParameterSpace:
        return
    
    def get_aqcf_min_max(
        self,
        reg_model: gpytorch.models.ExactGP,
        beta: float,
        f_best_scaled: torch.Tensor,
        num_samples: int = 2000,
    ) -> Tuple[int, int]:
        """computes the min and max value of the acquisition function without
        the feasibility contribution. These values will be used to approximately
        normalize the acquisition function
        """

        if self.acquisition_type == "ucb":
            acqf = UpperConfidenceBound(
                reg_model,
                beta=torch.tensor([beta]).repeat(self.batch_size),
                objective=None,
                maximize=False,
            )
        
        else:
            raise NotImplementedError
     
        samples, _ = propose_randomly(
            num_samples,
            self.param_space,
            self.has_descriptors,
        )

        if (
            self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            samples = forward_normalize(
                samples, self.params_obj._mins_x, self.params_obj._maxs_x
            )

        acqf_vals = acqf(
            torch.tensor(samples)
            .view(samples.shape[0], 1, samples.shape[-1])
            .double()
        )
        min_ = torch.amin(acqf_vals).item()
        max_ = torch.amax(acqf_vals).item()

        if np.abs(max_ - min_) < 1e-6:
            max_ = 1.0
            min_ = 0.0

        return min_, max_
    
    
    def set_param_space(
        self, 
        super_param_space: ParameterSpace,
        func_param_space: ParameterSpace
    ): 

        super.param_space = super_param_space
        self.func_param_space = func_param_space

        return
    

    def _ask(self) -> List[ParameterVector]:

        #set search space
        bounds = []
        for tup in self.func_param_space.param_bounds:
            lower,upper = tup
            bounds.append([lower, upper])
            
        (
                self.train_x_scaled_cla,
                self.train_y_scaled_cla,
                self.train_x_scaled_reg,
                self.train_y_scaled_reg,
            ) = self.build_train_data()

        

        self.model = self.build_train_regression_gp(self.train_x_scaled_reg, self.train_y_scaled_reg)

    
        # Extract Gaussian Process hyper-parameters
        # kernel_k2 is the length-scale
        # theta_n is the scale factor
        raw_outputscale =self.model.covar_module.raw_outputscale
        constraint = self.model.covar_module.raw_outputscale_constraint
        kernel_k1 = constraint.transform(raw_outputscale).item()
        theta_n = np.sqrt(kernel_k1)
        kernel_k2 = np.float64(self.model.covar_module.base_kernel.raw_lengthscale.item())

        self.n_init_points, self.input_dim = np.shape(self.train_x_scaled_reg)
        


        # Re-arrange the initial random observations

        self.max_iter = self.iter_mul*self.input_dim
        Y_max = np.zeros((self.max_iter+self.n_init_points, 1))
        for i in range(self.n_init_points):
            Y_max[i] = np.max(self.train_y_scaled_reg[0:i+1])


        # Compute the acquisition function at the observation points
        # b_n is found by using the correct formula
        lengthscale = kernel_k2
        thetan_2 = kernel_k1
        radius = np.abs(np.max(bounds[:, 1]-bounds[:, 0]))
        b = 1/2*np.sqrt(2)*np.sqrt(thetan_2)/lengthscale
        a = 1
        tau_n = (4*self.input_dim+4)*np.log(self.n_iter) + 2*np.log(2*np.pi**2/(3*self.sigma)) \
                + 2*self.input_dim*np.log(self.input_dim*b*radius*np.sqrt(np.log(4*self.input_dim*a/self.sigma)))
        b_n = np.sqrt(np.abs(self.nu*tau_n))
        

        f_preds = self.model(self.train_x_scaled_reg)
        y_init_mean, y_init_std = f_preds.mean.detach().numpy(), f_preds.variance.detach().numpy()
        acq_init = y_init_mean.ravel() + b_n*y_init_std.ravel()
        print('b_n is: {}'.format(b_n))

        # Optimize the acquisition function

        x_max = self.get_aqcf_min_max
        x_max = acq_maximize_fixopt(gp, b_n, bounds)
        max_acq = acq_gp(x_max, gp, b_n)
        y_acq = acq_gp(np.asarray(X_init), gp, b_n)

        # Compute the maximum regret
        X_init_temp = X_init.copy()
        X_init_temp.append(x_max)
        Y_lcb_temp = acq_lcb(np.asarray(X_init_temp), gp, b_n)
        regret = max_acq - np.max(Y_lcb_temp)
        regret_iter.append(regret)
        print('Regret: {}'.format(regret))

        # Check if regret < 0, redo the optimization, typically redo the optimization
        # with starting point in X_init_temp

        # Expand if regret < epsilon or the first iteration
            # Expand the search space based on analytical formula in Theorem 1

            # Find d_{gamma}
            # For SE kernel, it's easy to find values of r to make k(r) < gamma

            # Recompute bounds

            # Adjust iteration after adjusting bounds

            # Re-optimize the acquisition function with the new bound
            # and the new b_n

            # Save some parameters of the bound
        
        # Check if the acquistion function argmax is at infinity
        # Then re-optimize within the bound that has the largest Y value
            # Set a minimal number of local optimizations
        
        # Compute y_max


        # Update the kernel


        # Extract Gaussian Process hyper-parameters and update prior kernel
        # kernel_k2 is the length-scale
        # theta_n is the scale factor


        

        



        return


    
    

