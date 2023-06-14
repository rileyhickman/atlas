import os
import pickle
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from scipy.optimize import minimize

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
from gpytorch.constraints import Positive, Interval

from botorch.optim.fit import fit_gpytorch_scipy
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
from atlas.optimizers.acquisition_optimizers import (
    GradientOptimizer, GeneticOptimizer
)
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

from atlas.optimizers.acqfs import (
    LowerConfidenceBound,
    FeasibilityAwareLCB,
    FeasibilityAwareEI,
    FeasibilityAwareGeneral,
    FeasibilityAwareQEI,
    FeasibilityAwareVarainceBased,
    FeasibilityAwareUCB,
    VarianceBased,
    create_available_options,
    get_batch_initial_conditions,
)

from numpy.linalg import pinv

from olympus.objects import (
    ObjectParameter,
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)

from sklearn.preprocessing import normalize


class DynamicSSPlanner(BasePlanner):
    """Wrapper for GP-based Bayesian optimization using a dynamic search space bzsed on Ha et al.
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
    
        
        # Set some parameters for b_n and the evaluation budget
        nu = 0.2, 
        sigma = 0.1,    

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
        use_min_filter:bool=True,
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

        self.epsilon= epsilon
        self.iter_mul = iter_mul

        self.regret = 10*self.epsilon

        self.nu = nu
        self.sigma = sigma

        self.n_iter_i: int = 1
        self.iter_adjust:int = 0
        self.n_iter:int = None

        self.func_param_space: ParameterSpace = None
        self.param_space: ParameterSpace = None
        

        # List to store important values
        self.regret_iter = []
        self.r_optimal_iter = []
        self.bounds_iter =  []
    
    def _ask(self) -> List[ParameterVector]:

        """query the planner for a batch of new parameter points to measure"""
        # if we have all nan values, just keep randomly sampling

        
        if np.logical_or(
            len(self._values) < self.num_init_design,
            np.all(np.isnan(self._values)),
        ):
            # set parameter space for the initial design planner
            self.init_design_planner.set_param_space(self.func_param_space)

            # sample using initial design strategy (with same batch size)
            return_params = []
            for _ in range(self.batch_size):
                # TODO: this is pretty sloppy - consider standardizing this
                if self.init_design_strategy == "random":
                    self.init_design_planner._tell(
                        iteration=self.num_init_design_completed
                    )
                else:
                    self.init_design_planner.tell()
                rec_params = self.init_design_planner.ask()
                if isinstance(rec_params, list):
                    return_params.append(rec_params[0])
                elif isinstance(rec_params, ParameterVector):
                    return_params.append(rec_params)
                else:
                    raise TypeError
                self.num_init_design_completed += (
                    1  # batch_size always 1 for init design planner
                )
            final_params = return_params
        
        else:

            n_iter = self.n_iter_i - self.iter_adjust

            
            #set search space
            bounds = []
            for tup in self.func_param_space.param_bounds:
                lower,upper = tup
                bounds.append([lower, upper])
            bounds = np.array(bounds)

        
            print("Bounds {}".format(bounds))
            print(f"n_iter: {n_iter}")
            print(f"n_iter_i:{self.n_iter_i}")
            print(f"iter_adjust: {self.iter_adjust}")


    
            (
                    self.train_x_scaled_cla,
                    self.train_y_scaled_cla,
                    self.train_x_scaled_reg,
                    self.train_y_scaled_reg,
                ) = self.build_train_data()

            self.reg_model = self.build_train_regression_gp2(self.train_x_scaled_reg, self.train_y_scaled_reg)
            self.cla_model, self.cla_likelihood = None, None
            f_best_argmin = torch.argmin(self.train_y_scaled_reg)
            f_best_scaled = self.train_y_scaled_reg[f_best_argmin][0].float()
            infeas_ratio = (
                    torch.sum(self.train_y_scaled_cla)
                    / self.train_x_scaled_cla.size(0)
                ).item()

            # Extract Gaussian Process hyper-parameters
            # kernel_k2 is the length-scale
            # theta_n is the scale factor

            self.n_init_points, self.input_dim = np.shape(self.train_x_scaled_reg.detach().numpy())

            raw_outputscale =self.reg_model.covar_module.outputscale
            raw_lengthscale = self.reg_model.covar_module.base_kernel.raw_lengthscale
            outputscale_constraint = self.reg_model.covar_module.raw_outputscale_constraint
            ls_constraint = self.reg_model.covar_module.base_kernel.raw_lengthscale_constraint

            kernel_k1 = self.reg_model.covar_module.outputscale.detach().numpy().item()
            theta_n = np.sqrt(kernel_k1)
            kernel_k2 = np.float64(ls_constraint.transform(raw_lengthscale).item())

        
            # Re-arrange the initial random observations
            self.max_iter = self.iter_mul*self.input_dim
            Y_max = np.zeros((self.max_iter+self.n_init_points, 1))
            for i in range(self.n_init_points):
    
                Y_max[i] = np.max(torch.flatten(self.train_y_scaled_reg[0:i+1]).detach().numpy())


            # Compute the acquisition function at the observation points
            # b_n is found by using the correct formula
            lengthscale = kernel_k2
            thetan_2 = kernel_k1

            radius = np.abs(np.max(bounds[:, 1]-bounds[:, 0]))
            b = 1/2*np.sqrt(2)*np.sqrt(thetan_2)/lengthscale
            a = 1
            tau_n = (4*self.input_dim+4)*np.log(n_iter) + 2*np.log(2*np.pi**2/(3*self.sigma)) \
                    + 2*self.input_dim*np.log(self.input_dim*b*radius*np.sqrt(np.log(4*self.input_dim*a/self.sigma)))
            b_n = np.sqrt(np.abs(self.nu*tau_n))

            #print(lengthscale)
            #print(thetan_2)
            #print(radius)
            #print(b)
            #print(a)
            #print(tau_n)
            #print(b_n)
        

            f_preds = self.reg_model(self.train_x_scaled_reg)
            y_init_mean, y_init_std = f_preds.mean.detach().numpy(), f_preds.variance.detach().numpy()
            acq_init = y_init_mean.ravel() + b_n*y_init_std.ravel()
            #print('b_n is: {}'.format(b_n))

            # Optimize the acquisition function

            f_best_argmin = torch.argmin(self.train_y_scaled_reg)
            f_best_scaled = self.train_y_scaled_reg[f_best_argmin][0].float()
            acqf_min_max = self.get_aqcf_min_max(self.reg_model, f_best_scaled, b_n=b_n)

            if self.acquisition_type == "ucb":
                self.ucbacqf = FeasibilityAwareUCB(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    self.func_param_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    beta=torch.tensor([b_n]),
                )

                self.lcbacqf = FeasibilityAwareLCB(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    self.func_param_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    beta=torch.tensor([b_n]),
                )
                
            else:
                raise NotImplementedError


            if self.acquisition_optimizer_kind=='gradient':
                    acquisition_optimizer = GradientOptimizer(
                        self.params_obj,
                        self.acquisition_type,
                        self.ucbacqf,
                        self.known_constraints,
                        self.batch_size,
                        self.feas_strategy,
                        self.fca_constraint,
                        self._params,
                        timings_dict={},
                        use_reg_only=False,

                    )
            elif self.acquisition_optimizer_kind == 'genetic':
                acquisition_optimizer = GeneticOptimizer(
                    self.params_obj,
                    self.acquisition_type,
                    self.ucbacqf,
                    self.known_constraints,
                    self.batch_size,
                    self.feas_strategy,
                    self.fca_constraint,
                    self._params,
                    timings_dict={},
                    use_reg_only=False,

                )
            
            acquisition_optimizer.param_space = self.func_param_space
            
            self.mins_x = acquisition_optimizer._mins_x
            self.maxs_x = acquisition_optimizer._maxs_x



            # print("Unnormalized x:")
            # print(reverse_normalize(self.train_x_scaled_reg, self.mins_x, self.maxs_x))

            # print("Unormalized x:")
            # print(self.train_x_scaled_reg, self.mins_x, self.maxs_x)
    
            final_params = acquisition_optimizer.optimize()
            x_max = forward_normalize(torch.tensor([final_params[0].to_list()]), self.mins_x, self.maxs_x)
            final_params = acquisition_optimizer.optimize()
            max_acq = self.ucbacqf(x_max).detach().numpy()


            # Compute the maximum regret
            X_init_temp = self.train_x_scaled_reg.numpy().tolist()
            batch_size = len(X_init_temp)
            #y_acq = np.zeros(len(self.train_x_scaled_reg))
            #y_acq = self.ucbacqf(torch.reshape(self.train_x_scaled_reg, (batch_size,1,self.input_dim))).detach().numpy()
            X_init_temp.append(x_max.flatten().tolist())
            Y_lcb_temp = self.lcbacqf(torch.reshape(self.train_x_scaled_reg, (batch_size,1,self.input_dim))).detach().numpy()
            regret = max_acq - np.max(Y_lcb_temp)
            self.regret_iter.append(regret)
            print('Regret: {}'.format(regret))

            # Check if regret < 0, redo the optimization, typically redo the optimization
            # with starting point in X_init_temp
            indices_max_X = np.argsort(Y_lcb_temp)[::-1]
            n_search_X = np.min([20, len(indices_max_X)])
            if (regret + 1/n_iter**2 < 0):
                print('Regret < 0, redo the optimization')
                for i in range(n_search_X):
                    final_params = acquisition_optimizer.optimize()
                    x_max_temp = forward_normalize(torch.tensor([final_params[0].to_list()]), self.mins_x, self.maxs_x)
                    max_acq_temp = self.ucbacqf(x_max_temp).detach().numpy()

                    # Store it if better than previous minimum(maximum).
                    if max_acq is None or max_acq_temp >= max_acq:
                        print("triggered")
                        x_max = x_max_temp
                        max_acq = max_acq_temp
                    
                    # Recompute regret
                    regret = max_acq - np.max(Y_lcb_temp)
                print("FINAL PARAMS:")
                print(final_params)
                print('Regret: {}'.format(regret))
            
            

            # Expand if regret < epsilon or the first iteration

            if self.compute_expansion_trigger(regret, n_iter):
                print('Expanding bounds')
                #Y = (self.train_y_scaled_reg.numpy()-np.mean(self.train_y_scaled_reg.numpy()))/(np.max(self.train_y_scaled_reg.numpy())-np.min(self.train_y_scaled_reg.numpy()))

                X = reverse_normalize(self.train_x_scaled_reg, self.mins_x, self.maxs_x)
                X, Y = self.build_train_data_custom()

                K = self.gram_matrix(X, 1, kernel_k2)

            
                bounds, scale_l, self.bound_len, n_iter= self.compute_new_search_space(K, Y, X, b_n, theta_n, kernel_k1, kernel_k2, acq_init, n_iter, bounds)
                


                # Re-optimize the acquisition function with the new bound
                # and the new b_n
                n_iter = self.n_iter_i - self.iter_adjust 


                raw_outputscale =self.reg_model.covar_module.raw_outputscale
                raw_lengthscale = self.reg_model.covar_module.base_kernel.raw_lengthscale
                outputscale_constraint = self.reg_model.covar_module.raw_outputscale_constraint
                ls_constraint = self.reg_model.covar_module.base_kernel.raw_lengthscale_constraint

                thetan_2 = outputscale_constraint.transform(raw_outputscale).item()
                lengthscale = np.float64(ls_constraint.transform(raw_lengthscale).item())

                radius = np.abs(np.max(bounds[:, 1]-bounds[:, 0]))
                b = 1/2*np.sqrt(2)*np.sqrt(thetan_2)/lengthscale
                a = 1
                tau_n = (4*self.input_dim+4)*np.log(n_iter) + 2*np.log(2*np.pi**2/(3*self.sigma)) \
                        + 2*self.input_dim*np.log(self.input_dim*b*radius*np.sqrt(np.log(4*self.input_dim*a/self.sigma)))
                b_n = np.sqrt(np.abs(self.nu*tau_n))

            
                if self.acquisition_type == "ucb":
                    self.ucbacqf = FeasibilityAwareUCB(
                        self.reg_model,
                        self.cla_model,
                        self.cla_likelihood,
                        self.func_param_space,
                        f_best_scaled,
                        self.feas_strategy,
                        self.feas_param,
                        infeas_ratio,
                        acqf_min_max,
                        beta=torch.tensor([b_n])
                    )
                else: raise NotImplementedError

                if self.acquisition_optimizer_kind=='gradient':
                    acquisition_optimizer = GradientOptimizer(
                        self.params_obj,
                        self.acquisition_type,
                        self.ucbacqf,
                        self.known_constraints,
                        self.batch_size,
                        self.feas_strategy,
                        self.fca_constraint,
                        self._params,
                        timings_dict={},
                        use_reg_only=False,

                    )
                elif self.acquisition_optimizer_kind == 'genetic':
                    acquisition_optimizer = GeneticOptimizer(
                        self.params_obj,
                        self.acquisition_type,
                        self.ucbacqf,
                        self.known_constraints,
                        self.batch_size,
                        self.feas_strategy,
                        self.fca_constraint,
                        self._params,
                        timings_dict={},
                        use_reg_only=False,

                    )
                acquisition_optimizer.param_space = self.func_param_space
            
                final_params = acquisition_optimizer.optimize()
                print("FINAL PARAMS:")
                print(final_params)
                x_max = forward_normalize(torch.tensor([final_params[0].to_list()]), self.mins_x, self.maxs_x)
                max_acq = self.ucbacqf(x_max)

                y_acq = np.zeros(len(self.train_x_scaled_reg))

                # Save some parameters of the bound
                X_init_bound = X
                Y_bound = np.copy(Y)
                lengthscale_bound = lengthscale = np.float64(self.reg_model.covar_module.base_kernel.raw_lengthscale.item())
                scale_l_bound = scale_l
            
        
            # Check if the acquistion function argmax is at infinity
            # Then re-optimize within the bound that has the largest Y value
            if ((max_acq - b_n*np.sqrt(thetan_2)) >= -self.epsilon) & ((max_acq - b_n*np.sqrt(thetan_2)) <= 0):

                X_init_bound, Y_bound = self.build_train_data_custom()

                final_params = self.argmax_infinity_reoptimize(
                                final_params, 
                                b_n, 
                                Y_bound, 
                                X_init_bound, 
                                self.bound_len, 
                                self.func_param_space, 
                                f_best_scaled, 
                                infeas_ratio, 
                                acqf_min_max
                                )
                print("FINAL PARAMS:")
                print(final_params)
        
        self.n_iter_i +=1
            
        return final_params


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
            ls_prior = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            ml_value = np.zeros(len(ls_prior))
            gp_temp = []
          
            flattened_train_y = torch.flatten(train_y)
        

            for idx, ls in enumerate(ls_prior):

                rbfkernel = RBFKernel()
                scaled_rbf_kernel = ScaleKernel(rbfkernel)
                #scaled_rbf_kernel = ScaleKernel(rbfkernel, outputscale_constraint=Interval(lower_bound=0.1, upper_bound=100000))

                
                model = SingleTaskGP(train_x, train_y, covar_module=scaled_rbf_kernel)
                model.covar_module.base_kernel.lengthscale = ls
                model.covar_module.outputscale = 1

                kernel_k1 = model.covar_module.outputscale.detach().numpy().item()
                lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy().item()
                #print(f"kernel k1 before opt:{kernel_k1}")
                #print(f"lengthscale before opt:{lengthscale}")

                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                mll.train()
                model.train()
                
                # fit the GP
                start_time = time.time()
                with gpytorch.settings.cholesky_jitter(self.max_jitter):
                    fit_gpytorch_scipy(mll)
                gp_train_time = time.time() - start_time
                # Logger.log(
                #     f"Regression surrogate GP trained in {round(gp_train_time,3)} sec using {ls} as a lengthscale prior",
                #     "INFO",
                # )

                # kernel_k1 = model.covar_module.outputscale.detach().numpy().item()
                # lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy().item()
                # #print(f"kernel k1 after opt:{kernel_k1}")
                # #print(f"lengthscale after opt:{lengthscale}")
                # #print()

                model.eval()
                model.eval()

                output = model(train_x)
                loss = mll(output, flattened_train_y)
                ml_value[idx] = loss.detach().numpy().item()
                
                gp_temp.append(model)

                #print(ml_value)
        

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
    
    def build_train_regression_gp2(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> gpytorch.models.ExactGP:

        
        """Build the regression GP model and likelihood"""
        # infer the model based on the parameter types
        if self.problem_type in [
            "fully_continuous",
            "fully_discrete",
            "mixed_disc_cont",
        ]:  
            rbfkernel = RBFKernel()
            scaled_rbf_kernel = ScaleKernel(rbfkernel)
            model = SingleTaskGP(train_x, train_y, covar_module=scaled_rbf_kernel)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit the GP
        start_time = time.time()

        
        with gpytorch.settings.cholesky_jitter(self.max_jitter):
            fit_gpytorch_model(mll)
        gp_train_time = time.time() - start_time
        Logger.log(
            f"Regression surrogate GP trained in {round(gp_train_time,3)} sec",
            "INFO",
        )

        kernel_k1 = model.covar_module.outputscale.item()
        lengthscale = model.covar_module.base_kernel.lengthscale.item()

        print(f"kernel k1 after opt:{kernel_k1}")
        print(f"lengthscale after opt:{lengthscale}")

        return model
    
    def compute_expansion_trigger(self, regret, n_iter):
        if (regret <= self.epsilon - 1/n_iter**2) | (self.n_iter_i == 1):
            return True
        return False
    
    def compute_new_search_space(self, k, y, x, b_n, theta_n, kernel_k1, kernel_k2, aq_init, n_iter, bounds):

        K_inv = pinv(k)
        b = np.matmul(K_inv, y)
        b_pos = b[b >= 0]
        b_neg = b[b <= 0]
        U, Sigma, V = np.linalg.svd(K_inv)
        lambda_max = np.max(Sigma)
        n = self.train_x_scaled_reg.size(dim=0)
    
        gamma = min(0.25*self.epsilon/max(np.sum(b_pos), -np.sum(b_neg)),
                    1/b_n*np.sqrt((0.5*self.epsilon*b_n*theta_n-0.0625*self.epsilon**2)/(n*lambda_max)))
        print(f"gamma: {gamma}")
        print(f"kernel_k1: {kernel_k1}")
        print(f"kernel_k2: {kernel_k2}")
        

        if (gamma > 1):
            raise ValueError('Gamma needs to be smaller than 1. Something wrong!')
        
        scale_l = np.sqrt(-2*np.log(gamma/kernel_k1))

        if np.isnan(scale_l):
            scale_l = 2
        print('The scale is: {}'.format(scale_l))

        bounds_ori_all = []
        for n_obs in range(len(aq_init)):
            X0 = x[n_obs]
            # Set the rectangle bounds
            bounds_ori_temp = np.asarray((X0 - scale_l*kernel_k2,
                                            X0 + scale_l*kernel_k2))
            bounds_ori = bounds_ori_temp.T
            bounds_ori_all.append(bounds_ori)

        self.bound_len = scale_l*kernel_k2
        temp = np.asarray(bounds_ori_all)

        temp_min = np.min(temp[:, :, 0], axis=0)
        temp_max = np.max(temp[:, :, 1], axis=0)

        bounds_ori = np.stack((temp_min, temp_max)).T
        bounds_new = bounds_ori.copy()

        print('Old bound: {}'.format(bounds))
        print('New bound: {}'.format(bounds_new))
  

        new_func_parameter_space = ParameterSpace()
        param_names = self.func_param_space.param_names

        for idx in range(self.input_dim):
            new_func_parameter_space.add(
                ParameterContinuous(
                    name=param_names[idx],
                    low = bounds_new[idx][0],
                    high = bounds_new[idx][1]
                )
            )

        self.bounds_iter.append(self.func_param_space)
        self.func_param_space = new_func_parameter_space


        # Adjust iteration after adjusting bounds
        self.iter_adjust += (n_iter - 1)
        return bounds_new, scale_l, self.bound_len, n_iter
    
    def get_aqcf_min_max(
        self,
        reg_model: gpytorch.models.ExactGP,
        f_best_scaled: torch.Tensor,
        num_samples: int = 2000,
        b_n: Optional[float] = None
    ) -> Tuple[int, int]:
        """computes the min and max value of the acquisition function without
        the feasibility contribution. These values will be used to approximately
        normalize the acquisition function
        """

        if self.acquisition_type == "ucb":
            acqf = UpperConfidenceBound(
                reg_model,
                beta=torch.tensor([b_n]).repeat(self.batch_size),
                objective=None,
                maximize=False,
            )
        
        else:
            raise NotImplementedError
    
        samples, _ = propose_randomly(
            num_samples,
            self.func_param_space,
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

        self.param_space = super_param_space

        self.bounds_iter.append(self.func_param_space)
        self.func_param_space = func_param_space

        return
    
    def gram_matrix(self, X, k1, k2):
        dim = len(X)
        K = np.zeros((dim, dim))
        for i in range(0, dim):
            for j in range(0, dim):
                K[i, j] = k1*np.exp(-1/(2*k2**2)*np.linalg.norm(X[i]-X[j])**2)

        return K
    
    def argmax_infinity_reoptimize(
        self, 
        final_params, 
        b_n, 
        Y_bound, 
        X_init_bound, 
        bound_len, 
        bounds, 
        f_best_scaled, 
        infeas_ratio, 
        acqf_min_max
        ):

        print('Re-optimize within smaller spheres')
        indices_max = np.argsort(Y_bound)[::-1]
        
        # Set a minimal number of local optimizations
        n_search = np.min([5, len(indices_max)])

        raw_outputscale =self.reg_model.covar_module.raw_outputscale
        constraint = self.reg_model.covar_module.raw_outputscale_constraint
        kernel_k1 = constraint.transform(raw_outputscale).item()
        for i in range(n_search):
            X0 = X_init_bound[indices_max[i]]
            bounds_new = np.asarray((X0 - bound_len,
                                    X0 + bound_len))
            bounds_new = np.squeeze(bounds_new.T)

            redo_bounds_space = ParameterSpace()
            param_names = self.func_param_space.param_names

            for idx in range(len(bounds_new)):
                redo_bounds_space.add(
                    ParameterContinuous(
                        name=param_names[idx],
                        low = bounds_new[idx][0],
                        high = bounds_new[idx][1]
                    )
                )
            
            
            if self.acquisition_type == "ucb":
                self.ucbacqf = FeasibilityAwareUCB(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    redo_bounds_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    beta=torch.tensor([b_n]),
                )
            else: raise NotImplementedError

            if self.acquisition_optimizer_kind=='gradient':
                    acquisition_optimizer = GradientOptimizer(
                        self.params_obj,
                        self.acquisition_type,
                        self.ucbacqf,
                        self.known_constraints,
                        self.batch_size,
                        self.feas_strategy,
                        self.fca_constraint,
                        self._params,
                        timings_dict={},
                        use_reg_only=False,

                    )
            elif self.acquisition_optimizer_kind == 'genetic':
                acquisition_optimizer = GeneticOptimizer(
                    self.params_obj,
                    self.acquisition_type,
                    self.ucbacqf,
                    self.known_constraints,
                    self.batch_size,
                    self.feas_strategy,
                    self.fca_constraint,
                    self._params,
                    timings_dict={},
                    use_reg_only=False,

                )
            acquisition_optimizer.param_space = self.func_param_space

            return_params_local = acquisition_optimizer.optimize()
            x_max_local = forward_normalize(torch.tensor([return_params_local[0].to_list()]), self.mins_x, self.maxs_x)
            max_acq_local = self.ucbacqf(x_max_local).detach().numpy()
            
            max_inf = b_n*np.sqrt(kernel_k1)
        
            if np.abs(max_acq_local-max_inf) <= self.epsilon:

                final_params = return_params_local
                
                return final_params
        return final_params
          
    def build_train_data_custom(self) -> Tuple[torch.Tensor, torch.tensor]:
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
        

        # always forward transform the objectives for the regression problem
        train_y_reg = forward_normalize(
            train_y_reg, self._means_y, self._stds_y
        )

        # convert to torch tensors and return
        return (
            train_x_reg,
            train_y_reg
        )



    
    
'''
raw_outputscale =self.reg_model.covar_module.raw_outputscale
constraint = self.reg_model.covar_module.raw_outputscale_constraint
kernel_k1 = constraint.transform(raw_outputscale).item()

lengthscale = np.float64(self.reg_model.covar_module.base_kernel.raw_lengthscale.item())
thetan_2 = kernel_k1

'''