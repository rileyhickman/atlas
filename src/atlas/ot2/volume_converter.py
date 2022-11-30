#!/usr/bin/env python

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import olympus
import pandas as pd
import seaborn as sns
import yaml
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterVector,
)
from olympus.planners import RandomSearch

from atlas.optimizers.acqfs import create_available_options
from atlas.optimizers.gp.planner import BoTorchPlanner


def get_param_space(
    campaign_config: Dict[Any, Any],
) -> Tuple[ParameterSpace, ParameterDiscrete, Dict[str, List[str]]]:
    """Generate Olympus ParameterSpace object based on the supplied
    configuration for the parameters
    """
    full_space_instructions = {}
    func_param_space = ParameterSpace()
    full_param_space = ParameterSpace()
    prep = campaign_config["preparation"]
    drug_names = [p for p in prep.keys() if prep[p]["type"] == "drug"]
    lipid_names = [p for p in prep.keys() if prep[p]["type"] == "lipid"]
    solvent_names = [p for p in prep.keys() if prep[p]["type"] == "solvent"]

    for component in drug_names + lipid_names:
        if prep[component]["levels"]:
            # we have levels --> functional parameter
            param_to_add = ParameterDiscrete(
                name=component, options=prep[component]["levels"]
            )
            func_param_space.add(param_to_add)
            full_param_space.add(param_to_add)
        else:
            # no levels, non-functional parameter
            param_to_add = ParameterContinuous(
                name=component,
                low=0.0,
                high=1.0,
            )
            full_param_space.add(param_to_add)
            # add instructions, list of parameter names that dictate value
            full_space_instructions[component] = prep[component][
                "fractional_range"
            ]

    return func_param_space, full_param_space, full_space_instructions


def get_value_space(campaign_config: Dict[Any, Any]) -> ParameterSpace:
    """Generate Olympus ParameterSpace object based on the supplied
    configuration for the objectives
    """
    objectives = campaign_config["objectives"]
    value_space = ParameterSpace()
    for obj in objectives.keys():
        value_space.add(ParameterContinuous(name=objectives[obj]["name"]))
    return value_space


def compute_target_conc(campaign_config: Dict[Any, Any]) -> Dict[Any, Any]:
    """Compute molar target concentrations and total solvent values"""
    prep = campaign_config["preparation"]
    drug_names = [p for p in prep.keys() if prep[p]["type"] == "drug"]
    lipid_names = [p for p in prep.keys() if prep[p]["type"] == "lipid"]
    solvent_names = [p for p in prep.keys() if prep[p]["type"] == "solvent"]

    for component in drug_names + lipid_names:
        component_info = prep[component]
        # target concentration (umol/mL)
        target_conc_mol = (
            component_info["target_conc"] / component_info["mol_wt"] * 1000.0
        )
        component_info["target_conc_mol"] = target_conc_mol
        # total solvent (mL)
        total_solvent = (
            component_info["mass_weighted"] / component_info["target_conc"]
        )
        component_info["total_solvent"] = total_solvent

    return campaign_config


def compute_total_lipid_masses(
    campaign_config: Dict[Any, Any],
    batch_params: List[ParameterVector],
) -> np.ndarray:
    total_lipid_masses = []
    prep = campaign_config["preparation"]
    lipid_names = [p for p in prep.keys() if prep[p]["type"] == "lipid"]

    for params in batch_params:
        total_mass = 0.0
        for name in lipid_names:
            total_mass += prep[name]["mol_wt"] * params[name]
        total_lipid_masses.append(
            campaign_config["processing_params"]["total_lipid_conc"]
            * total_mass
        )
    return np.array(total_lipid_masses)


def compute_transfer_volumes(
    campaign_config: Dict[Any, Any],
    batch_params: List[ParameterVector],
) -> Dict[str, np.ndarray]:
    volume_prec = campaign_config["general"]["volume_prec"]
    prep = campaign_config["preparation"]
    drug_names = [p for p in prep.keys() if prep[p]["type"] == "drug"]
    lipid_names = [p for p in prep.keys() if prep[p]["type"] == "lipid"]
    solvent_names = [p for p in prep.keys() if prep[p]["type"] == "solvent"]

    org_phase_per_stock = campaign_config["processing_params"][
        "org_phase_per_stock"
    ]
    org_phase_per_well = campaign_config["processing_params"][
        "org_phase_per_well"
    ]
    total_lipid_conc = campaign_config["processing_params"]["total_lipid_conc"]

    total_lipid_masses = compute_total_lipid_masses(
        campaign_config, batch_params
    )

    transfer_volumes = {}
    # compute drug volumes
    for name in drug_names:
        frac_amt = np.array([params[name] for params in batch_params])
        target_conc = prep[name]["target_conc"]
        vols = (
            frac_amt
            * total_lipid_masses
            / org_phase_per_well
            * org_phase_per_stock
            / target_conc
        )
        transfer_volumes[name] = np.around(vols, decimals=volume_prec)

    # compute lipid volumes --> use molar target volume instead
    for name in lipid_names:
        frac_amt = np.array([params[name] for params in batch_params])
        target_conc_mol = prep[name]["target_conc_mol"]
        vols = (
            frac_amt
            * total_lipid_conc
            / org_phase_per_well
            * org_phase_per_stock
            / target_conc_mol
            * 1000.0
        )

        transfer_volumes[name] = np.around(vols, decimals=volume_prec)

    # compute the solvent transfer volumes
    if not len(solvent_names) == 1:
        print("Multiple solvents not yet implemented...")
        quit()
    else:
        vols = org_phase_per_stock - np.sum(
            list(transfer_volumes.values()), axis=0
        )
        transfer_volumes[solvent_names[0]] = vols  # hardcoded

    return transfer_volumes


def func_to_full_params(
    func_batch_params: List[ParameterVector],
    full_param_space: ParameterSpace,
    full_space_instructions: Dict[str, List[str]],
) -> List[ParameterVector]:
    full_batch_params = []

    for func_params in func_batch_params:
        full_params_dict = func_params.to_dict().copy()
        for aux_param, instructions in full_space_instructions.items():
            full_params_dict[aux_param] = 1.0 - np.sum(
                [func_params[p] for p in instructions]
            )
        full_batch_params.append(
            ParameterVector().from_dict(
                full_params_dict, param_space=full_param_space
            )
        )

    return full_batch_params


# def check_stock_solutions(func_param_space, full_space_instructions, campaign_config):
#     ''' Checks all possible options to see if there are any volumes that are
#     < 20 uL and != 0.0
#     '''
#     tmp_campaign_config = campaign_config.copy()
#     # generate all possible options given the full param space

#     param_names = [p.name for p in func_param_space]
#     param_options = [p.options for p in func_param_space]
#     cart_product = list(itertools.product(*param_options))
#     cart_product = [ParameterVector().from_list(list(elem), param_space=func_param_space) for elem in cart_product]

#     # convert to full param space
#     full_product = func_to_full_params(cart_product, full_space_instructions)

#     is_sat_volumes = False
#     while not is_sat_volumes:

#         # compute tranfer volumes
#         transfer_volumes = compute_transfer_volumes(tmp_campaign_config, full_product)
#         # check the volumes
#         bad_target_conc = []
#         for component in transfer_volumes.keys():
#             vols = transfer_volumes[component]
#             bad_idx = np.where((vols<20.)&(vols!=0.0))[0]
#             if bad_idx.shape[0]>0:
#                 # we have some volumes that are not good, need to reduce the
#                 # target concentration of this component
#                 bad_target_conc.append(component)
#             else:
#                 # all volumes are good
#                 pass

#         if bad_target_conc == []:
#             # done
#             is_sat_volumes = True
#         else:
#             # decrase the target concentration for all the problematic components
#             for component in bad_target_conc:
#                 # TODO: implement this...
#                 pass

#     return None


if __name__ == "__main__":

    content = open("campaign_config.yaml", "r")
    campaign_config = yaml.full_load(content)
    # print(campaign_config)

    # compute molar target concentrations and total solvent volumes
    campaign_config = compute_target_conc(campaign_config)

    batch_size = campaign_config["general"]["batch_size"]

    (
        func_param_space,
        full_param_space,
        full_space_instructions,
    ) = get_param_space(campaign_config)
    value_space = get_value_space(campaign_config)
    print("\nfunc param space")
    print(func_param_space)
    print("\nfull param space")
    print(full_param_space)
    print("\nfull space instructions")
    print(full_space_instructions)
    print("\nvalue space")
    print(value_space)

    # functional campaign
    func_campaign = Campaign()
    func_campaign.set_param_space(func_param_space)
    func_campaign.set_value_space(value_space)

    # full campaign
    full_campaign = Campaign()
    full_campaign.set_param_space(full_param_space)
    full_campaign.set_value_space(value_space)

    # instantiate planner with functional parameter space
    # planner = RandomSearch(goal='maximize')
    planner = BoTorchPlanner(
        goal="maximize",
        batch_size=batch_size,
    )
    planner.set_param_space(func_param_space)

    # # check the stock solutions
    # check_stock_solutions(func_param_space, full_space_instructions, campaign_config)

    # ask for recommendations
    func_batch_params = planner.recommend(func_campaign.observations)

    assert len(func_batch_params) == batch_size
    print(f"\nFUNC PARAMS : {func_batch_params}")

    # convert to full space batch params
    full_batch_params = func_to_full_params(
        func_batch_params, full_param_space, full_space_instructions
    )
    print(f"\nFULL PARAMS : {full_batch_params}")

    # change the params from fractional amounts to transfer volumes for the OT2
    transfer_volumes = compute_transfer_volumes(
        campaign_config, full_batch_params
    )
    print("\nTRANSFER VOLUMES : ", transfer_volumes)

    # TODO: convert OT2 script to the transfer volumes
