#!/usr/bin/env python

import yaml
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import olympus
from olympus.objects import (
    ParameterContinuous,
    ParameterDiscrete,
    ParameterCategorical,
    ParameterVector
)
from olympus.campaigns import Campaign, ParameterSpace
from olympus.planners import RandomSearch

# from atlas.optimizers.gp.planner import BoTorchPlanner

def get_param_space(campaign_config):
    ''' Generate Olympus ParameterSpace object based on the supplied
    configuration for the parameters
    '''
    full_space_instructions = {}
    func_param_space = ParameterSpace()
    full_param_space = ParameterSpace()
    prep = campaign_config['preparation']
    for drug in prep['drugs'].keys():
        if prep['drugs'][drug]['levels']:

            # we have levels --> functional parameter
            param_to_add = ParameterDiscrete(
                    name=prep['drugs'][drug]['name'],
                    options=prep['drugs'][drug]['levels']
            )
            func_param_space.add(param_to_add)
            full_param_space.add(param_to_add)
        else:
            # no levels, non-functional parameter
            param_to_add = ParameterContinuous(
                name=prep['drugs'][drug]['name'],
                low=0.,
                high=1.,
            )
            full_param_space.add(param_to_add)
            # add instructions, list of parameter names that dictate value
            full_space_instructions[prep['drugs'][drug]['name']] = prep['drugs'][drug]['fractional_range']

    for lipid in prep['lipids'].keys():
        if prep['lipids'][lipid]['levels']:
            # we have levels --> functional parameter
            param_to_add = ParameterDiscrete(
                    name=prep['lipids'][lipid]['name'],
                    options=prep['lipids'][lipid]['levels']
            )
            func_param_space.add(param_to_add)
            full_param_space.add(param_to_add)
        else:
            # no levels, non-functional parameter
            param_to_add = ParameterContinuous(
                name=prep['lipids'][lipid]['name'],
                low=0.,
                high=1.,
            )
            full_param_space.add(param_to_add)
            # add instructions, list of parameter names that dictate value
            full_space_instructions[prep['lipids'][lipid]['name']] = prep['lipids'][lipid]['fractional_range']
    return func_param_space, full_param_space, full_space_instructions


def get_value_space(campaign_config):
    ''' Generate Olympus ParameterSpace object based on the supplied
    configuration for the objectives
    '''
    objectives = campaign_config['objectives']
    value_space = ParameterSpace()
    for obj in objectives.keys():
        value_space.add(
            ParameterContinuous(
                name=objectives[obj]['name']
            )
        )
    return value_space


#-----------------------------------------
# helper functions for volume calculations
# ----------------------------------------

def compute_target_conc(campaign_config):
    ''' Compute molar target concentrations and total solvent values
    '''

    prep = campaign_config['preparation']
    drugs = prep['drugs']
    lipids = prep['lipids']

    for drug in drugs.keys():
        drug_info = drugs[drug]
        # target concentration (umol/mL)
        target_conc_mol = drug_info['target_conc']/drug_info['mol_wt']*1000.
        drug_info['target_conc_mol'] = target_conc_mol
        # total solvent (mL)
        total_solvent = drug_info['mass_weighted']/drug_info['target_conc']
        drug_info['total_solvent'] = total_solvent

    for lipid in lipids.keys():
        lipid_info = lipids[lipid]
        # target concentration (umol/mL)
        target_conc_mol = lipid_info['target_conc']/lipid_info['mol_wt']*1000.
        lipid_info['target_conc_mol'] = target_conc_mol
        # total solvent (mL)
        total_solvent = lipid_info['mass_weighted']/lipid_info['target_conc']
        lipid_info['total_solvent'] = total_solvent

    return campaign_config

def compute_total_lipid_masses(campaign_config, batch_params):
    total_lipid_masses = []
    lipids = campaign_config['preparation']['lipids']

    for params in batch_params:
        total_mass = 0.
        for name, info in lipids.items():
            total_mass += info['mol_wt']*params[lipids[name]['name']]
        total_lipid_masses.append(
            campaign_config['processing_params']['total_lipid_conc']*total_mass
        )
    return np.array(total_lipid_masses)

def compute_transfer_volumes(campaign_config, batch_params,):
    volume_prec = campaign_config['general']['volume_prec']
    prep = campaign_config['preparation']
    drugs = prep['drugs']
    lipids = prep['lipids']

    total_lipid_masses = compute_total_lipid_masses(
        campaign_config, batch_params
    )

    transfer_volumes = {}
    # compute drug volumes
    for name, info in drugs.items():
        frac_amt = np.array([params[drugs[name]['name']] for params in batch_params])
        org_phase_per_stock = campaign_config['processing_params']['org_phase_per_stock']
        org_phase_per_well = campaign_config['processing_params']['org_phase_per_well']
        target_conc = drugs[name]['target_conc']
        vols = frac_amt*total_lipid_masses/org_phase_per_well*org_phase_per_stock/target_conc

        transfer_volumes[drugs[name]['name']] = np.around(vols, decimals=volume_prec)

    # compute lipid volumes --> use molar target volume instead
    for name, info in lipids.items():
        frac_amt = np.array([params[lipids[name]['name']] for params in batch_params])
        org_phase_per_stock = campaign_config['processing_params']['org_phase_per_stock']
        org_phase_per_well = campaign_config['processing_params']['org_phase_per_well']
        target_conc_mol = lipids[name]['target_conc_mol']
        total_lipid_conc = campaign_config['processing_params']['total_lipid_conc']
        vols = frac_amt*total_lipid_conc/org_phase_per_well*org_phase_per_stock/target_conc_mol*1000.

        transfer_volumes[lipids[name]['name']] = np.around(vols, decimals=volume_prec)

    return transfer_volumes

def func_to_full_params(
    func_batch_params,
    # func_param_space,
    # full_param_space,
    full_space_instructions
):
    full_batch_params = []

    for func_params in func_batch_params:
        full_params_dict = func_params.to_dict().copy()
        for aux_param, instructions in full_space_instructions.items():
            full_params_dict[aux_param] = 1. - np.sum([func_params[p] for p in instructions])

        full_batch_params.append(
            ParameterVector().from_dict(full_params_dict)
        )

    return full_batch_params


def check_stock_solutions():

    return None


if __name__ == '__main__':

    content = open('campaign_config.yaml', 'r')
    campaign_config = yaml.full_load(content)
    #print(campaign_config)

    # compute molar target concentrations and total solvent volumes
    campaign_config = compute_target_conc(campaign_config)

    batch_size = campaign_config['general']['batch_size']

    func_param_space, full_param_space, full_space_instructions = get_param_space(campaign_config)
    value_space = get_value_space(campaign_config)
    print('\nfunc param space')
    print(func_param_space)
    print('\nfull param space')
    print(full_param_space)
    print('\nfull space instructions')
    print(full_space_instructions)
    print('\nvalue space')
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
    planner = RandomSearch(goal='maximize')
    planner.set_param_space(func_param_space)

    # ask for recommendations
    func_batch_params = []
    for _ in range(batch_size):
        params = planner.recommend(func_campaign.observations)
        func_batch_params.extend(params)

    assert len(func_batch_params) == batch_size
    print(f'\nFUNC PARAMS : {func_batch_params}')

    # convert to full space batch params
    full_batch_params = func_to_full_params(func_batch_params, full_space_instructions)
    print(f'\nFULL PARAMS : {full_batch_params}')

    # change the params from fractional amounts to transfer volumes for the OT2
    transfer_volumes = compute_transfer_volumes(campaign_config, full_batch_params)
    print('\nTRANSFER VOLUMES : ', transfer_volumes)

    # TODO: convert OT2 script to the transfer volumes
