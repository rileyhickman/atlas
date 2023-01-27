#!/usr/bin/env python

import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterVector

from atlas import Logger
from atlas.optimizers.gp.planner import BoTorchPlanner
from atlas.ot2.volume_converter import *
from atlas.sheets.sheet_manager import SheetManager

from ot2_control.protocols.manager_dummy import ProtocolManager


class Orchestrator:
    """High-level orchestrator for drug formulation experiment"""

    def __init__(self, config_file: str):
        # load config file
        Logger.log_welcome()
        if os.path.isfile(config_file):
            content = open(f"{config_file}", "r")
            self.campaign_config = yaml.full_load(content)
        else:
            Logger.log(f"File {config_file} does not exist", "FATAL")

        # compute molar target concentrations
        self.campaign_config = compute_target_conc(self.campaign_config)

        self.BATCH_SIZE = self.campaign_config["general"]["batch_size"]
        self.BUDGET = self.campaign_config["general"]["budget"]
        self.TIMEOUT = self.campaign_config["general"]["timeout"]

        # get Olympus parameter space and value space
        Logger.log_chapter(
            "Initializing parameter space and campaign",
            line="=",
            style="bold #d9ed92",
        )
        (
            self.func_param_space,
            self.full_param_space,
            self.full_space_instructions,
        ) = self.get_param_space(self.campaign_config)
        self.value_space = self.get_value_space(self.campaign_config)
        # set campaign
        self.set_campaign()
        Logger.log_config(self.full_campaign, self.campaign_config)

        Logger.log_chapter(
            "Initializing experiment planner", line="=", style="bold #d9ed92"
        )
        # set planner
        self.set_planner()

        Logger.log_chapter(
            "Initializing Google sheet manager", line="=", style="bold #d9ed92"
        )
        # setup Google sheet manager
        self.instantiate_sheet_manager()



    def func_to_full_params(
        self,
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


    def get_param_space(
        self, campaign_config: Dict[Any, Any],
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


    def get_value_space(self, campaign_config: Dict[Any, Any]) -> ParameterSpace:
        """Generate Olympus ParameterSpace object based on the supplied
        configuration for the objectives
        """
        objectives = campaign_config["objectives"]
        value_space = ParameterSpace()
        for obj in objectives.keys():
            value_space.add(ParameterContinuous(name=objectives[obj]["name"]))
        return value_space


    def set_campaign(self):
        # functional campaign
        self.func_campaign = Campaign()
        self.func_campaign.set_param_space(self.func_param_space)
        self.func_campaign.set_value_space(self.value_space)

        # full campaign
        self.full_campaign = Campaign()
        self.full_campaign.set_param_space(self.full_param_space)
        self.full_campaign.set_value_space(self.value_space)

    def set_planner(self):
        # instantiate planner with functional parameter space
        # check for multiple objectives
        objectives = self.campaign_config["objectives"]
        if len(objectives) > 1:
            setattr(self, "is_moo", True)
            # multiple objectives
            goals = [
                campaign_config["objectives"][f"obj{ix+1}"]["goal"]
                for ix in range(len(objectives))
            ]
            self.planner = BoTorchPlanner(
                goal="minimize",
                batch_size=self.BATCH_SIZE,
                init_design_strategy="random",
                num_init_design=self.BATCH_SIZE,
                feas_strategy="fca",
                feas_param=0.2,
                is_moo=True,
                value_space=self.func_campaign.value_space,
                scalarizer_kind="Hypervolume",
                moo_params={},
                goals=goals,
            )
        elif len(objectives) == 1:
            # single objective
            setattr(self, "is_moo", False)
            self.planner = BoTorchPlanner(
                goal="".join(
                    (
                        self.campaign_config["objectives"]["obj1"]["goal"],
                        "imize",
                    )
                ),
                batch_size=self.BATCH_SIZE,
                init_design_strategy="random",
                num_init_design=self.BATCH_SIZE,
            )
        else:
            Logger.log("You musy provide at least one objective", "FATAL")

        self.planner.set_param_space(self.func_param_space)

    def instantiate_sheet_manager(self):
        """Generates instance of Google sheet API manager"""
        sheet_config = self.campaign_config["sheets"]
        self.sheet_manager = SheetManager(config=sheet_config)

    def instantiate_protocol_manager(
        self, parameters: Dict[str, np.ndarray]
    ) -> ProtocolManager:
        """Generates protocol manager instance"""
        protocol_config = self.campaign_config["protocol"]
        protocol_manager = ProtocolManager(
            protocol_name=protocol_config["name"],
            protocol_parameters=parameters,
        )
        return protocol_manager

    def check_ot2_server_connection(self):
        """Verify connection to OT2 robot server"""
        return None

    # def copy_to_ot2(self):
    # 	''' Copy the file to the OT2
    # 	'''
    # 	# TODO: implement this
    # 	# https://support.opentrons.com/s/article/Connecting-to-your-OT-2-with-SSH#:~:text=In%20the%20Opentrons%20App%2C%20find,be%20made%20over%20Wi%2DFi.
    # 	# https://support.opentrons.com/s/article/Setting-up-SSH-access-to-your-OT-2

    # 	filename = f'__OT2_file_{self.protocol_name}.py'

    # 	# scp the file to the open

    # 	return None

    def orchestrate(self):
        """Execute the experiment for a given budget of measurements or until a cumulative timeout
        threshold is achieved
        """
        if not self.TIMEOUT:
            Logger.log(
                f"Timeout not set. Experiment will proceed indefinitley for {self.BUDGET} measurements",
                "WARNING",
            )
        elif isinstance(self.TIMEOUT, int):
            Logger.log(
                f"Experiment will proceed for {self.BUDGET} measurements or {self.TIMEOUT} seconds",
                "INFO",
            )
        else:
            Logger.log(
                f"Timeout type not understood. Must provide integer.", "FATAL"
            )

        # begin experiment
        Logger.log_chapter(
            "Commencing experiment", line="=", style="bold #d9ed92"
        )
        num_batches = 0
        while len(self.func_campaign.observations.get_values()) < self.BUDGET:

            # print('CAMPAIGN OBS : ', self.full_campaign.observations.get_params())
            vals = self.full_campaign.observations.get_values()
            # print([np.3isnan(v[0]) for v in vals])
            # print(
            #     "CAMPAIGN OBS : ", self.full_campaign.observations.get_values()
            # )

            num_obs = len(self.func_campaign.observations.get_values())

            Logger.log_chapter(
                f"STARTING BATCH NUMBER {num_batches+1} ({num_obs}/{self.BUDGET} OBSERVATIONS FOUND)",
                line="-",
                style="cyan",
            )

            if self.is_moo:
                obs_for_planner = self.func_campaign.scalarized_observations
            else:
                obs_for_planner = self.func_campaign.observations

            func_batch_params = self.planner.recommend(obs_for_planner)

            # convert the samples to the "full" parameter space
            full_batch_params = self.func_to_full_params(
                func_batch_params,
                self.full_param_space,
                self.full_space_instructions,
            )

            # write new params to sheet
            df = self.sheet_manager.df_from_campaign(
                self.full_campaign, full_batch_params
            )

            self.sheet_manager.write_sheet(df)

            # convert params to transfer volumes for the OT2
            transfer_volumes = compute_transfer_volumes(
                self.campaign_config, full_batch_params
            )

            # TODO: convert the transfer volumes to be recognizable by the protocol (do in volume converter)

            # TODO: generate protocol manager, make OT2 run file, send to OT2 and execute it
            protocol_manager = self.instantiate_protocol_manager(
                parameters=transfer_volumes
            )
            protocol_manager.spawn_protocol_file()

            # TODO: send to OT2

            # execute the protocol
            protocol_manager.execute_protocol(
                simulation=self.campaign_config["protocol"]["simulation"]
            )

            # monitor the google sheet for results
            self.sheet_manager.monitor_sheet()

            # read in the results
            df = self.sheet_manager.read_sheet()

            # add the newest results to the Olympus campaigns
            for sample_ix, (func_sample, full_sample) in enumerate(
                zip(func_batch_params, full_batch_params)
            ):
                value_names = [v.name for v in self.value_space]
                row = df.iloc[-self.BATCH_SIZE + sample_ix, :][
                    value_names
                ].to_dict()
                measurement = ParameterVector().from_dict(row)

                self.func_campaign.add_observation(func_sample, measurement)
                self.full_campaign.add_observation(full_sample, measurement)

            num_batches += 1


if __name__ == "__main__":

    runner = Orchestrator(config_file="campaign_config.yaml")

    runner.orchestrate()
