#!/usr/bin/env python

import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterVector,
)
from olympus.utils.data_transformer import cube_to_simpl
from ot2_control.protocols.manager import ProtocolManager

from atlas import Logger
from atlas.optimizers.gp.planner import BoTorchPlanner
from atlas.ot2.cv import Camera
from atlas.ot2.remote_host import Host


class Orchestrator:
    """High-level orchestrator for the color optimization
    experiment with the OT-2 robot
    """

    def __init__(self, config_file: str):
        Logger.log_welcome()
        if os.path.isfile(config_file):
            content = open(f"{config_file}", "r")
            self.config = yaml.full_load(content)
        else:
            Logger.log(f"File {config_file} does not exist", "FATAL")

        # generate directory for the campaign
        now = datetime.now()
        camp_name = self.config["protocol"]["name"]
        self.campaign_dir = (
            f"camp_{camp_name}_{now.year}-{now.month}-{now.day}/"
        )
        os.makedirs(self.campaign_dir, exist_ok=False)

        self.BATCH_SIZE = self.config["general"]["batch_size"]
        self.BUDGET = self.config["general"]["budget"]
        self.TIMEOUT = self.config["general"]["timeout"]
        self.TOTAL_VOLUME = self.config["general"]["total_volume"]

        Logger.log_chapter(
            "Initializing parameter space and campaign",
            line="=",
            style="bold #d9ed92",
        )

        self.func_param_space, self.full_param_space = self.get_param_space(
            self.config
        )
        self.value_space = self.get_value_space(self.config)

        self.set_campaign()

        self.set_planner()

        Logger.log_chapter(
            "Initializing SSH connection to OT-2 robot",
            line="=",
            style="bold #d9ed92",
        )

        self.host = Host(
            hostname="192.168.0.168",
            username="root",
            key_filename="/Users/rileyhickman/.ssh/id_rsa",
        )
        self.dest_path = self.config["protocol"]["dest_path"]

    def set_campaign(self):
        # functional campaign
        self.func_campaign = Campaign()
        self.func_campaign.set_param_space(self.func_param_space)
        self.func_campaign.set_value_space(self.value_space)

        # full campaign
        self.full_campaign = Campaign()
        self.full_campaign.set_param_space(self.full_param_space)
        self.full_campaign.set_value_space(self.value_space)

    def get_param_space(self, config):
        """Generate Olympus param space"""
        func_param_space = ParameterSpace()
        full_param_space = ParameterSpace()
        params = config["params"]
        param_names = [k for k in params.keys()]

        for component in param_names:
            if isinstance(params[component]["low"], float):
                # we have range, add to functional params
                param_to_add = ParameterContinuous(
                    name=component,
                    low=params[component]["low"],
                    high=params[component]["high"],
                )
                func_param_space.add(param_to_add)
                full_param_space.add(param_to_add)
            else:
                param_to_add = ParameterContinuous(
                    name=component,
                    low=0.0,
                    high=1.0,
                )
                full_param_space.add(param_to_add)
        return func_param_space, full_param_space

    def get_value_space(self, config):
        objectives = config["objs"]
        value_space = ParameterSpace()
        for obj in objectives.keys():
            value_space.add(ParameterContinuous(name=objectives[obj]["name"]))
        return value_space

    def compute_volumes(self, func_params):
        # func_params are just red, yellow, blue
        func_params_arr = func_params[0].to_array() 

        full_params_arr = np.array([func_params_arr / np.sum(func_params_arr)])

        full_params = [
            ParameterVector().from_array(i, param_space=self.full_param_space)
            for i in full_params_arr
        ]

        v = full_params_arr * self.TOTAL_VOLUME
        volumes = {}
        for ix, param in enumerate(self.full_param_space):
            volumes[param.name] = v[:, ix]

        return volumes, full_params

    def set_planner(self):
        self.planner = BoTorchPlanner(
            goal=self.config["objs"]["loss"]["goal"],
            batch_size=self.BATCH_SIZE,
            init_design_strategy="random",
            num_init_design=4,
        )
        self.planner.set_param_space(self.func_param_space)

    def instantiate_protocol_manager(self, parameters):

        protocol_config = self.config["protocol"]
        protocol_manager = ProtocolManager(
            protocol_name=protocol_config["name"],
            protocol_parameters=parameters,
        )
        return protocol_manager

    def execute_protocol(self):
        """Run the protocol on the OT2 server with optional preceeding simulation"""
        protocol_name = self.config["protocol"]["name"]
        filename = f"__OT2_file_{protocol_name}.py"
        if simulation:
            # execute simulation to assure there are no bugs in the parameterized OT-2 protocol
            msg = f'Simulation of OT2 protocol "{self.protocol_name}" requested. Executing simulation with `opentrons_simulate {filename}`'
            Logger.log(msg, "WARNING")
            result = subprocess.run(
                f"opentrons_simulate {filename}",
                capture_output=True,
                shell=True,
            )
            # TODO: parse the results of the simulation (# results.stderr)
            if True:
                Logger.log(
                    f'Simualtion of OT2 protocol "{protocol_name}" finished successfully',
                    "INFO",
                )
            else:
                Logger.log(
                    f'Simulation of OT2 protocol "{protocol_name}" failed!',
                    "FATAL",
                )

        # copy the file to the OT-2 and execute it there
        self.host.put_file(filename, dest_path=self.dest_path)

        time.sleep(0.5)
        # TODO: update this for production
        # self.host.run_command(command=f'opentrons_execute {dest_path}{filename}')
        stdin, stdout, stderr = self.host.run_command(
            command=f"opentrons_simulate {self.dest_path}{filename}",
            return_info=True,
        )

    def orchestrate(self):

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
        iteration = 1
        num_batches = 0
        while len(self.func_campaign.observations.get_values()) < self.BUDGET:

            num_obs = len(self.func_campaign.observations.get_values())

            Logger.log_chapter(
                f"STARTING BATCH NUMBER {num_batches+1} ({num_obs}/{self.BUDGET} OBSERVATIONS FOUND)",
                line="-",
                style="cyan",
            )

            # make directory to store the results of the iteration
            iter_dir = f"{self.campaign_dir}iter_{iteration}/"
            os.makedirs(iter_dir, exist_ok=False)

            # instantiate webcam object powered by opencv
            self.camera = Camera(
                grid_dims=(
                    self.config["general"]["grid_dims"]["rows"],
                    self.config["general"]["grid_dims"]["cols"],
                ),
                save_img_path=iter_dir,
                hough_config={},
            )

            func_batch_params = self.planner.recommend(
                self.func_campaign.observations
            )

            transfer_volumes, full_batch_params = self.compute_volumes(
                func_batch_params
            )

            
            print('\nPROPOSED FUNC PARAMS : ', func_batch_params)

            print("\nPROPOSED PARAMS : ", full_batch_params)

            print("\nTRANSFER VOLUMES : ", transfer_volumes)
            print("\n")

            # add iteration number and target hexcode to parameters
            transfer_volumes["iteration"] = iteration

            # protocol_manager = self.instantiate_protocol_manager(
            #     parameters=transfer_volumes
            # )
            # protocol_manager.spawn_protocol_file()

            # write the parameters to a pickle file
            pickle.dump(transfer_volumes, open('params.pkl', 'wb'))

            filename = "__OT2_file_color_mixing.py"

            _ = self.host.put_file('params.pkl', dest_path=self.dest_path+'pickup_dir/')

            time.sleep(0.5)

            # execute the color mixing i.e. sample prep + measurement of loss with the
            # webcam and opencv
        
            input("Press Enter once OT-2 protocol is complete...")

            print("Removing files...")

            os.system('rm params.pkl')

            # take a picture with the webcam and measure the loss
            # and save image
            loss, avg_meas_rgb = self.camera.make_measurement(
                iteration=iteration,
                target_rgb=Camera.hex_to_rgb(
                    self.config["general"]["target_hexcode"]
                ),
                save_img=True,
            )

            # loss = 100
            # avg_meas_rgb = [111, 200, 233]

            # add the results the of the experiment to the parameter directory
            # and save to disk
            transfer_volumes["loss"] = loss
            transfer_volumes["avg_meas_rgb"] = avg_meas_rgb
            pickle.dump(
                transfer_volumes, open(f"{iter_dir}parameters.pkl", "wb")
            )

            print("\nLOSS : ", loss)
            print("\n")

            # add to Olympus campaigns
            self.func_campaign.add_observation(func_batch_params[0], loss)
            self.full_campaign.add_observation(full_batch_params[0], loss)

            # save to disk
            pickle.dump(
                [self.func_campaign, self.full_campaign],
                open(self.campaign_dir + "results.pkl", "wb"),
            )

            iteration += 1
            num_batches += 1


if __name__ == "__main__":

    runner = Orchestrator(config_file="config.yaml")

    runner.orchestrate()
