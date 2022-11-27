#!/usr/bin/env python
from opentrons import protocol_api
from atlas.ot2.example_protocols.manager import ProtocolManager

metadata = {"protocolName": "dummy", "dateModified": "27/11/2022", "author": "Soteria Therapeutics", "apiLevel": "2.10"}

protocol_parameters = {"paclitaxel": [70.0, 102.0], "cholesterol": [82.0, 186.0], "dspc": [247.0, 29.0], "dmg-peg_2000": [30.0, 60.0], "ethanol": [371.0, 423.0]}

def run(protocol: protocol_api.ProtocolContext):
	manager = ProtocolManager(protocol_name="dummy", protocol_parameters=protocol_parameters)
	protocol_module_instance = manager.protocol_module(parameters=protocol_parameters)
	protocol_module_instance.run(protocol)