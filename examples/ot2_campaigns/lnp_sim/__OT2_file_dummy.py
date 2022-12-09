#!/usr/bin/env python
from opentrons import protocol_api
from atlas.ot2.example_protocols.manager import ProtocolManager

metadata = {"protocolName": "dummy", "dateModified": "5/12/2022", "author": "Soteria Therapeutics", "apiLevel": "2.10"}

protocol_parameters = {"paclitaxel": [24.0, 23.0], "cholesterol": [186.0, 186.0], "dspc": [36.0, 42.0], "dmg-peg_2000": [30.0, 0.0], "ethanol": [524.0, 549.0]}

def run(protocol: protocol_api.ProtocolContext):
	manager = ProtocolManager(protocol_name="dummy", protocol_parameters=protocol_parameters)
	protocol_module_instance = manager.protocol_module(parameters=protocol_parameters)
	protocol_module_instance.run(protocol)