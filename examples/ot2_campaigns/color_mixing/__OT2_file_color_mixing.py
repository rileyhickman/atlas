#!/usr/bin/env python
from opentrons import protocol_api

metadata = {"protocolName": "color_mixing", "dateModified": "2/12/2022", "author": "Soteria Therapeutics", "apiLevel": "2.10"}

protocol_parameters = {"red": [295.0389725903844], "yellow": [175.92478500580714], "blue": [1.7395467317157989], "green": [127.29669567209268], "iteration": 8}

def run(protocol: protocol_api.ProtocolContext):
	protocol_module_instance = ColorMixing(parameters=protocol_parameters)
	protocol_module_instance.run(protocol)
