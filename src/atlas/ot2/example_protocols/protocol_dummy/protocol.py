#!/usr/bin/env python

from opentrons import protocol_api

from atlas.ot2.example_protocols.abstract_protocol import AbstractProtocol


class Dummy(AbstractProtocol):
    """Dummy OT2 protocol for debugging"""

    def __init__(self, parameters={}):
        self.parameters = parameters

    def run(self, protocol: protocol_api.ProtocolContext):
        tips = protocol.load_labware("opentrons_96_tiprack_300ul", 1)
        reservoir = protocol.load_labware("nest_12_reservoir_15ml", 2)
        plate = protocol.load_labware("nest_96_wellplate_200ul_flat", 3)

        print("parameter :", self.parameters)
