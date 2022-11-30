#!/usr/bin/env python

from opentrons import protocol_api

metadata = {
    "protocolName": "dummy",
    "dateModified": "28/11/2022",
    "author": "Soteria Therapeutics",
    "apiLevel": "2.10",
}


def run(protocol: protocol_api.ProtocolContext):

    p300t_1 = protocol.load_labware("opentrons_96_tiprack_300ul", "8")
    p300t_2 = protocol.load_labware("opentrons_96_tiprack_300ul", "1")

    # Set up Pipettes
    p300 = protocol.load_instrument(
        "p300_single_gen2",
        "right",
        tip_racks=[p300t_1, p300t_2],
    )
    # p300_8 = protocol.load_instrument('p300_multi_gen2', 'left', tip_racks=[p300t_1])

    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for row_ix in letters:

        # pick up tip with p300
        p300.pick_up_tip(p300t_1[f"{row_ix}12"])

        p300.drop_tip(p300t_2[f"{row_ix}1"])

    p300.home()
