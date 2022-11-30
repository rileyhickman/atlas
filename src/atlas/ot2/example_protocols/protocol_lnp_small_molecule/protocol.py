#!/usr/bin/env python

from opentrons import protocol_api

from atlas import Logger
from atlas.ot2.example_protocols.abstract_protocol import AbstractProtocol


class LnpSmallMolecule(AbstractProtocol):
    """Basic LNP formulation of small molecules

    parameters are the quantities of stock solutions to be added for the organic stock per well
    batch size is 32 (triplicate measurements on a 96 well plate)
    parameters required: drug, cholesterol, pc, peglipid ethanol
    """

    def __init__(self, parameters={}):
        self.params = parameters()

        self.num_formulations = 32  # batch size
        self.max_pv = 200  # uL ethanol/methanol

        self.letters = ["A", "B", "C", "D", "E", "F", "G", "H"]  # rows
        self.numbers = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
        ]  # columns

    def _prep_pipettes(self):
        # set up pipette tips (this should be done before setting up pipettes)
        p300t_1 = self.protocol.load_labware(
            "opentrons_96_tiprack_300ul", "10"
        )
        p300t_2 = self.protocol.load_labware(
            "opentrons_96_tiprack_300ul", "11"
        )
        p300t_3 = self.protocol.load_labware("opentrons_96_tiprack_300ul", "9")

        # set up pipettes
        self.p300 = self.protocol.load_instrument(
            "p300_single_gen2", "right", tip_racks=[p300t_1, p300t_2, p300t_3]
        )
        self.p300_8 = protocol.load_instrument(
            "p300_multi_gen2", "left", tip_racks=[p300t_1, p300t_2, p300t_3]
        )

        # set up pipette aspirating and dispensing flow rate
        self.p300.flow_rate.aspirate = 300
        self.p300.flow_rate.dispense = 300
        self.p300_8.flow_rate.aspirate = 300
        self.p300_8.flow_rate.dispense = 300

    def _set_labware(self):

        # set up Labwares
        # A1 drug A2 cholestrol; A3 pc; A4 peglipid; ;
        self.scinVial = self.protocol.load_labware(
            "allenlab_8_wellplate_20000ul", "1"
        )
        self.drug_stock = self.scinVial["A1"]
        self.cholestrol_stock = self.scinVial["A2"]
        self.pc_stock = self.scinVial["A3"]
        self.peglipid_stock = self.scinVial["A4"]

        # A1 ethanol
        self.res1 = self.protocol.load_labware("nest_1_reservoir_195ml", "5")
        self.ethanol_stock = self.res1["A1"]

        # A1 aqueous solution for hardening
        self.res2 = self.protocol.load_labware("nest_1_reservoir_195ml", "6")
        self.harden_stock = self.res2["A1"]

        # aqueous phase in column 11 and 12 (~2mL/well)
        self.organic = self.protocol.load_labware(
            "allenlabresevoir_96_wellplate_2200ul", "4"
        )
        self.LNP = self.protocol.load_labware(
            "corning_96_wellplate_360ul_flat", "2"
        )
        self.harden_plate = self.protocol.load_labware(
            "allenlabresevoir_96_wellplate_2200ul", "3"
        )

    def _organic_transfer(self, pipette, source, volume, to, max_pv):
        if volume > self.max_pv:
            x = 2
            v = volume
            while v > self.max_pv:
                v = volume / x
                x += 1
            for i in range(x - 1):
                pipette.transfer(v, source, to, new_tip="never")
        else:
            pipette.transfer(volume, source, to, new_tip="never")

    def _matrix(self, pipette, stock, destination, volume, rows, columns):
        pipette.pick_up_tip()
        i = 0
        for y in range(columns):
            for x in range(rows):
                character = letters[x]
                num = numbers[y]
                if volume[i] > 0.0:
                    self._organic_transfer(
                        pipette,
                        stock,
                        volume[i],
                        destination[character + num],
                        self.max_pv,
                    )
                else:
                    # requested volume is 0.
                    pass
                i += 1
        pipette.drop_tip()

    def _aqueous_phase(self):
        """Take water from large resevoir and disperse to 96 well plate, each well
        with 250 uL water
        """
        # position of pipette aspirating and dispensing position (distance (mm) above the labware bottom)
        self.p300_8.well_bottom_clearance.dispense = 15
        self.p300_8.well_bottom_clearance.aspirate = 2

        self.p300_8.pick_up_tip()
        for col in range(0, 12):  # iterate over columns 8 rows, 12 columns
            # harden_stock is reservoir of water
            p300_8.transfer(
                250,
                self.harden_stock,
                self.LNP.rows()[0][col],
                blow_out=True,
                blowout_location="destination well",
                new_tip="never",
            )
        self.p300_8.drop_tip()

    def _organic_phase(self):
        # position of pipette aspirating and dispensing position (distance (mm) above the labware bottom)
        self.p300.well_bottom_clearance.dispense = 45
        self.p300.well_bottom_clearance.aspirate = 2

        # drug
        self._matrix(
            pipette=self.p300,
            stock=self.drug_stock,
            destination=self.organic,
            volume=self.params["drug"],
            rows=8,
            columns=self.num_formulations // 8,
        )

        # cholesterol
        self._matrix(
            pipette=self.p300,
            stock=self.cholesterol_stock,
            destination=self.organic,
            volume=self.params["cholesterol"],
            rows=8,
            columns=self.num_formulations // 8,
        )

        # pc
        self._matrix(
            pipette=self.p300,
            stock=self.pc_stock,
            destination=self.organic,
            volume=self.params["pc"],
            rows=8,
            columns=self.num_formulations // 8,
        )

        # peglipid
        self._matrix(
            pipette=self.p300,
            stock=self.peglipid_stock,
            destination=self.organic,
            volume=self.params["peglipid"],
            rows=8,
            columns=self.num_formulations // 8,
        )

        # ethanol
        self._matrix(
            pipette=self.p300,
            stock=self.ethanol_stock,
            destination=self.organic,
            volume=self.params["ethanol"],
            rows=8,
            columns=self.num_formulations // 8,
        )

    def _mix(self):
        """mix the organic phase"""
        self.p300_8.well_bottom_clearance.dispense = 2
        self.p300_8.well_bottom_clearance.aspirate = 2

        # mix them for 10 cycles and mix 150 uL for each cycle
        for i in range(self.num_formulations // 8):
            self.p300_8.pick_up_tip()
            self.p300_8.mix(10, self.max_pv, self.organic.rows()[0][i])
            self.p300_8.drop_tip()

    def _formulation(self):
        """inject 50uL of organic phase into 250uL of aqueous phase and mix them"""

        # Position of pipette aspirating and dispensing position (distance (mm) above the labware bottom)
        self.p300_8.well_bottom_clearance.dispense = 2
        self.p300_8.well_bottom_clearance.aspirate = 2

        # Inject organic phase (50 uL) to 96 well plates (12 columns) for formulation
        # during same protcol always use the same vol organic phase

        for i in range(self.num_formulations // 8):
            for j in range(96 // self.num_formulations):
                self.p300_8.pick_up_tip()
                self.p300_8.transfer(
                    50,
                    self.organic.rows()[0][i],
                    self.LNP.rows()[0][i * 96 // self.num_formulations + j],
                    mix_after=(30, 200),
                    new_tip="never",
                )
                self.p300_8.drop_tip()

    def _hardening(self):
        # 10x dilution with water --> has to do with purification
        # Position of pipette aspirating and dispensing position (distance (mm) above the labware bottom)
        self.p300_8.well_bottom_clearance.dispense = 15
        self.p300_8.well_bottom_clearance.aspirate = 2

        self.p300_8.pick_up_tip()
        # Add DI water to 96 filter well plates (12 columns)
        for i in range(0, 12):
            self.p300_8.transfer(
                900,
                self.harden_stock,
                self.harden_plate.rows()[0][i],
                blow_out=True,
                blowout_location="destination well",
                new_tip="never",
            )
        self.p300_8.drop_tip()

        self.p300_8.well_bottom_clearance.dispense = 5
        self.p300_8.well_bottom_clearance.aspirate = 2

        # Inject particles (100 uL) to 96 deep well plates (12 columns) with DI water for hardening
        for i in range(12):
            self.p300_8.pick_up_tip()
            self.p300_8.transfer(
                100,
                self.LNP.rows()[0][i],
                self.harden_plate.rows()[0][i],
                new_tip="never",
            )
            self.p300_8.mix(30, 250, self.harden_plate.rows()[0][i])
            self.p300_8.drop_tip()

    def run(self, protocol: protocol_api.ProtocolContext):

        if not hasattr(self, "protocol"):
            setattr(self, "protocol", protocol)

        self._prep_pipettes()
        self._set_labware()

        self._aqueous_phase()  # prep aqueous phase
        self._organic_phase()  # prep organic phase
        self._mix()  # mix organic phase
        self._formulation()  # injection of organic phase into aqueous phase
        self._hardening()  # dilute particles with water to shrink to real size
