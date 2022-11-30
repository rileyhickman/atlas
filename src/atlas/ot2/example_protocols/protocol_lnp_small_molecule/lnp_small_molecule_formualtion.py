from opentrons import protocol_api

metadata = {
    "Date last modified": "Nov 8 2022",
    "Author": "Allen Lab",
    "apiLevel": "2.10",
}


def run(protocol: protocol_api.ProtocolContext):

    # Set up pipette tips (this should be done before setting up pipettes)
    p300t_1 = protocol.load_labware("opentrons_96_tiprack_300ul", "10")
    p300t_2 = protocol.load_labware("opentrons_96_tiprack_300ul", "11")
    p300t_3 = protocol.load_labware("opentrons_96_tiprack_300ul", "9")

    # Set up Pipettes
    p300 = protocol.load_instrument(
        "p300_single_gen2", "right", tip_racks=[p300t_1, p300t_2, p300t_3]
    )
    p300_8 = protocol.load_instrument(
        "p300_multi_gen2", "left", tip_racks=[p300t_1, p300t_2, p300t_3]
    )

    # Set up pipette aspirating and dispensing flow rate
    p300.flow_rate.aspirate = 300
    p300.flow_rate.dispense = 300
    p300_8.flow_rate.aspirate = 300
    p300_8.flow_rate.dispense = 300

    # Set up Labwares
    # A1 drug A2 cholestrol; A3 pc; A4 peglipid; ;

    scinVial = protocol.load_labware("allenlab_8_wellplate_20000ul", "1")
    drug_stock = scinVial["A1"]
    cholestrol_stock = scinVial["A2"]
    pc_stock = scinVial["A3"]
    peglipid_stock = scinVial["A4"]

    # A1 ethanol
    res1 = protocol.load_labware("nest_1_reservoir_195ml", "5")
    ethanol_stock = res1["A1"]

    # A1 aqueous solution for hardening
    res2 = protocol.load_labware("nest_1_reservoir_195ml", "6")
    harden_stock = res2["A1"]

    # aqueous phase in column 11 and 12 (~2mL/well)
    organic = protocol.load_labware(
        "allenlabresevoir_96_wellplate_2200ul", "4"
    )
    LNP = protocol.load_labware("corning_96_wellplate_360ul_flat", "2")
    harden_plate = protocol.load_labware(
        "allenlabresevoir_96_wellplate_2200ul", "3"
    )

    # Quantites of stock solution to be added for organic stock per well
    drug = [
        33.5,
        34.9,
        36.3,
        38.1,
        22.8,
        24.1,
        25.5,
        27.4,
        67.1,
        69.8,
        72.6,
        76.2,
        45.5,
        48.3,
        51.0,
        54.7,
        134.1,
        139.6,
        145.1,
        152.5,
        91.1,
        96.6,
        102.1,
        109.4,
        268.3,
        279.3,
        290.3,
        304.9,
        182.2,
        193.2,
        204.2,
        218.9,
    ]
    cholestrol = [
        82.5,
        82.5,
        82.5,
        82.5,
        185.6,
        185.6,
        185.6,
        185.6,
        82.5,
        82.5,
        82.5,
        82.5,
        185.6,
        185.6,
        185.6,
        185.6,
        82.5,
        82.5,
        82.5,
        82.5,
        185.6,
        185.6,
        185.6,
        185.6,
        82.5,
        82.5,
        82.5,
        82.5,
        185.6,
        185.6,
        185.6,
        185.6,
    ]
    pc = [
        252.8,
        246.5,
        240.2,
        231.8,
        42.1,
        35.8,
        29.5,
        21.1,
        252.8,
        246.5,
        240.2,
        231.8,
        42.1,
        35.8,
        29.5,
        21.1,
        252.8,
        246.5,
        240.2,
        231.8,
        42.1,
        35.8,
        29.5,
        21.1,
        252.8,
        246.5,
        240.2,
        231.8,
        42.1,
        35.8,
        29.5,
        21.1,
    ]
    peglipid = [
        0.0,
        30.1,
        60.2,
        100.4,
        0.0,
        30.1,
        60.2,
        100.4,
        0.0,
        30.1,
        60.2,
        100.4,
        0.0,
        30.1,
        60.2,
        100.4,
        0.0,
        30.1,
        60.2,
        100.4,
        0.0,
        30.1,
        60.2,
        100.4,
        0.0,
        30.1,
        60.2,
        100.4,
        0.0,
        30.1,
        60.2,
        100.4,
    ]
    ethanol = [
        431.1,
        406.0,
        380.8,
        347.3,
        549.5,
        524.3,
        499.2,
        465.6,
        397.6,
        371.1,
        344.5,
        309.1,
        526.7,
        500.2,
        473.6,
        438.3,
        330.5,
        301.2,
        272.0,
        232.9,
        481.2,
        451.9,
        422.6,
        383.5,
        196.4,
        161.6,
        126.8,
        80.4,
        390.1,
        355.3,
        320.5,
        274.1,
    ]

    # number of unique formulations on a 96-well plate
    formulations = 32

    # Helper function, organic solution transfer has lower max volume taken into pipette to prevent damage
    # max_pv = 200 uL for ethanol/methanol

    max_pv = 200

    def organic_transfer(pipette, source, volume, to, max_pv):

        if volume > max_pv:
            x = 2
            v = volume
            while v > max_pv:
                v = volume / x
                x += 1
            for i in range(x - 1):
                pipette.transfer(v, source, to, new_tip="never")
        else:
            pipette.transfer(volume, source, to, new_tip="never")

    # Matrix of 96 well plate or receiver
    def matrix(pipette, stock, destination, volume, rows, columns):
        """add in volumes as specified in list"""
        pipette.pick_up_tip()
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        numbers = [
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
        ]
        i = 0
        for y in range(columns):
            for x in range(rows):
                character = letters[x]
                num = numbers[y]
                if volume[i] > 0:
                    organic_transfer(
                        pipette,
                        stock,
                        volume[i],
                        destination[character + num],
                        max_pv,
                    )
                i = i + 1
        pipette.drop_tip()

    # Add 250 uL of aqueous phase to 96-well plates
    def aqueous_phase():
        """take water from large resevoir and disperse to 96 well plate, each well 250 uL of water"""

        # Position of pipette aspirating and dispensing position (distance (mm) above the labware bottom)
        p300_8.well_bottom_clearance.dispense = 15
        p300_8.well_bottom_clearance.aspirate = 2

        p300_8.pick_up_tip()
        for i in range(0, 12):  # iterate over columns 8 rows, 12 columns
            # harden_stock is reservoir of water
            p300_8.transfer(
                250,
                harden_stock,
                LNP.rows()[0][i],
                blow_out=True,
                blowout_location="destination well",
                new_tip="never",
            )
        p300_8.drop_tip()

    # Prepare the organic phase
    def organic_phase():

        # Position of pipette aspirating and dispensing position (distance (mm) above the labware bottom)
        p300.well_bottom_clearance.dispense = 45
        p300.well_bottom_clearance.aspirate = 2

        # p300 is the name of the pipette

        # drug
        matrix(p300, drug_stock, organic, drug, 8, int(formulations / 8))

        # cholestrol
        matrix(
            p300,
            cholestrol_stock,
            organic,
            cholestrol,
            8,
            int(formulations / 8),
        )

        # pc
        matrix(p300, pc_stock, organic, pc, 8, int(formulations / 8))

        # peglipid
        matrix(
            p300, peglipid_stock, organic, peglipid, 8, int(formulations / 8)
        )

        # ethanol
        matrix(p300, ethanol_stock, organic, ethanol, 8, int(formulations / 8))

    # Mix the organic phase
    def mix():

        p300_8.well_bottom_clearance.dispense = 2
        p300_8.well_bottom_clearance.aspirate = 2

        # mix them for 10 cycles and mix 150 uL for each cycle
        for i in range(int(formulations / 8)):
            p300_8.pick_up_tip()
            p300_8.mix(10, max_pv, organic.rows()[0][i])
            p300_8.drop_tip()

    # Formulation: inject 50 uL of organic phase to 250 of aqueous phase and mix them
    def formulation():

        # Position of pipette aspirating and dispensing position (distance (mm) above the labware bottom)
        p300_8.well_bottom_clearance.dispense = 2
        p300_8.well_bottom_clearance.aspirate = 2

        # Inject organic phase (50 uL) to 96 well plates (12 columns) for formulation
        # during same protcol always use the same vol organic phase

        for i in range(0, int(formulations / 8)):
            for j in range(0, int(96 / formulations)):
                p300_8.pick_up_tip()
                p300_8.transfer(
                    50,
                    organic.rows()[0][i],
                    LNP.rows()[0][i * int(96 / formulations) + j],
                    mix_after=(30, 200),
                    new_tip="never",
                )
                p300_8.drop_tip()

    # Harden the particles
    def hardening():
        # 10x dilution with water --> has to do with purification

        # Position of pipette aspirating and dispensing position (distance (mm) above the labware bottom)
        p300_8.well_bottom_clearance.dispense = 15
        p300_8.well_bottom_clearance.aspirate = 2

        p300_8.pick_up_tip()
        # Add DI water to 96 filter well plates (12 columns)
        for i in range(0, 12):
            p300_8.transfer(
                900,
                harden_stock,
                harden_plate.rows()[0][i],
                blow_out=True,
                blowout_location="destination well",
                new_tip="never",
            )
        p300_8.drop_tip()

        p300_8.well_bottom_clearance.dispense = 5
        p300_8.well_bottom_clearance.aspirate = 2

        # Inject particles (100 uL) to 96 deep well plates (12 columns) with DI water for hardening
        for i in range(0, 12):
            p300_8.pick_up_tip()
            p300_8.transfer(
                100,
                LNP.rows()[0][i],
                harden_plate.rows()[0][i],
                new_tip="never",
            )
            p300_8.mix(30, 250, harden_plate.rows()[0][i])
            p300_8.drop_tip()

    aqueous_phase()  # prep aqueous phase
    organic_phase()  # prep this
    mix()  # mix organic phase
    formulation()  # injection of organic phase into aqueous phase
    hardening()  # dilution of particles using water to shrink particle to real size
