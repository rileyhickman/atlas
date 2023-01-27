#!/usr/bin/env python

import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime


class Monitor:
    def __init__(
        self, pickup_dir, protocol_name, monitor_interval=2.0, timeout=3600.0
    ):
        self.pickup_dir = pickup_dir
        self.protocol_name = protocol_name
        self.monitor_interval = monitor_interval
        self.timeout = timeout

        self.date = datetime.now()

        self.metadata_dict = {
            "protocolName": self.protocol_name,
            "dateModified": f"{self.date.day}/{self.date.month}/{self.date.year}",
            "author": "Soteria Therapeutics",
            "apiLevel": "2.10",
        }

    def monitor(self):
        has_parameters = False
        start_time = time.time()
        while not has_parameters:
            if os.path.exists(f"{self.pickup_dir}params.pkl"):
                params = pickle.load(
                    open(f"{self.pickup_dir}params.pkl", "rb")
                )
                has_parameters = True
            else:
                time.sleep(self.monitor_interval)
                # check timeout
                if time.time() - start_time >= self.timeout:
                    print("total experiment timeout reached!")
                    quit()

        # make sure params are of the right type (json serializable)
        protocol_parameters = {}
        for key, val in params.items():
            if not isinstance(val, (float, int)):
                protocol_parameters[key] = list(val)
            else:
                protocol_parameters[key] = val

        return protocol_parameters

    def execute(self, protocol_parameters):
        self.write_protocol_file(protocol_parameters)

        # join files
        os.system(
            f"cat __OT2_file_{self.protocol_name}.py abstract_protocol.txt protocol.txt >> test.py"
        )

        # execute the protocol on the OT-2
        #os.system("opentrons_execute test.py")
        os.system("opentrons_simulate test.py")

        # once the protocol has finished, clean up the directories
        os.system(f"rm __OT2_file_{self.protocol_name}.py test.py {self.pickup_dir}params.pkl")



    def write_protocol_file(self, protocol_parameters):
        s = "#!/usr/bin/env python\n"
        s += "from opentrons import protocol_api\n\n"

        s += "metadata = " + json.dumps(self.metadata_dict) + "\n\n"

        # write dictionary of parameters for protocol instance
        s += (
            "protocol_parameters = " + json.dumps(protocol_parameters) + "\n\n"
        )

        s += "def run(protocol: protocol_api.ProtocolContext):\n"
        s += f"\tprotocol_module_instance = ColorMixing(parameters=protocol_parameters)\n"
        s += f"\tprotocol_module_instance.run(protocol)\n"

        with open(f"__OT2_file_{self.protocol_name}.py", "w") as f:
            f.write(s)


if __name__ == "__main__":

    monitor = Monitor(
        pickup_dir="test_dir/",
        protocol_name="dummy",
        monitor_interval=2.0,
        timeout=3600.0,
    )

    while True:
        params = monitor.monitor()
        print('PARAMS : ', params)
        monitor.execute(params)
