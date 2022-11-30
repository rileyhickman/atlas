#!/usr/bin/env python

import glob
import os
import sys

__home__ = os.path.dirname(os.path.abspath(__file__))


class ProtocolLoader:
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._find_protocols()

    def __getattr__(self, attr):
        if attr in ["Protocol", "AbstractProtocol", "CustomProtocol"]:
            attr_file = ProtocolLoader.class_to_file(attr)
            module = __import__(
                f"atlas.ot2.example_protocols.{attr_file}", fromlist=[attr]
            )
            _class = getattr(module, attr)
            setattr(self, attr, _class)
            return _class
        else:
            protocol = ProtocolLoader.import_protocol(attr)
            setattr(self, attr, protocol)
            return protocol

    @staticmethod
    def class_to_file(class_name):
        file_name = class_name[0].lower()
        for character in class_name[1:]:
            if character.isupper():
                file_name += f"_{character.lower()}"
            else:
                file_name += character
        return file_name

    @staticmethod
    def file_to_class(file_name):
        class_name = file_name[0].upper()
        next_upper = False
        for character in file_name[1:]:
            if character == "_":
                next_upper = True
                continue
            if next_upper:
                class_name += character.upper()
            else:
                class_name += character
            next_upper = False
        return class_name

    @staticmethod
    def import_protocol(attr):
        attr_file = ProtocolLoader.class_to_file(attr)
        module = __import__(
            f"atlas.ot2.example_protocols.protocol_{attr_file}",
            fromlist=[attr],
        )
        _class = getattr(module, attr)
        return _class

    def _find_protocols(self):
        self.protocol_files = []
        self.protocol_names = []
        self.protocols_map = {}
        for dir_name in glob.glob(f"{__home__}/protocol_*"):

            if "/" in dir_name:
                protocol_name = dir_name.split("/")[-1][9:]
            elif "\\" in dir_name:
                protocol_name = dir_name.split("/")[-1][9:]

            self.protocol_files.append(protocol_name)
            self.protocol_names.append(
                ProtocolLoader.file_to_class(protocol_name)
            )

    def get_protocols_list(self):
        return sorted(self.protocol_names)


sys.modules[__name__] = ProtocolLoader(**locals())
