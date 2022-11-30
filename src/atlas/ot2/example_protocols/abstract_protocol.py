#!/usr/bin/env python

from abc import abstractmethod

from atlas import Logger


class AbstractProtocol:
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """run method for the OT2 Python API"""
        pass
