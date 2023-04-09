#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from deap import base, creator, tools
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace
from rich.progress import track

from atlas import Logger
from atlas.optimizers.acqfs import create_available_options
from atlas.optimizers.acquisition_optimizers.base_optimizer import (
    AcquisitionOptimizer,
)
from atlas.optimizers.params import Parameters
from atlas.optimizers.utils import (
    cat_param_to_feat,
    forward_normalize,
    forward_standardize,
    get_cat_dims,
    get_fixed_features_list,
    infer_problem_type,
    param_vector_to_dict,
    propose_randomly,
    reverse_normalize,
    reverse_standardize,
)


class GeneticGeneralOptimizer(AcquisitionOptimizer):
    def __init__(
        self,
    ):
        pass
