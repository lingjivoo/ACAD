"""Functional interface"""

import warnings
import math
import torch
from .v_dropout import *

def dropout(input, prob,training=False, inplace=False):  #v_drop
    return Dropout.apply(input, prob,training, inplace)
