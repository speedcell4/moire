from typing import Callable

import dynet as dy

Expression = dy.Expression
Activation = Callable[[Expression], Expression]

Parameters = dy.Parameters
LookupParameters = dy.LookupParameters
ParameterCollection = dy.ParameterCollection

from .array import *
from .configuration import config
