"""Niv Lab software for preprocessing eyelink eyetracking data."""

__version__ = '0.2'

from .screen import (ScreenInfo)
from .raw import Raw
from .preprocessing import (align_to_aoi, compute_fixations)
from .fht import epoching_fht
