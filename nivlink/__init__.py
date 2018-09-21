"""Niv Lab software for preprocessing eyelink eyetracking data."""

__version__ = '0.2'

from .preprocessing import (align_to_aoi, compute_fixations)
from .screen import (ScreenInfo)
from .fht import epoching_fht
