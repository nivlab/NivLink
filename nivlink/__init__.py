'''Niv Lab software for preprocessing eyelink eyetracking data.'''

__version__ = '0.1'

from .fht import epoching_fht
from .preprocessing import (align_to_aoi, compute_fixations)
from .screen import (ScreenInfo)
