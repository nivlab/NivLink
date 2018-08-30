'''Niv Lab software for preprocessing eyelink eyetracking data.'''

__version__ = '0.1'

from .epoching import epoching_fht
from .preprocessing import (align_to_aoi, compute_fixations)
from .screen import (ScreenInfo)
from .mapping import (map_to_feats)