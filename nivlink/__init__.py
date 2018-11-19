"""Niv Lab software for preprocessing eyelink eyetracking data."""

__version__ = '0.2'

from .raw import (Raw)
from .epochs import (Epochs)
from .gaze import (align_to_aoi, compute_fixations)
from .screen import (Screen)
