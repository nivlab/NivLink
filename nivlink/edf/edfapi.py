import sys
import os.path as op
from ctypes import (c_int, Structure, c_char, c_char_p, c_ubyte,
                    c_short, c_ushort, c_uint, c_float, POINTER, CDLL)

"""Wrapper for edfapi.so

This script makes accessible to native python the C functions part of 
the EyeLink EDF Access API. The corresponding documentation for each
function can be found in the EyeLink EDF Users manual or in the
EDF Access API headers (edf.h).
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load EDF API library. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Locate EDF library.
if sys.platform.startswith('linux'):
    fname = op.join('edfapi', 'linux', 'libedfapi.so.masked') 
elif sys.platform.startswith('darwin'):
    fname = op.join('edfapi', 'macosx', 'edfapi') 
else:
    raise OSError('EDF reading currently not supported for Windows.')
                  
## Load EDF library.
fname = op.join( op.dirname(__file__), fname )
edfapi = CDLL(fname)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define data types.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

cf = c_float         # Float array, shape (1,)
cf2 = c_float * 2    # Float array, shape (2,)

class LSTRING(Structure): 
    """String class for storing EyeLink messages."""
    pass
                  
LSTRING.__slots__ = ['len', 'c']
LSTRING._fields_ = [('len', c_short), ('c', c_char)]


class FSAMPLE(Structure): 
    """The FSAMPLE structure holds information for a sample in the EDF file.
    Depending on the recording options set for the recording session, some of the
    fields may be empty."""
    pass

FSAMPLE.__slots__ = [
    'time', 'px', 'py', 'hx', 'hy', 'pa', 'gx', 'gy', 'rx', 'ry', 'gxvel', 'gyvel', 
    'hxvel', 'hyvel', 'rxvel', 'ryvel', 'fgxvel', 'fgyvel', 'fhxvel', 'fhyvel', 
    'frxvel', 'fryvel', 'hdata', 'flags', 'input', 'buttons', 'htype', 'errors']
FSAMPLE._fields_ = [
    ('time', c_uint),       # time stamp of sample 
    ('px', cf2),            # pupil x
    ('py', cf2),            # pupil y
    ('hx', cf2),            # headref x
    ('hy', cf2),            # headref y
    ('pa', cf2),            # pupil size or area 
    ('gx', cf2),            # screen gaze x
    ('gy', cf2),            # screen gaze y
    ('rx', cf),             # screen pixels per degree x
    ('ry', cf),             # screen pixels per degree y
    ('gxvel', cf2),         # gaze x velocity
    ('gyvel', cf2),         # gaze y velocity
    ('hxvel', cf2),         # headref x velocity 
    ('hyvel', cf2),         # headref y velocity 
    ('rxvel', cf2),         # raw x velocity
    ('ryvel', cf2),         # raw y velocity
    ('fgxvel', cf2),        # fast gaze x velocity 
    ('fgyvel', cf2),        # fast gaze y velocity
    ('fhxvel', cf2),        # fast headref x velocity
    ('fhyvel', cf2),        # fast headref y velocity
    ('frxvel', cf2),        # fast raw x velocity
    ('fryvel', cf2),        # fast raw y velocity
    ('hdata', c_short * 8), # head-tracker data (not pre-scaled)
    ('flags', c_ushort),    # flags to indicate contents
    ('input', c_ushort),    # extra (input word)
    ('buttons', c_ushort),  # button state & changes
    ('htype', c_short),     # head-tracker data type
    ('errors', c_ushort)    # process error flags 
]  


class FEVENT(Structure):
    """The FEVENT structure holds information for an event in the EDF file. 
    Depending on the recording options set for the recording session and 
    the event type, some of the fields may be empty."""
    pass

FEVENT.__slots__ = [
    'time', 'type', 'read', 'sttime', 'entime', 'hstx', 'hsty', 'gstx', 'gsty',
    'sta', 'henx', 'heny', 'genx', 'geny', 'ena', 'havx', 'havy', 'gavx', 'gavy', 
    'ava', 'avel', 'pvel', 'svel', 'evel', 'supd_x', 'eupd_x', 'supd_y', 'eupd_y', 
    'eye', 'status', 'flags', 'input', 'buttons', 'parsedby', 'message']
FEVENT._fields_ = [
    ('time', c_uint),             # effective time of event
    ('type', c_short),            # event type
    ('read', c_ushort),           # flags which items were included
    ('sttime', c_uint),           # event start time
    ('entime', c_uint),           # event end time
    ('hstx', cf),                 # headref starting x
    ('hsty', cf),                 # headref starting y
    ('gstx', cf),                 # gaze starting x
    ('gsty', cf),                 # gaze starting y
    ('sta', cf), 
    ('henx', cf),                 # headref ending x 
    ('heny', cf),                 # headref ending y
    ('genx', cf),                 # gaze ending x 
    ('geny', cf),                 # gaze ending y
    ('ena', cf), 
    ('havx', cf),                 # headref average x
    ('havy', cf),                 # headref average y
    ('gavx', cf),                 # gaze average x
    ('gavy', cf),                 # gaze average y
    ('ava', cf), 
    ('avel', cf),                 # accumulated average velocity 
    ('pvel', cf),                 # accumulated peak velocity 
    ('svel', cf),                 # start velocity
    ('evel', cf),                 # end velocity
    ('supd_x', cf),               # start units-per-degree x
    ('eupd_x', cf),               # end units-per-degree x
    ('supd_y', cf),               # start units-per-degree y
    ('eupd_y', cf),               # end units-per-degree y
    ('eye', c_short),             # eye: 0=left, 1=right 
    ('status', c_ushort),         # error, warning flags 
    ('flags', c_ushort),          # error, warning flags 
    ('input', c_ushort),          # extra (input word)
    ('buttons', c_ushort),        # button state & changes
    ('parsedby', c_ushort),       # 7 bits of flags: PARSEDBY codes 
    ('message', POINTER(LSTRING)) # any message string
]


class RECORDINGS(Structure):
    """The RECORDINGS structure holds information about a recording block in 
    an EDF file. A RECORDINGS structure is present at the start of recording 
    and the end of recording. Conceptually a RECORDINGS structure is similar 
    to the START and END lines inserted in an EyeLink ASC file. RECORDINGS 
    with a state field = 0 represent the end of a recording block, and contain 
    information regarding the recording options set before recording was initiated."""
    pass

RECORDINGS.__slots__ = [
    'time', 'sample_rate', 'eflags', 'sflags', 'state', 'record_type',
    'pupil_type', 'recording_mode', 'filter_type', 'pos_type', 'eye']
RECORDINGS._fields_ = [
    ('time', c_uint),            # start time or end time
    ('sample_rate', cf),         # 250 or 500 
    ('eflags', c_ushort),
    ('sflags', c_ushort),
    ('state', c_ubyte),          # 0 = END, 1=START 
    ('record_type', c_ubyte),
    ('pupil_type', c_ubyte),     # 0 = AREA, 1 = DIAMETER 
    ('recording_mode', c_ubyte), # 0 = PUPIL, 1 = CR
    ('filter_type', c_ubyte),    # 1, 2, 3 
    ('pos_type', c_ubyte),       # 0 = GAZE, 1= HREF, 2 = RAW 
    ('eye', c_ubyte)             # 1=LEFT, 2=RIGHT, 3=LEFT and RIGHT
]


class EDFFILE(Structure):
    """EDFFILE is a dummy structure that holds an EDF file handle."""
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define EDF API functions.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

doc = """Opens the EDF file passed in by edf_file_name and preprocesses the EDF file.

Parameters
----------
edf_file_name: str
    Name of the EDF file to be opened.
consistency: 0 | 1 | 2 
    Consistency check control. If 0, no consistency check. If 1, check
    consistency and report. If 2, check consistency and fix.
load_events: 0 | 1
    Load (1) or skip loading (0) events.
load_samples: 0 | 1
    Load (1) or skip loading (0) samples.
errval : int
    Error value to return if unsucessful.

Returns
-------
EDFFILE: pointer
    if successful a pointer to EDFFILE structure is returned. Otherwise NULL is returned. 
"""
edf_open_file = edfapi.edf_open_file
edf_open_file.__doc__ = doc
edf_open_file.argtypes = [c_char_p, c_int, c_int, c_int, POINTER(c_int)]
edf_open_file.restype = POINTER(EDFFILE)


doc = """Closes an EDF file pointed to by the given EDFFILE pointer and releases all of the 
resources (memory and physical file) related to this EDF file.

Parameters
----------
EDFFILE : pointer
    A valid pointer to EDFFILE structure created by calling `edf_open_file`.

Returns
-------
errval : int
    If successful it returns 0, otherwise a non zero is returned. 
"""
edf_close_file = edfapi.edf_close_file
edf_close_file.__doc__ = doc
edf_close_file.argtypes = [POINTER(EDFFILE)]
edf_close_file.restype = c_int


doc = """Returns the type of the next data element in the EDF file pointed to by *edf. 

Parameters
----------
EDFFILE : pointer
    A valid pointer to EDFFILE structure created by calling `edf_open_file`.

Returns
-------
DATA : STRUCTURE
    Upcoming data event.

Notes
-----
Each call to edf_get_next_data() will retrieve the next data element within the data file. 
The contents of the data element are not accessed using this method, only the type of the 
element is provided. Use edf_get_float_data() instead to access the contents of the data element.
"""
edf_get_next_data = edfapi.edf_get_next_data
edf_get_next_data.__doc__ = doc
edf_get_next_data.argtypes = [POINTER(EDFFILE)]
edf_get_next_data.restype = c_int


doc = """Returns the length of the preamble text.

Parameters
----------
EDFFILE : pointer
    A valid pointer to EDFFILE structure created by calling `edf_open_file`.

Returns
-------
len : int
    An integer for the length of preamble text.
"""
edf_get_preamble_text_length = edfapi.edf_get_preamble_text_length
edf_get_preamble_text_length.__doc__ = doc
edf_get_preamble_text_length.argtypes = [POINTER(EDFFILE)]
edf_get_preamble_text_length.restype = c_int


doc = """Copies the preamble text into the given buffer. 

Parameters
----------
EDFFILE : pointer
    A valid pointer to EDFFILE structure created by calling `edf_open_file`.
buffer : str
    A character array to be filled by the preamble text. 
Lenght : int
    Length of the buffer.

Returns
-------
Errval : int
    Returns 0 if the operation is successful. 

Notes
-----
If the preamble text is longer than the length the text will be truncated. 
The returned content will always be null terminated.
"""
edf_get_preamble_text = edfapi.edf_get_preamble_text
edf_get_preamble_text.__doc__ = doc
edf_get_preamble_text.argtypes = [POINTER(EDFFILE), c_char_p, c_int]
edf_get_preamble_text.restype = c_int


doc = """Return information about a recording block in an EDF file.

Parameters
----------
EDFFILE : pointer
    A valid pointer to EDFFILE structure created by calling `edf_open_file`.

Returns
-------
RECORDINGS : pointer
    A sample RECORDINGS structure.
"""
edf_get_recording_data = edfapi.edf_get_recording_data
edf_get_recording_data.__doc__ = doc
edf_get_recording_data.argtypes = [POINTER(EDFFILE)]
edf_get_recording_data.restype = POINTER(RECORDINGS)


doc = """Return information for a sample in the EDF file. 

Parameters
----------
EDFFILE : pointer
    A valid pointer to EDFFILE structure created by calling `edf_open_file`.

Returns
-------
FSAMPLE : pointer
    A sample FSAMPLE structure.
"""
edf_get_sample_data = edfapi.edf_get_sample_data
edf_get_sample_data.__doc__ = doc
edf_get_sample_data.argtypes = [POINTER(EDFFILE)]
edf_get_sample_data.restype = POINTER(FSAMPLE)


doc = """Return information for an event in the EDF file.

Parameters
----------
EDFFILE : pointer
    A valid pointer to EDFFILE structure created by calling `edf_open_file`.

Returns
-------
FEVENT : pointer
    A sample FEVENT structure.
"""
edf_get_event_data = edfapi.edf_get_event_data
edf_get_event_data.__doc__ = doc
edf_get_event_data.argtypes = [POINTER(EDFFILE)]
edf_get_event_data.restype = POINTER(FEVENT)
