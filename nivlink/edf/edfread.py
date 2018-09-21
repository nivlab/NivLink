import os
from datetime import datetime
from ctypes import byref, c_int, create_string_buffer, string_at
from .edfapi import (edf_open_file, edf_close_file, edf_get_next_data,
                    edf_get_preamble_text_length, edf_get_preamble_text,
                    edf_get_recording_data, edf_get_sample_data, edf_get_event_data)
from .constants import event_codes
error_code = byref(c_int(1))

def edf_parse_preamble(EDFFILE):
    """Parse EDF preamble for dictionary lookup."""
    
    ## Preallocate space.
    n = edf_get_preamble_text_length(EDFFILE)
    preamble = create_string_buffer(n)
    
    ## Extract preamble text.
    edf_get_preamble_text(EDFFILE, preamble, n + 1)
    
    ## Preprocess preamble text.
    preamble = preamble.value.decode('ASCII').split('\n')
    preamble = [s.replace('**','').strip() for s in preamble]
    preamble = [s for s in preamble if s]
    
    ## Store in dictionary.
    info = {}
    for line in preamble:
        k, v = line[:line.find(':')], line[line.find(':')+1:].strip()
        if 'DATE' in k: 
            fmt = '%a %b %d %H:%M:%S %Y'
            info['meas_date'] = datetime.strptime(v, fmt)
        elif 'CAMERA' in k:
            info['camera'] = v
        elif 'VERSION' in k:
            info['version'] = v
        
    return info

def edf_read(fname)

    ## Define EDF filepath.
    fname = os.path.normpath(os.path.abspath(fname).encode("ASCII"))
    if not os.path.isfile(fname): raise IOError('File not found.')
        
    ## Preallocate space.
    samples, blinks, saccades, messages = [], [], [], []
    
    ## Open EDFFILE.
    EDFFILE = edf_open_file(fname, 1, 1, 1, error_code)

    ## Parse preamble (initialize info.)
    info = edf_parse_preamble(EDFFILE)

    ## Main loop.
    event = True
    while event:

        ## Get next event.
        event = edf_get_next_data(edf)
        
        if event_codes[event] == 'SAMPLES':
            
            ## do some stuff.
            
        elif event_codes[event] == 'STARTBLINK':
            
            ## do some stuff
            
        elif event_codes[event] == 'STARTSACC':
            
            ## do some stuff.
            
        elif event_codes[event] == 'MESSAGEEVENT':
            
            ## do some stuff.
            
        elif event_codes[event] == 'RECORDING':
            
            ## do some stuff.
            
        else:
            
            continue
            

    ## Close EDFFILE.
    edf_close_file(EDFFILE);