import os
from numpy import array, float64, unicode_, searchsorted
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

#TODO: allow binocular recording
def edf_parse_sample(EDFFILE):
    """Return sample info: time, eye fixation, pupil size (left/right)."""    
    sample = edf_get_sample_data(EDFFILE).contents    
    return (sample.time, sample.gx[0], sample.gx[1], sample.gy[0], sample.gy[1],
            sample.pa[0], sample.pa[1])

def edf_parse_blink(EDFFILE):
    """Return blink info: start, end."""
    blink = edf_get_event_data(EDFFILE).contents
    return (blink.sttime, blink.entime)

def edf_parse_saccade(EDFFILE):
    """Return saccade info: start, end."""
    saccade = edf_get_event_data(EDFFILE).contents
    return (saccade.sttime, saccade.entime)

def edf_parse_message(EDFFILE):
    """Return message info."""
    message = edf_get_event_data(EDFFILE).contents
    time = message.sttime
    message = string_at(byref(message.message[0]), message.message.contents.len + 1)[2:]
    message = message.decode('UTF-8')
    return (time, message)

def edf_parse_recording(EDFFILE, info):
    """Return recording info: sample rate, eye info."""
    recording = edf_get_recording_data(EDFFILE).contents
    if recording.state:
        info['sfreq'] = recording.sample_rate
        info['eye'] = {1:'LEFT', 2:'RIGHT', 3:'BOTH'}.get(recording.eye,'NA')
        info['pupil'] = {0:'AREA', 1:'DIAMETER'}.get(recording.pupil_type,'NA')
    return info
        
def edf_read(fname):
    """Read and parse EDF file.
    
    Parameters
    ----------
    fname : str
        Path to EDF file.
        
    Returns
    -------
    info : dict
        EDF file metadata.
    times : array, shape (n,)
        Time of recording samples (in seconds).
    data : array, shape (n, 3)
        Recording samples comprised of gaze_x, gaze_y, pupil.
    blinks : array, shape (i, 2)
        Detected blinks detailed by their start and end.
    saccades : array, shape (j, 2)
        Detected saccades detailed by their start and end.
    messages : array, shape (k, 2)
        Detected messages detailed by their time and message.
    """
    
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
        event = edf_get_next_data(EDFFILE)
        code = event_codes.get(event, 'NA')
        
        if code == 'NA':
            edf_close_file(EDFFILE);
            raise ValueError('Code %s not recognized.' %code)
        
        elif code == 'SAMPLES':
            samples.append( edf_parse_sample(EDFFILE) )
            
        elif code == 'ENDBLINK':
            blinks.append( edf_parse_blink(EDFFILE) )
            
        elif code == 'ENDSACC':
            saccades.append( edf_parse_blink(EDFFILE) )
            
        elif code == 'MESSAGEEVENT':
            messages.append( edf_parse_message(EDFFILE) )
            
        elif code == 'RECORDING':
            info = edf_parse_recording(EDFFILE, info)
            
    ## Close EDFFILE.
    edf_close_file(EDFFILE);
    
    ## Extract data.    
    samples = array(samples, dtype=float64)
    if info['eye'] == 'LEFT': 
        data = samples[:,1::2]
    elif info['eye'] == 'RIGHT': 
        data = samples[:,2::2]
    else: 
        raise ValueError('Binocular data not supported.')
    
    ## Format time.
    times = samples[:,0].astype(int)
    start_time = int(times[0])
    times -= start_time  
    
    ## Format blinks.
    blinks = array(blinks, dtype=int) - start_time
    blinks = searchsorted(times, blinks)
    
    ## Format saccades.
    saccades = array(saccades, dtype=int) - start_time
    saccades = searchsorted(times, saccades)
    
    ## Format messages.
    messages = array(messages, dtype=[('sample',int),('message',unicode_, 80)])        
    messages['sample'] = messages['sample'] - start_time
    messages['sample'] = searchsorted(times, messages['sample'])
    
    ## Define channel names.
    ch_names = ['gx','gy','pupil']
    
    return info, data, blinks, saccades, messages, ch_names