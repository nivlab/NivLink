import os, re
import numpy as np
from copy import deepcopy
from .edf import edf_read

def _load_npz(fname):
    """Load raw from NumPy compressed file."""
    npz = np.load(fname, allow_pickle=True)
    return (npz['info'].tolist(), npz['data'], npz['blinks'], 
            npz['saccades'], npz['messages'], 
            tuple(npz['ch_names']), tuple(npz['eye_names']))

class Raw(object):
    """Raw data instance.
    
    Parameters
    ----------
    fname : str
        The raw file to load. Supported file extensions are .edf and .npz.
        
    Attributes
    ----------
    info : dict
        Recording metadata.
    n_samp : int
        Total number of samples in the raw file.
    data : array, shape (n_times, n_eyes, n_channels)
        Recording samples comprised of gaze_x, gaze_y, pupil.
    ch_names : list
        Names of data channels.
    eye_names : list
        Order of data channels (by eye).
    blinks : array, shape (i, 2)
        Detected blinks detailed by their start and end.
    saccades : array, shape (j, 2)
        Detected saccades detailed by their start and end.
    messages : array, shape (k, 2)
        Detected messages detailed by their time and message.
        
    Notes
    -----
    When reading in raw data from an EDF file, NivLink applies no transformations.
    That is, whatever values are present in the raw data are transcribed faithfully
    into the corresponding Nivlink.Raw object. This means that missing data will 
    typically be represented by large or otherwise outlier values (and not, for example,
    by NaNs). We leave it to the user to define what are pathological values.
    
    Similarly, we do not transform raw pixel values. Typically, this means that a 
    value of 0 indicates the top of the monitor. Please check your EyeLink recording
    parameters to confirm.
    
    Monocular and binocular recordings are supported by NivLink. In the case of binocular
    data, the order of data is left followed by right eye.    
    """
    
    def __init__(self, fname):
        
        ## Read file.
        _, ext = os.path.splitext(fname.lower())
        if ext == '.edf':
            info, data, blinks, saccades, messages, ch_names, eye_names = edf_read(fname)
        elif ext == '.npz':
            info, data, blinks, saccades, messages, ch_names, eye_names = _load_npz(fname)
        else: 
            raise IOError('Raw supports only .edf or .npz files.')
                
        ## Store metadata.
        self.info = info
        self.n_samp = data.shape[0]
        self.ch_names = ch_names
        self.eye_names = eye_names
        
        ## Store samples.
        self.data = data
        self.blinks = blinks
        self.saccades = saccades
        self.messages = messages
        
    def __repr__(self):
        return '<Raw | {0} samples>'.format(self.n_samp)
    
    def copy(self):
        """Return copy of Raw instance."""
        return deepcopy(self)
    
    def find_events(self, pattern, return_messages=False):
        """Find events from messages.

        Parameters
        ----------
        pattern : string
            Pattern to search for in messages. Supports regex.
        return_messages : bool
            Return matching messages.

        Returns
        -------
        onsets : array, shape (n_events,) 
            Event times (in seconds) corresponding to events that were found.
        messages : array, shape (n_events,)
            Corresponding messages. Returns if return_messages = True.
        """

        ## Identify matching messages.
        f = lambda string: True if re.search(pattern,string) is not None else False
        ix = [f(msg) for msg in self.messages['message']]

        ## Gather events.
        onsets = self.messages['sample'][ix]
        messages = self.messages['message'][ix]

        if return_messages: return onsets, messages
        else: return onsets
            
    def save(self, fname, overwrite=False):
        """Save data to NumPy compressed format.
        
        Parameters
        ----------
        fname : str
            Filename to use.
        overwrite : bool
            If True, overwrite file (if it exists).
        """
        
        ## Check if exists.
        if os.path.isfile(fname) and not overwrite: 
            raise IOError('file "%s" already exists.' %fname) 
        
        ## Otherwise save.
        np.savez_compressed(fname, info=self.info, data=self.data, blinks=self.blinks, 
                            saccades=self.saccades, messages=self.messages, 
                            ch_names=self.ch_names, eye_names=self.eye_names)