import numpy as np
import os.path as op
from .edf import edf_read

def _load_npz(fname):
    """Load raw from NumPy compressed file."""
    npz = np.load(fname)
    return (npz['info'].tolist(), npz['times'], npz['data'], 
            npz['blinks'], npz['saccades'], npz['messages'])

class Raw(object):
    """Raw data object.
    
    Parameters
    ----------
    fname : str
        The raw file to load. Supported file extensions are .edf and .npz.
        
    Attributes
    ----------
    info : dict
        Recording metadata.
    n_times : int
        Total number of time points in the raw file.
    times : array, shape (n_times,)
        Time vector in seconds starting at 0. Time interval between consecutive 
        time samples is equal to the inverse of the sampling frequency.
    data : array, shape (n_times, 3)
        Recording samples comprised of gaze_x, gaze_y, pupil.
    blinks : array, shape (i, 2)
        Detected blinks detailed by their start and end.
    saccades : array, shape (j, 2)
        Detected saccades detailed by their start and end.
    messages : array, shape (k, 2)
        Detected messages detailed by their time and message.
    """
    
    def __init__(self, fname):
        
        ## Read file.
        _, ext = op.splitext(fname.lower())
        if ext == '.edf':
            info, times, data, blinks, saccades, messages = edf_read(fname)
        elif ext == '.npz':
            info, times, data, blinks, saccades, messages = _load_npz(fname)
        else: 
            raise IOError('Raw supports only .edf or .npz files.')
                
        ## Store metadata.
        self.info = info
        self.times = times
        self.n_times = times.size
        
        ## Store samples.
        self.data = data
        self.blinks = blinks
        self.saccades = saccades
        self.messages = messages
        
    def __repr__(self):
        return '<Raw | {0} samples>'.format(self.n_times)
    
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
        if op.isfile(fname) and not overwrite: 
            raise IOError('file "%s" already exists.' %fname) 
        
        ## Otherwise save.
        np.savez_compressed(fname, info=self.info, times=self.times, data=self.data, 
                            blinks=self.blinks, saccades=self.saccades, messages=self.messages)