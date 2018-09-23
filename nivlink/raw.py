import re
import numpy as np
import os.path as op
from scipy.ndimage import measurements
from .edf import edf_read

def _load_npz(fname):
    """Load raw from NumPy compressed file."""
    npz = np.load(fname)
    return (npz['info'].tolist(), npz['times'], npz['data'], 
            npz['blinks'], npz['saccades'], npz['messages'])

def _moving_average(data_set, window):
    """Simple moving average."""
    weights = np.ones(window) / window
    return np.convolve(data_set, weights, mode='valid')

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
    
    def detect_blinks(self, window, min_dist=0.1, overwrite=True):
        """Detect blinks from pupillometry data.
        
        Parameters
        ----------
        window : int | float
            Length of smoothing window. If int, number of samples. If float, 
            number of seconds.
        min_dist : float
            Minimum length (in seconds) between two successive blinks. 
            If shorter, blinks are merged into single event.
        overwrite : bool
            Overwrite EyeLink-detected blinks. If False, returns blinks.
        
        Notes
        -----
        Blink detection algorithm presented in Hershman et al. (2018). Briefly, 
        the algorithm:
        
        1. Detects blink periods (pupil = 0).
        2. Merges nearby blinks, defined as occurring within :code:`min_dist`.
        3. Extends blink period to start/end of slope. 
        
        See paper for details. Usually preferable to EyeLink default blinks. 
        
        References
        ----------
        [1] Hersman et al. (2018). https://doi.org/10.3758/s13428-017-1008-1.
        """
        if isinstance(window, float): window = int(window * self.info['sfreq'])
        assert isinstance(window, int)
        
        ## STEP 1: Identify blinks.
        blinks, n_blinks = measurements.label(self.data[:,-1] == 0)
        onsets = measurements.minimum(np.arange(blinks.size), labels=blinks, index=np.arange(n_blinks)+1)
        offsets = measurements.maximum(np.arange(blinks.size), labels=blinks, index=np.arange(n_blinks)+1)
        blinks = np.column_stack((onsets, offsets))

        ## STEP 2: Merge blinks.
        while True:

            ## Check if any blinks within min dist.
            adjacency = np.diff(self.times[blinks.flatten()])[1::2] < min_dist
            if not np.any(adjacency): break

            ## Update blinks.
            i = np.argmax(adjacency)
            blinks[i,-1] = blinks[i+1,-1]            # Store second blink offset as first.
            blinks = np.delete(blinks, i+1, axis=0)  # Remove redundant blink.

        ## STEP 3: Extend windows.
        for i, (onset, offset) in enumerate(blinks):

            ## Extend onset.
            smooth = _moving_average(self.data[onset-window*10:onset,-1], window)
            first_samp = np.argmax(np.diff(smooth)[::-1] >= 0)
            blinks[i,0] -= first_samp

            ## Extend offset.
            smooth = _moving_average(self.data[offset:offset+window*10:,-1], window)
            first_samp = np.argmax(np.diff(smooth) <= 0)
            blinks[i,1] += first_samp
            
        if overwrite: self.blinks = blinks
        else: return blinks
    
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
            Event times (in samples) corresponding to events that were found.
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
        if op.isfile(fname) and not overwrite: 
            raise IOError('file "%s" already exists.' %fname) 
        
        ## Otherwise save.
        np.savez_compressed(fname, info=self.info, times=self.times, data=self.data, 
                            blinks=self.blinks, saccades=self.saccades, messages=self.messages)