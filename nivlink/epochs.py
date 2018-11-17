import numpy as np
from copy import deepcopy

class Epochs(object):
    """Epochs extracted from a Raw instance.
    
    Parameters
    ----------
    raw : instance of `Raw`
        Raw data to be epoched.
    onsets : array
            Onset of events (in raw indices).
    tmin : float | array
        Start time before event. If float, all events start at same
        time relative to event onset.
    tmax : float | array
        End time after event. If float, all events start at same
        time relative to event onset.
    picks : 'gaze' | 'pupil' | None
        Data types to include (if None, all data are used).
        
    Attributes
    ----------
    info : dict
        Recording metadata.
    times : array, shape (n_times,)
        Time vector in seconds. Goes from `tmin` to `tmax`. Time interval
        between consecutive time samples is equal to the inverse of the
        sampling frequency.
    data : array, shape (n_trials, n_times, n_channels)
        Recording samples.       
    ch_names : list
        Names of data channels.
    """
    
    def __init__(self, raw, events, tmin=0, tmax=1, picks=None):
        
        ## Define metadata.
        self.info = deepcopy(raw.info)

        ## Error-catching.
        assert np.ndim(events) == 1
        if isinstance(tmin, (int, float)): tmin = np.repeat(tmin, events.size)
        if isinstance(tmax, (int, float)): tmax = np.repeat(tmax, events.size)
        assert np.size(events) == np.size(tmin) == np.size(tmax)

        ## Convert times to sampling frequency.
        sfreq = self.info['sfreq']
        tmin = np.array(tmin * sfreq).astype(int) / sfreq
        tmax = np.array(tmax * sfreq).astype(int) / sfreq   
        self.times = np.arange(tmin.min(), tmax.max(), 1/sfreq)

        ## Define indices of data relative to raw.
        raw_idx = np.column_stack([events + tmin * sfreq, events + tmax * sfreq])

        ## Define indices of data relative to epochs.
        epoch_idx = (np.column_stack([tmin,tmax]) - tmin.min()) * sfreq

        ## Make epochs.
        self.data = np.ones((events.shape[0], self.times.size, 3)) * np.nan
        index = np.column_stack((raw_idx, epoch_idx)).astype(int)
        for i, (r1, r2, e1, e2) in enumerate(index): 
            self.data[i,e1:e2,:] = raw.data[r1:r2].copy()
        self.data = self.data.swapaxes(1,2)
            
        ## Update channels.
        if picks is None:
            self.ch_names = raw.ch_names
        elif picks.lower().startswith('g'):
            self.ch_names = ['gx','gy']
            self.data = self.data[:,:2]
        elif picks.lower().startswith('p'):
            self.ch_names = ['pupil']
            self.data = self.data[:,-1:]
    
    def __repr__(self):
        return '<Epochs | {0} trials, {2} times>'.format(*self.data.shape)
    
    def copy(self):
        """Return copy of Raw instance."""
        return deepcopy(self)
    
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
                            saccades=self.saccades, messages=self.messages, ch_names=self.ch_names)