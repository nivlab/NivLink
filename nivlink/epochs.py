import numpy as np
from copy import deepcopy

class Epochs(object):
    """Epochs extracted from a Raw instance.
    
    Parameters
    ----------
    raw : instance of `Raw`
        Raw data to be epoched.
    events : array
        Event onsets (in sample indices).
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
    data : array, shape (n_trials, n_times, n_channels)
        Recording samples.    
    times : array, shape (n_times,)
        Time vector in seconds. Goes from `tmin` to `tmax`. Time interval
        between consecutive time samples is equal to the inverse of the
        sampling frequency.
    events : array, shape (n_trials, 2)
        Onset and offset of trials.
    ch_names : list
        Names of data channels.
    """
    
    def __init__(self, raw, events, tmin=0, tmax=1, picks=None, blinks=False, saccades=False):
        
        ## Define metadata.
        self.info = deepcopy(raw.info)

        ## Define channels.
        if picks is None: ch_names = ['gx','gy','pupil']
        elif picks.lower().startswith('g'): ch_names = ['gx','gy']
        elif picks.lower().startswith('p'): ch_names = ['pupil']
        self.ch_names = np.intersect1d(ch_names, raw.ch_names)
        ch_ix = np.in1d(raw.ch_names,self.ch_names)
            
        ## Define events.
        assert np.ndim(events) == 1
        if isinstance(tmin, (int, float)): tmin = np.repeat(tmin, events.size)
        if isinstance(tmax, (int, float)): tmax = np.repeat(tmax, events.size)
        assert np.size(events) == np.size(tmin) == np.size(tmax)
        self.extents = np.column_stack([tmin, tmax])
        
        ## Convert times to sampling frequency.
        sfreq = self.info['sfreq']
        tmin = np.array(tmin * sfreq).astype(int) / sfreq
        tmax = np.array(tmax * sfreq).astype(int) / sfreq   
        self.times = np.arange(tmin.min(), tmax.max(), 1/sfreq)

        ## Define indices of data relative to raw.
        raw_ix = np.column_stack([events + tmin * sfreq, events + tmax * sfreq])

        ## Define indices of data relative to epochs.
        epoch_ix = (np.column_stack([tmin,tmax]) - tmin.min()) * sfreq
        self._ix = epoch_ix.astype(int)
        
        ## Make epochs.
        self.data = np.ones((events.shape[0], self.times.size, len(self.ch_names))) * np.nan
        index = np.column_stack((raw_ix, epoch_ix)).astype(int)
        for i, (r1, r2, e1, e2) in enumerate(index): 
            self.data[i,e1:e2,:] = deepcopy(raw.data[r1:r2,ch_ix])
        self.data = self.data.swapaxes(1,2)
            
        ## Re-reference artifacts to epochs.
        if blinks: self.blinks = self._align_artifacts(raw.blinks, raw_ix)
        if saccades: self.saccades = self._align_artifacts(raw.saccades, raw_ix)
        
    def _align_artifacts(self, artifacts, raw_ix):
        """Re-aligns artifacts (blinks, saccades) from raw to epochs times.
        
        Parameters
        ----------
        artifacts : array, shape=(n_artifacts, 2)
            Blinks or saccades index array from Raw object.
        raw_ix : array, shape=(n_events, 2)
            Events index array computed in Epoching.
            
        Returns
        -------
        artifacts : array, shape=(n_artifacts, 3)
            Artifacts (blinks, saccades) overlapping with events. First column 
            denotes event number, second column denotes artifact onset, and
            third column denotes artifact offset.
        """
            
        ## Broadcast trial onsets/offsets to number of blinks.
        n_events, _, n_times = self.data.shape
        onsets  = np.broadcast_to(raw_ix[:,0], (artifacts.shape[0], n_events)).T
        offsets = np.broadcast_to(raw_ix[:,1], (artifacts.shape[0], n_events)).T

        ## Identify where blinks overlap with trials.
        overlap = np.logical_and(onsets < artifacts[:,1], offsets > artifacts[:,0])
        trials, ix = np.where(overlap)

        ## Assemble blinks data. Realign to epoch times.
        artifacts = np.column_stack([trials, artifacts[ix]])
        for i in np.unique(artifacts[:,0]):
            artifacts[artifacts[:,0]==i,1:] -= int(raw_ix[i,0])
            
        ## Boundary correction.
        artifacts = np.where(artifacts < 0, 0, artifacts)
        artifacts = np.where(artifacts > n_times, n_times, artifacts)
        
        return artifacts
    
    def __repr__(self):
        return '<Epochs | {0} trials, {2} samples>'.format(*self.data.shape)
    
    def copy(self):
        """Return copy of Raw instance."""
        return deepcopy(self)