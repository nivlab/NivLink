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
    eyes : 'LEFT' | 'RIGHT' | None
        Eye recordings to include (if None, all data are used).
    blinks : True | False
        Include blinks and re-reference to epochs.
    saccades : True | False
        Include saccades and re-ference to epochs.
        
    Attributes
    ----------
    info : dict
        Recording metadata.
    data : array, shape (n_trials, n_eyes, n_channels, n_times)
        Recording samples.    
    times : array, shape (n_times,)
        Time vector in seconds. Goes from `tmin` to `tmax`. Time interval
        between consecutive time samples is equal to the inverse of the
        sampling frequency.
    extents : array, shape (n_trials, 2)
        Onset and offset of trials.
    ch_names : list, shape (n_channels)
        Names of data channels.    
    eye_names : list, shape (n_eyes)
        Order of data channels (by eye).
    blinks : array, shape (i, 3)
        (If included) Detected blinks detailed by their trial, start, and end.
    saccades : array, shape (j, 3)
        (If included) Detected saccades detailed by their trial, start, and end.
    """
    
    def __init__(self, raw, events, tmin=0, tmax=1, picks=None, eyes=None, 
                 blinks=True, saccades=True):
        
        ## Define metadata.
        self.info = deepcopy(raw.info)

        ## Define channels.
        if picks is None: ch_names = ('gx','gy','pupil')
        elif picks.lower().startswith('g'): ch_names = ('gx','gy')
        elif picks.lower().startswith('p'): ch_names = ('pupil')
        else: raise ValueError(f'"{picks}" not valid input for picks.')
        self.ch_names = tuple(np.intersect1d(ch_names, raw.ch_names))
        ch_ix = np.in1d(raw.ch_names,self.ch_names)

        ## Define eyes.
        if eyes is None: eye_names = deepcopy(raw.eye_names)
        elif eyes.lower().startswith('l'): eye_names = ('LEFT')
        elif eyes.lower().startswith('r'): eye_names = ('RIGHT')
        else: raise ValueError(f'"{eyes}" not valid input for eyes.')
        self.eye_names = tuple(np.intersect1d(eye_names, raw.eye_names))
        eye_ix = np.in1d(raw.eye_names,self.eye_names)
            
        ## Define events.
        assert np.ndim(events) == 1
        if isinstance(tmin, (int, float)): tmin = np.repeat(float(tmin), events.size)
        if isinstance(tmax, (int, float)): tmax = np.repeat(float(tmax), events.size)
        assert np.size(events) == np.size(tmin) == np.size(tmax)
        self.extents = np.column_stack([tmin, tmax])
        
        ## Convert times to sampling frequency.
        sfreq = self.info['sfreq']
        tmin = np.array(tmin * sfreq).astype(int) / sfreq
        tmax = np.array(tmax * sfreq).astype(int) / sfreq   
        self.times = np.arange(tmin.min(), tmax.max(), 1/sfreq)

        ## Define indices of data relative to raw.
        raw_ix = np.column_stack([events + tmin * sfreq, events + tmax * sfreq])
        raw_ix = np.rint(raw_ix).astype(int)
        
        ## Define indices of data relative to epochs.
        epoch_ix = (np.column_stack([tmin,tmax]) - tmin.min()) * sfreq
        epoch_ix = np.rint(epoch_ix).astype(int)
        
        ## Error-catching: assert equal array lengths.
        epoch_ix[:,-1] += np.squeeze(np.diff(raw_ix) - np.diff(epoch_ix))
        self._ix = epoch_ix.astype(int)
        
        ## Make epochs.
        self.data = np.ones((events.shape[0], self.times.size, len(self.eye_names), len(self.ch_names))) * np.nan
        index = np.column_stack((raw_ix, epoch_ix))
        for i, (r1, r2, e1, e2) in enumerate(index):
            # TODO: This ugly syntax should be replaced in time (numpy issues 13255)
            self.data[i,e1:e2,...] = deepcopy(raw.data[r1:r2,eye_ix][...,ch_ix])
        self.data = np.moveaxis(self.data,1,-1)
                        
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
        n_events, _, _, n_times = self.data.shape
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
        return '<Epochs | {0} trials, {3} samples>'.format(*self.data.shape)
    
    def copy(self):
        """Return copy of Raw instance."""
        return deepcopy(self)