import numpy as np

def epoching_fht(raw, info, events, template='Start of Run%s'):
    '''Epoch the raw eyetracking data from the FHTconfidence dimensions
    task dataset collected by Angela Radulescu & Julie Newman.
    
    Parameters
    ----------
    raw : array, shape (n_times, 3)
      Raw eyetracking data. The first column contains the event messages.
      The second and third columns contain the eye positions in the x- and
      y-dimensions, respectively.
    info : instance of `ScreenInfo`
      Eyetracking acquisition information.
    events : array, shape (n_trials, 3)
      Events data. Each row represents a trial. The first column
      denotes the block. The second column denotes the trial onset
      relative to the block. The third column denotes the trial length.
    template : str
      Template for block start note as found in the EyeLink output file.
    
    Returns
    -------
    epochs : array, shape (n_trials, n_times, 2)
      Epoched eyetracking timeseries data. Last dimension is the
      spatial dimension of the eyetracker (xdim, ydim).
    '''
    
    ## Define elapsed time relative to eyetracking.
    times = np.arange(0, raw.shape[0] / info.sfreq, 1 / info.sfreq)
    
    ## Define start of blocks relative to elapsed time.
    blocks = np.unique(events[:,0]).astype(int)
    indices = [(raw[:,0]==template %i).argmax() for i in blocks]
    block_starts = np.zeros_like(times)
    block_starts[indices] = blocks
    
    ## Offset trial starts by block offsets.
    events = events.copy()
    for block, offset in zip(blocks, times[block_starts.nonzero()]):
        events[events[:,0]==block, 1] += offset
        
    ## Redefine trial onset as index corresponding to elapsed time.
    onsets = np.array([np.argmax(times > t) for t in events[:,1]])
    offsets = (onsets + info.sfreq * events[:,-1]).astype(int)
    
    ## Epoching.
    epochs = np.array([raw[i:j+1, 1:] for i,j in np.column_stack([onsets,offsets])])
    
    return epochs.astype(float)