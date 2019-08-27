import numpy as np
from pandas import DataFrame
from scipy.ndimage import measurements
from .raw import Raw
from .epochs import Epochs    

def align_to_aoi(data, screen, mapping=None):
    """Align eyetracking data to areas of interest.

    Parameters
    ----------
    data : Raw | Epochs | array, shape=(n_trials, n_eyes, n_channels, n_times)
        Trials to be aligned.
    screen : nivlink.Screen
        Eyetracking acquisition information.
    mapping : array, shape (n_trials,)
        Mapping of trials to screens. If None, all trials mapped to 
        first Screen. Should be zero-indexed.

    Returns
    -------
    aligned : array, shape (n_trials, n_eyes, n_times)
        Eyetracking timeseries aligned to areas of interest.  

    Notes
    -----
    The alignment step makes two critical assumptions during processing:

    1. Eyetracking positions are rounded down to the nearest pixel.
    2. Eyetracking positions outside (xdim, ydim) are set to NaN.
    """
    
    if isinstance(data, Raw):
        
        ## Error-catching: force gx and gy to be present.
        if not np.all(np.in1d(['gx','gy'], data.ch_names)):
            raise ValueError('Both gaze channels (gx, gy) must be present.')
                        
        ## Copy data.
        gaze_ix = np.in1d(data.ch_names, ['gx','gy'])
        data = data.data[..., gaze_ix].copy()
        data = np.expand_dims(data, 0)
        data = data.swapaxes(2,3)
        
    if isinstance(data, Epochs):
        
        ## Error-catching: force gx and gy to be present.
        if not np.all(np.in1d(['gx','gy'], data.ch_names)):
            raise ValueError('Both gaze channels (gx, gy) must be present.')
                        
        ## Copy data.
        gaze_ix = np.in1d(data.ch_names, ['gx','gy'])
        data = data.data[..., gaze_ix, :].copy()
        data = data.swapaxes(2,3)
        
    else:
                
        ## Error-catching: force gx and gy to be present.
        if np.ndim(data) != 4:
            raise ValueError('data must be shape (..., 2, n_trials)')
        elif np.shape(data)[-2] != 2:
            raise ValueError('data must be shape (..., 2, n_trials)')
           
        ## Copy data.
        data = np.array(data.copy())
        data = data.swapaxes(2,3)
            
    ## Collect metadata. Preallocate space.
    n_trials, n_eyes, n_times, n_dim = data.shape
    xd, yd, n_screens = screen.indices.shape    
    
    ## Round gaze data to nearest pixel.
    data = np.floor(data).astype(int)
    
    ## Identify missing data.
    missing_x = np.logical_or(data[...,0] < 0, data[...,0] >= screen.xdim )
    missing_y = np.logical_or(data[...,1] < 0, data[...,1] >= screen.ydim )
    missing = np.logical_or(missing_x, missing_y)
    
    ## Mask missing data.
    data[missing] = 0
    
    ## Preallocate space.
    aligned = np.zeros_like(missing, dtype=int)
    
    ## Define screen indices.
    if mapping is None: mapping = np.zeros(n_trials, dtype=int)
    assert np.size(mapping) == n_trials
    
    ## Main loop.
    for ix in np.unique(mapping):
        
        ## Extract current screen.
        current_screen = screen.indices[...,ix]
    
        ## Define row and column indices.
        row = data[mapping == ix, :, :, 0].flatten()
        col = data[mapping == ix, :, :, 1].flatten()
        
        ## Align data and reshape.
        t = np.sum(mapping == ix)
        aligned[mapping == ix] = current_screen[row, col].reshape(t, n_eyes, n_times)
    
    ## Mask missing data.
    aligned[missing] = 0
    
    return aligned

def compute_fixations(aligned, times, labels=None):
    """Compute fixations from aligned timeseries. Fixations are defined
    as contiguous samples of eyetracking data aligned to the same AoI.

    Parameters
    ----------
    aligned : array, shape (n_trials, n_times)
        Eyetracking timeseries aligned to areas of interest.  
    times : array, shape (n_times,)
        Time vector in seconds.
    labels : list
        List of areas of interest to include in processing. Defaults 
        to all non-zero values in aligned.

    Returns
    -------
    fixations : pd.DataFrame
      Pandas DataFrame where each row details the (Trial, AoI, 
      Onset, Offset, Duration) of the fixation.
      
    Notes
    -----
    Currently supports only monocular data. In the case of binocular
    data, the user can simply pass the aligned object twice (once
    per eye).
    """
    
    ## Error-catching.
    assert np.ndim(aligned) == 2
    assert np.shape(aligned)[-1] == np.size(times)
    
    ## Define labels list.
    if labels is None: labels = [i for i in np.unique(aligned) if i]

    ## Append extra timepoint to end of each trial. This prevents clusters across
    ## successive trials.
    n_trials, n_times = aligned.shape
    aligned = np.hstack([aligned, np.zeros((n_trials,1))]).flatten()

    ## Precompute trial info.
    trials = np.repeat(np.arange(n_trials),n_times+1) + 1
    times = np.broadcast_to(np.append(times,0), (n_trials,n_times+1)).flatten()
    
    ## Preallocate space.
    df = DataFrame([], columns=('Trial','AoI','Onset','Offset'))

    for label in labels:

        ## Identify clusters.
        clusters, n_clusters = measurements.label(aligned == label)

        ## Identify cluster info.
        trial = measurements.minimum(trials, labels=clusters, index=np.arange(n_clusters)+1)
        onset = measurements.minimum(times, labels=clusters, index=np.arange(n_clusters)+1)
        offset = measurements.maximum(times, labels=clusters, index=np.arange(n_clusters)+1)

        ## Append to DataFrame.
        dat = np.column_stack((trial, np.ones_like(trial)*label, onset, offset))
        df = df.append(DataFrame(dat, columns=df.columns))

    df = df.sort_values(['Trial','Onset']).reset_index(drop=True)
    df['Duration'] = df.Offset - df.Onset

    return df