import numpy as np
from pandas import DataFrame
from scipy.ndimage import measurements
from .raw import Raw
from .epochs import Epochs    

def align_to_aoi(data, screen, screenidx):
    """Align eyetracking data to areas of interest. Please see notes.

    Parameters
    ----------
    data : Raw | Epochs | array, shape=(n_trials, n_times, 2)
        Trials to be aligned.
    screen : instance of Screen
        Eyetracking acquisition information.
    screenidx : array, shape (n_trials,)
        Mapping of trial to screen index.  

    Returns
    -------
    aligned : array, shape (n_trials, n_times)
        Eyetracking timeseries aligned to areas of interest.  

    Notes
    -----
    The alignment step makes two critical assumptions during processing:

    1. Eyetracking positions are rounded down to the nearest pixel.
    2. Eyetracking positions outside (xdim, ydim) are set to NaN.
    """

    ## Data handling (Raw, Epochs).
    if isinstance(data, (Raw, Epochs)):
        
        ## Error-catching: force gx and gy to be present.
        if not np.all(np.in1d(['gx','gy'], data.ch_names)):
            raise ValueError('Both gaze channels (gx, gy) must be present.')
            
        ## Copy data.
        data = data.data[:,np.in1d(data.ch_names, ['gx','gy'])].copy()
        if data.ndim == 2: data = np.expand_dims(data, 0)
        else: data = data.swapaxes(1,2)
        
    ## Data handling (all else).
    else:
                
        ## Error-catching: force gx and gy to be present.
        if np.shape(data)[-1] != 2:
            raise ValueError('data last dimension must be length 2, i.e. (xdim, ydim)')
           
        ## Copy data.
        data = np.array(data.copy())
        if data.ndim == 2: data = np.expand_dims(data, 0)
            
    ## Collect metadata. Preallocate space.
    n_trials, n_times, n_dim = data.shape
    xd, yd, n_screens = screen.indices.shape    
    aligned = np.zeros(n_trials * n_times)

    ## Unfold screen index variable into the events timeline.
    trials_long = np.repeat(np.arange(1,n_trials+1),n_times)
    screenidx_long = np.squeeze(screenidx[trials_long-1])

    ## Extract row (xdim) and col (ydim) info.
    row, col = np.floor(data.reshape(n_trials*n_times,n_dim)).T

    ## Identify missing data.
    row[np.logical_or(row < 0, row >= screen.xdim)] = np.nan    # Eyefix outside screen x-bound.
    col[np.logical_or(col < 0, col >= screen.ydim)] = np.nan    # Eyefix outside screen y-bound.
    missing = np.logical_or(np.isnan(row), np.isnan(col))

    ## Align fixations for each screen.
    for i in range(n_screens):

        ## Identify events associated with this screen.
        this_screen = (screenidx_long == i+1)

        ## Combine with info about missing data.
        x = np.logical_and(~missing, this_screen)

        ## Align eyefix with screen labels.
        aligned[x] = screen.indices[row[x].astype(int), col[x].astype(int), i]

    return aligned.reshape(n_trials, n_times)

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
    """

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