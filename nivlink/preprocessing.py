import numpy as np
from pandas import DataFrame
from scipy.ndimage import measurements
    
def align_to_aoi(epochs, info):
    '''Align eyetracking data to areas of interest. Please see notes.
    
    Parameters
    ----------
    epochs : array, shape (n_trials, n_times, n_dim)
      Epoched eyetracking timeseries data. Last dimension
      must be (xdim, ydim).
    info : instance of `ScreenInfo`
      Eyetracking acquisition information.
    
    Returns
    -------
    aligned : array, shape (n_trials, n_times)
      Eyetracking timeseries aligned to areas of interest.  
    
    Notes
    -----
    The alignment step makes two critical assumptions during processing:
    (1) Eyetracking positions are rounded to the nearest pixel.
    (2) Eyetracking positions outside (xdim, ydim) are set to NaN.
    '''
    
    if not epochs.ndim == 3:
        raise ValueError('epochs must be shape (n_trials, n_times, n_dim)')
    if not epochs.shape[-1] == 2:
        raise ValueError('epochs last dimension must be length 2, i.e. (xdim, ydim)')
    
    ## Collect metadata. Preallocate space.
    n_trials, n_times, n_dim = epochs.shape    
    aligned = np.zeros(n_trials * n_times)
    
    ## Extract row (xdim) and col (ydim) info.
    row, col = epochs.reshape(n_trials*n_times,n_dim).T
    
    ## Identify missing data.
    row[np.logical_or(row < 0, row > info.xdim)] = np.nan    # Eyefix outside xdim.
    col[np.logical_or(col < 0, col > info.ydim)] = np.nan    # Eyefix outside ydim.
    missing = np.logical_or(np.isnan(row), np.isnan(col))
    
    ## Align eyefix with screen labels.
    aligned[~missing] = info.indices[row[~missing].astype(int), col[~missing].astype(int)]
    
    return aligned.reshape(n_trials, n_times)

def compute_fixations(aligned, info, labels=None):
    '''Compute fixations from aligned timeseries. Fixations are defined
    as contiguous samples of eyetracking data aligned to the same AoI.
    
    Parameters
    ----------
    aligned : array, shape (n_trials, n_times)
      Eyetracking timeseries aligned to areas of interest.  
    info : instance of `ScreenInfo`
      Eyetracking acquisition information.
    labels : list
      List of areas of interest to include in processing. Defaults to
      info.labels.
    
    Returns
    -------
    fixations : pd.DataFrame
      Pandas DataFrame where each row details the (Trial, AoI, 
      Onset, Offset, Duration) of the fixation.
    '''
    
    ## Define labels list.
    if labels is None: labels = info.labels

    ## Append extra timepoint to end of each trial. This prevents clusters across
    ## successive trials.
    n_trials, n_times = aligned.shape
    aligned = np.hstack([aligned, np.zeros((n_trials,1))]).flatten()

    ## Precompute trial and timing info.
    trials = np.repeat(np.arange(n_trials),n_times+1) + 1
    times = np.repeat(np.arange(n_times+1.),n_trials).reshape(n_trials,n_times+1,order='F').flatten() 
    times /= info.sfreq

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

    df = df.sort_values(['Trial','Onset'])
    df['Duration'] = df.Offset - df.Onset
    
    return df