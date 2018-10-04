import numpy as np

def epoching_moat(raw, info, events, template='Start of Run%s'):
    """Epoch the raw eyetracking data.
    
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
      
    Notes
    -----
    Designed for MOAT dataset collected by Daniel Bennett & Angela Radulescu. 
    Key assumption is that each "Start of Run message" is aligned to the 
    first stimulus onset within that run/block. We only look at last 4 blocks.
    """
    
    ## Define elapsed time relative to eyetracking.
    times = np.arange(0, raw.shape[0] / info.sfreq, 1 / info.sfreq)
    
    ## Define start of blocks relative to elapsed time.
    # We add 3 because we only look at test blocks.
    blocks = np.unique(events[:,0]).astype(int) + 3

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

def set_screen_moat(info):
    """Sets screen and AoIs for MOAT experiment.
    
    Parameters
    ----------
    info : instance of `ScreenInfo`
        Eyetracking acquisition information.
    
    Returns
    -------
    info_with_aoi : instance of `ScreenInfo` with AoIs added.
        Eyetracking acquisition information.
      
    Notes
    -----
    Designed for MOAT dataset collected by Daniel Bennett & Angela Radulescu. 
    """
    
    ## Define AoIs
    aois = np.empty((4,5)) # center x-coord, center y-coord, x-radius, y-radius
    # Large left ellipse
    aois[0] = [400, 400, 200, 400, np.radians(-135)]
    # Large right ellipse
    aois[1] = [1200, 400, 200, 400, np.radians(135)]
    # Small left ellipse
    aois[2] = [400, 400, 100*np.sqrt(2), 200*np.sqrt(2), np.radians(-135)]
    # Small right ellipse
    aois[3] = [1200, 400, 100*np.sqrt(2), 200*np.sqrt(2), np.radians(135)]

    ## Make masks 
    # Create screen sized array with unraveled indices.
    [X,Y] = np.unravel_index(np.arange(info.xdim * info.ydim),(info.xdim, info.ydim))
    # Make mask that keeps upper half of large left ellipse.
    mask1 = np.reshape(X < Y, (info.xdim, info.ydim)).astype(int)
    # Make mask that keeps lower half of large eft ellipse.
    mask2 = np.reshape(X > Y, (info.xdim, info.ydim)).astype(int)
    # Make mask that keeps lower half of large right ellipse.
    mask3 = np.reshape(X < -Y + info.xdim, (info.xdim, info.ydim)).astype(int)
    # Make mask that keeps upper half of large right ellipse.
    mask4 = np.reshape(X > -Y + info.xdim, (info.xdim, info.ydim)).astype(int)

    # Screen 1: whole ellipses
    info.add_ellipsoid_aoi(aois[2,0], aois[2,1], aois[2,2], aois[2,3], aois[2,4], 1)
    info.add_ellipsoid_aoi(aois[3,0], aois[3,1], aois[3,2], aois[3,3], aois[3,4], 1)

    # Screen 2: halved left ellipse, whole right ellipse
    info.add_ellipsoid_aoi(aois[0,0], aois[0,1], aois[0,2], aois[0,3], aois[0,4], 2, mask1)
    info.add_ellipsoid_aoi(aois[0,0], aois[0,1], aois[0,2], aois[0,3], aois[0,4], 2, mask2)
    info.add_ellipsoid_aoi(aois[3,0], aois[3,1], aois[3,2], aois[3,3], aois[3,4], 2)

    # Screen 2: whole left ellipse, halved right ellipse
    info.add_ellipsoid_aoi(aois[2,0], aois[2,1], aois[2,2], aois[2,3], aois[2,4], 3)
    info.add_ellipsoid_aoi(aois[1,0], aois[1,1], aois[1,2], aois[1,3], aois[1,4], 3, mask3)
    info.add_ellipsoid_aoi(aois[1,0], aois[1,1], aois[1,2], aois[1,3], aois[1,4], 3, mask4)

    # Screen 4: halved left ellipse, halved right ellipse
    info.add_ellipsoid_aoi(aois[0,0], aois[0,1], aois[0,2], aois[0,3], aois[0,4], 4, mask1)
    info.add_ellipsoid_aoi(aois[0,0], aois[0,1], aois[0,2], aois[0,3], aois[0,4], 4, mask2)
    info.add_ellipsoid_aoi(aois[1,0], aois[1,1], aois[1,2], aois[1,3], aois[1,4], 4, mask3)
    info.add_ellipsoid_aoi(aois[1,0], aois[1,1], aois[1,2], aois[1,3], aois[1,4], 4, mask4)

    info_with_aoi = info

    return info_with_aoi  
