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

    ## Recode block in events file (these are only the test blocks).
    events[:,0] = events[:,0] + int(3)

    ## Define elapsed time relative to eyetracking.
    times = np.arange(0, raw.shape[0] / info.sfreq, 1 / info.sfreq)
    
    ## Define start of blocks relative to elapsed time.
    # We add 3 because we only look at test blocks.
    blocks = np.unique(events[:,0]).astype(int) + 3
    # Where does each block start in the raw datastream?
    indices = [(raw[:,0]==template %i).argmax() for i in blocks]
    # Zero array defining time relative to eyetracking. 
    block_starts = np.zeros_like(times)
    # Add the block start marker to the corresponding position in the raw eyetracking.
    block_starts[indices] = blocks
    
    ## Offset trial starts by block offsets.
    # This puts trial starts on the raw eyetracking timeline.
    events = events.copy()
    for block, offset in zip(blocks, times[block_starts.nonzero()]):
        # For all trial starts in each block, add the corresponding raw eyetracking start time 
        events[events[:,0]==block, 1] += offset
    # The end result in column 2 should be the timestamp relative to the first sample of the raw data,
    # which is exactly the first column in the events matrix expected by the v0.2 epoching function. 
    # To switch to ragged arrays, all we should need to do right now is add the baseline column 
    # (all 0s in this case) and the variable trial duration column.     
 
    ## Redefine trial onset as index corresponding to elapsed time.
    onsets = np.array([np.argmax(times > t) for t in events[:,1]])
    offsets = (onsets + info.sfreq * events[:,-1]).astype(int)
    
    ## Epoching.
    epochs = np.array([raw[i:j+1, 1:] for i,j in np.column_stack([onsets,offsets])])

    return epochs.astype(float)

def set_screen_moat(info, custom_ctr_left=None, custom_ctr_right=None):
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
    Designed for MOAT dataset collected by Daniel Bennett & Angela Radulescu. Assumes
    a fixed 135 degree rotation for each ellipse.  
    """

    import math

    ## Define ellipse centers
    if custom_ctr_left is None: 
        ctr_left = (400,400) 
    else: 
        ctr_left = custom_ctr_left
    
    if custom_ctr_right is None: 
        ctr_right = (1200,400) 
    else: 
        ctr_right = custom_ctr_right

    ## Define AoIs
    aois = np.empty((4,5)) # center x-coord, center y-coord, x-radius, y-radius
    # Large left ellipse
    aois[0] = [ctr_left[0], ctr_left[1], 200, 400, np.radians(-135)]
    # Large right ellipse
    aois[1] = [ctr_right[0], ctr_right[1], 200, 400, np.radians(135)]
    # Small left ellipse
    aois[2] = [ctr_left[0], ctr_left[1], 100*np.sqrt(2), 200*np.sqrt(2), np.radians(-135)]
    # Small right ellipse
    aois[3] = [ctr_right[0], ctr_right[1], 100*np.sqrt(2), 200*np.sqrt(2), np.radians(135)]

    ## Make masks 
    # Define slope and intercept for inequality line
    slope_left = math.sin(math.radians(45)) / math.cos(math.radians(45))
    slope_right = math.sin(math.radians(-45)) / math.cos(math.radians(-45))
    int_left = ctr_left[1] - slope_left * ctr_left[0]
    int_right = ctr_right[1] - slope_right * ctr_right[0]
    # Create screen sized array with unraveled indices.
    # This gives us the cartesian coordinate grid in pixel space
    [X,Y] = np.unravel_index(np.arange(info.xdim * info.ydim),(info.xdim, info.ydim))
    # Make mask that keeps upper half of large left ellipse.
    mask1 = np.reshape(slope_left * X + int_left < Y, (info.xdim, info.ydim)).astype(int)
    # Make mask that keeps lower half of large left ellipse.
    mask2 = np.reshape(slope_left * X + int_left > Y, (info.xdim, info.ydim)).astype(int)
    # Make mask that keeps lower half of large right ellipse.
    mask3 = np.reshape(slope_right * X + int_right > Y, (info.xdim, info.ydim)).astype(int)
    # Make mask that keeps upper half of large right ellipse.
    mask4 = np.reshape(slope_right * X + int_right < Y, (info.xdim, info.ydim)).astype(int)

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

def plot_moat_heatmaps(info_with_aoi, H, contrast):
    """Plot raw data heatmaps with overlaid AoIs.
    
    Parameters
    ----------
    info_with_aoi : instance of `ScreenInfo`
        Eyetracking acquisition information with AoIs added.

    H: array, shape(xdim, ydim) 
        2D histogram of position in pixel space. 
    contrast : array, shape(0,1)
        Contrast for histogram plot. 
      
    Returns
    -------
    fig, ax : plt.figure
        Figure and axis of plot.
      
    Notes
    -----
    Requires matplotlib.
    """      

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 20));
    axes[0,0].imshow(H, interpolation='bilinear', cmap=cm.gnuplot, clim=(contrast[0], contrast[1]));
    axes[0,0].imshow(info_with_aoi.indices[:,:,0].T, alpha = 0.2, cmap = cm.gray)
    axes[0,0].set_xticks([]);
    axes[0,0].set_yticks([]);
    axes[0,0].set_title('Simple vs. simple');

    axes[0,1].imshow(H, interpolation='bilinear', cmap=cm.gnuplot, clim=(contrast[0], contrast[1]));
    axes[0,1].imshow(info_with_aoi.indices[:,:,1].T, alpha = 0.2, cmap = cm.gray)
    axes[0,1].set_xticks([]);
    axes[0,1].set_yticks([]);
    axes[0,1].set_title('Compound vs. simple');

    axes[1,0].imshow(H, interpolation='bilinear', cmap=cm.gnuplot, clim=(contrast[0], contrast[1]));
    axes[1,0].imshow(info_with_aoi.indices[:,:,2].T, alpha = 0.2, cmap = cm.gray)
    axes[1,0].set_xticks([]);
    axes[1,0].set_yticks([]);
    axes[1,0].set_title('Simple vs. compound');

    axes[1,1].imshow(H, interpolation='bilinear', cmap=cm.gnuplot, clim=(contrast[0], contrast[1]));
    axes[1,1].imshow(info_with_aoi.indices[:,:,3].T, alpha = 0.2, cmap = cm.gray)
    axes[1,1].set_xticks([]);
    axes[1,1].set_yticks([]);
    axes[1,1].set_title('Compound vs. compound');

    return fig, axes
