import numpy as np

def epoching_moat(messages, data, info, events):
    """Epoch the raw eyetracking data. This function has ragged array support.
       Can deprecate once we switch to NivLink 2.0.
    
    Parameters
    ----------
    messages: array, shape (n_times, 1) 
        Array containing messages fromt the raw eyetracking file
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

    ## 0-indexing (blocks start at 0).
    events[:,0] -= 1    

    ## Identify block starts (in samples).
    block_onsets, = np.where([True if msg.startswith('Start') else False for msg in messages])
    block_onsets = block_onsets[-4:]

    ## Convert events from seconds to samples.
    blocks, raw_index = events[:,0].astype(int), events[:,1:]    # Divvy up blocks & events.
    raw_index[:,1] += raw_index[:,0]                             # Convert duration to offset.
    raw_index = (raw_index * info.sfreq).astype(int)             # Convert time to samples.
    raw_index = (raw_index.T + block_onsets[blocks]).T           # Add block offsets to samples.

    ## Build epochs.
    n_trials = raw_index.shape[0]
    n_times = np.diff(raw_index).max()
    epochs = np.ones((n_trials, n_times, 2)) * np.nan
    for n, (r1, r2) in enumerate(raw_index): epochs[n,:(r2-r1)] = data[r1:r2]
    epochs = np.ma.masked_invalid(epochs)

    return epochs.astype(float)

def set_custom_centers(info, raw_data_pos):
    """Customize AoI centers for a particular subject. 
    
    Parameters
    ----------
    info : instance of `ScreenInfo`
        Eyetracking acquisition information.
    raw : array, shape (n_times, 2)
        Raw eyetracking data without message column.
    
    Returns
    -------
    custom_ctr_left : array, shape (1, 2) 
        New x, y coordinates of left ellipse 
    custom_ctr_right : array, shape (1, 2)
        New x, y coordinates of right ellipse
    """

    ## Remove NaNs from eye position data.
    mask = ~np.any(np.isnan(raw_data_pos),axis=1)
    x = raw_data_pos[mask,0]
    y = raw_data_pos[mask,1]
    
    ## Compute 2D histogram in pixel space. 
    xedges = np.arange(0,info.xdim+1)
    yedges = np.arange(0,info.ydim+1)
    H, xedges, yedges = np.histogram2d(x, y,bins=(xedges, yedges))
    H = H.T

    ## Determine custom AoI centers.
    center_mask = 300;
    H_left = np.zeros(H.shape); H_right = np.zeros(H.shape); 
    H_left[:,np.arange(1,int(info.xdim/2)-center_mask)] = H[:,np.arange(1,int(info.xdim/2)-center_mask)]
    H_right[:,np.arange(int(info.xdim/2)+center_mask,int(info.xdim))] = H[:,np.arange(int(info.xdim/2)+center_mask,int(info.xdim))]
    # Get indices of maximum each half.
    max_ind_left = np.unravel_index(np.argmax(H_left, axis=None), H_left.shape)
    max_ind_right = np.unravel_index(np.argmax(H_right, axis=None), H_right.shape)
    # Recode as custom AoI center.
    custom_ctr_left = (max_ind_left[1], max_ind_left[0])
    custom_ctr_right = (max_ind_right[1], max_ind_right[0])
    
    return custom_ctr_left, custom_ctr_right

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

def make_screen_idx(n_trials, featmap):
    """Sets screen and AoIs for MOAT experiment.
    
    Parameters
    ----------
    n_trials : int
        Number of trials in the experiment. 

    featmap : array, shape (n_trials, n_aois)
        Key for mapping AoIs to cues.

    Returns
    -------
    screen_idx : array, shape(n_trials, 1)
        Vector defining which AoI configuration present 
        on each trial.  
    """

    screen_idx = np.zeros((n_trials,1))
    empty_count = np.count_nonzero(featmap == 99, axis = 1)  
    idx_aoi1_filled = featmap[:,0] != 99
    idx_aoi2_filled = featmap[:,1] != 99

    idx_screen_1 = empty_count == 4
    idx_screen_2 = np.logical_and(empty_count == 3, idx_aoi2_filled)
    idx_screen_3 = np.logical_and(empty_count == 3, idx_aoi1_filled)
    idx_screen_4 = empty_count == 2

    screen_idx[idx_screen_1] = 1
    screen_idx[idx_screen_2] = 2
    screen_idx[idx_screen_3] = 3
    screen_idx[idx_screen_4] = 4

    return screen_idx

def remap_aois(fixations):
    """Recode AoIs. 
    
    Parameters
    ----------
    fixations : dataframe
       Contains eye position data mapped to NivLink AoIs (1-12). 

    Returns
    -------
    fixations : dataframe
       Contains eye position data mapped to MOAT recoded AoIs (1-6). 

    Notes
    -------
    Key: 
    1=6=[1], 2=5=[2], 3=9=[3], 4=10=[4], 7=11=[5], 8=12=[6]    
    """

    fixations = fixations.replace({'AoI': 5}, 2)
    fixations = fixations.replace({'AoI': 9}, 3)
    fixations = fixations.replace({'AoI': 10}, 4)
    fixations = fixations.replace({'AoI': 7}, 5)
    fixations = fixations.replace({'AoI': 11}, 5)
    fixations = fixations.replace({'AoI': 8}, 6)
    fixations = fixations.replace({'AoI': 12}, 6)

    return fixations

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
