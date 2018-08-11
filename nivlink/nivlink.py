import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import measurements

class ScreenInfo(object):
    
    def __init__(self, xdim, ydim, sfreq):
        '''Screen object
        
        Parameters
        ----------
        xdim : int
          Screen size along horizontal axis (pixels).
        ydim : int
          Screen size along vertical axis (pixels).
        sfreq : float
          Sampling rate of eyetracker.
        '''
        
        self.sfreq = sfreq        
        self.xdim = xdim
        self.ydim = ydim

        self.indices = np.zeros((xdim,ydim))
        self.features = []
        
    def add_rectangle_aoi(self, idx, xmin, xmax, ymin, ymax):
        '''Add rectangle area of interest to screen. Accepts absolute
        or fractional (relative) position. Modifies indices in place.
        
        Parameters
        ----------
        xmin, ymin : int or float
          Top-left corner of AOI.
        xmax, ymax : int or float
          Bottom-right corner of AOI.
        '''
        if idx < 1: raise ValueError('Index must be greater than 0!')
        xmin, xmax = [int(self.xdim * x) if isinstance(x,float) else x for x in [xmin,xmax]]
        ymin, ymax = [int(self.ydim * y) if isinstance(y,float) else y for y in [ymin,ymax]]
        
        self.indices[xmin:xmax,ymin:ymax] = idx
        self.features.append(idx)
        
    def plot_screen(self, height=3, ticks=False, cmap=None):
        '''Plot screen information.'''
        
        ## Initialize plot.
        ratio = float(self.xdim) / float(self.ydim)
        fig, ax = plt.subplots(1,1,figsize=(ratio*height, height))            
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
            
        ## Initialize colormap.
        if cmap is None:
            
            colors = ['k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            cmap = ListedColormap(colors[:np.unique(self.indices).size])
            
        ## Plotting.
        cbar = ax.imshow(self.indices.T, cmap=cmap, aspect='auto')
        fig.colorbar(cbar, cax, ticks=np.arange(len(cmap.colors)))
        if not ticks: ax.set(xticks=[], yticks=[])
        
        return fig, ax
    
def to_features(epochs, info):
    '''NEED TO CHECK: (1) is masking appropriate? (2) setting to int robust?'''
    assert epochs.ndim == 3
    assert epochs.shape[-1] == 2
    
    ## Collect metadata. Preallocate space.
    n_trials, n_samples, n_dim = epochs.shape    
    features = np.zeros(n_trials * n_samples)
    
    ## Extract row (xdim) and col (ydim) info.
    row, col = epochs.reshape(n_trials*n_samples,n_dim).T
    
    ## Identify missing data.
    row[np.logical_or(row < 0, row > info.xdim)] = np.nan    # Eyefix outside xdim.
    col[np.logical_or(col < 0, col > info.ydim)] = np.nan    # Eyefix outside ydim.
    missing = np.logical_or(np.isnan(row), np.isnan(col))
    
    ## Align eyefix with screen features.
    features[~missing] = info.indices[row[~missing].astype(int), col[~missing].astype(int)]
    
    return features.reshape(n_trials, n_samples)

def compute_fixations(ts_feat, sfreq, features=None):
    
    ## Define features list.
    if features is None: features = np.array([f for f in np.unique(ts_feat) if f])

    ## Append extra timepoint to end of each trial. This prevents clusters across
    ## successive trials.
    n_trials, n_times = ts_feat.shape
    ts_feat = np.hstack([ts_feat, np.zeros((n_trials,1))]).flatten()

    ## Precompute trial and timing info.
    trials = np.repeat(np.arange(n_trials),n_times+1) + 1
    times = np.repeat(np.arange(n_times+1.),n_trials).reshape(n_trials,n_times+1,order='F').flatten() 
    times /= sfreq

    ## Preallocate space.
    df = DataFrame([], columns=('Trial','SpatialFeature','tmin','tmax'))

    for feature in features:

        ## Identify clusters.
        clusters, n_clusters = measurements.label(ts_feat == feature)

        ## Identify cluster info.
        trial = measurements.minimum(trials, labels=clusters, index=np.arange(n_clusters)+1)
        tmin = measurements.minimum(times, labels=clusters, index=np.arange(n_clusters)+1)
        tmax = measurements.maximum(times, labels=clusters, index=np.arange(n_clusters)+1)

        ## Append to DataFrame.
        df = df.append(DataFrame(np.column_stack((trial, np.ones_like(trial)*feature, tmin, tmax)), columns=df.columns))

    df = df.sort_values(['Trial','tmin'])
    df['Duration'] = df.tmax - df.tmin
    
    return df