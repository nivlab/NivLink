import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ScreenInfo(object):
    
    def __init__(self, xdim, ydim, sfreq):
        '''A ScreenInfo object stores the information relevant to the
        eyetracking acquisition, including the (1) screen dimension,
        (2) stimuli layout, and (3) sampling frequency.
        
        A ScreenInfo object facilitates the drawing and storage of
        areas of interest (AoI) that are used later for binning 
        eyetracking position according to stimulus features of interest.
        
        Parameters
        ----------
        xdim : int
          Screen size along horizontal axis (in pixels).
        ydim : int
          Screen size along vertical axis (in pixels).
        sfreq : float
          Sampling rate of eyetracker.
        
        Attributes
        ----------
        labels : array
          List of unique AoIs.
        indices : array, shape (xdim, ydim)
          Look-up table matching pixels to AoIs.          
        '''
        
        self.sfreq = sfreq        
        self.xdim = xdim
        self.ydim = ydim

        self.labels = []
        self.indices = np.zeros((xdim,ydim))
        
    def add_rectangle_aoi(self, idx, xmin, xmax, ymin, ymax):
        '''Add rectangle area of interest to screen. Accepts absolute
        or fractional [0-1] position. 
        
        Parameters
        ----------
        idx : int
          Unique label for AoI.
        xmin, ymin : int or float
          Coordinates of top-left corner of AoI.
        xmax, ymax : int or float
          Coordinates of bottom-right corner of AoI.
          
        Returns
        -------
        None : `indices` and `labels` modified in place.
          
        '''
        if idx < 1: raise ValueError('Index must be greater than 0!')
        xmin, xmax = [int(self.xdim * x) if isinstance(x,float) else x for x in [xmin,xmax]]
        ymin, ymax = [int(self.ydim * y) if isinstance(y,float) else y for y in [ymin,ymax]]
        
        self.indices[xmin:xmax,ymin:ymax] = idx
        self.labels.append(idx)
        
    def plot_aoi(self, height=3, ticks=False, cmap=None):
        '''Plot areas of interest.
        
        Parameters
        ----------
        height : float
          Height of figure (in inches).
        ticks : bool
          Include axis ticks.
        cmap : matplotlib.cm object
          Colormap. Defaults to ListedColorMap.
          
        Returns
        -------
        fig, ax : plt.figure
          Figure and axis of plot.
        '''
        
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