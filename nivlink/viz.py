import os
import numpy as np

def plot_raw_blinks(fname, raw, overwrite=True, show=False):
    """Plot detected (and corrected) blinks in raw pupillometry data."""
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import BoxZoomTool, Range1d
    
    ## I/O handling.
    if not os.path.splitext(fname.lower()) == '.html': fname = '%s.html' %fname
    if os.path.isfile(fname): os.remove(fname)        
        
    ## Initialize html file.
    output_file(fname)

    ## Initialize figure.
    x_range = (raw.times.min(), raw.times.max())
    y_range = np.array((np.nanmin(raw.data[:,-1]),np.nanmax(raw.data[:,-1])))
    y_range += 0.1 * np.concatenate([-np.diff(y_range), np.diff(y_range)])
    plot = figure(plot_width=1200, plot_height=300, toolbar_location="right", 
                  x_range = x_range, y_range=y_range, tools="wheel_zoom,reset",
                  x_axis_label='Time (s)', y_axis_label='Pupillometry (au)',
                  title="Blink Correction")
    plot.add_tools(BoxZoomTool(dimensions="width"))

    ## Plot raw pupillometry data.
    plot.line(raw.times, raw.data[:,-1], line_width=2)
    
    ## Plot blink periods.
    X, Y = [], []
    for i, j in raw.blinks:
        X.append(raw.times[i:j])
        Y.append(raw.data[i:j,-1])
        plot.patch(np.hstack((X[-1],X[-1][::-1])), 
                   np.hstack((np.ones_like(X[-1])*3000, np.ones_like(X[-1])*5000)),
                  color='black', alpha=0.07)
    plot.multi_line(X, Y, line_width=2, line_color='orange')    
    
    ## Save/display.
    if show: show(plot)
    
# plot_raw_blinks("test", raw, show=True)

def plot_heatmaps(info_with_aoi, raw_pos_data, contrast, config):
    """Plot raw data heatmaps with overlaid AoIs.
    
    Parameters
    ----------
    info_with_aoi : instance of `Screen`
        Eyetracking acquisition information with AoIs added.
    raw_pos_data: array, shape(xdim, ydim) 
        Raw x,y gaze data. 
    contrast : array, shape(0,1)
        Contrast for histogram plot. 
    config : int
        Specifies the aoi configuration we wish to display. 

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
    from matplotlib.patches import Rectangle

    ## Remove NaNs.
    mask = ~np.any(np.isnan(raw_pos_data),axis=1)
    x = raw_pos_data[mask,0]
    y = raw_pos_data[mask,1]
    x_y = np.column_stack([x,y])

    ## Compute 2D histogram in pixel space. 
    xedges = np.arange(0,info_with_aoi.xdim+1)
    yedges = np.arange(0,info_with_aoi.ydim+1)
    H, xedges, yedges = np.histogram2d(x, y,bins=(xedges, yedges))
    H = H.T

    fig, ax = plt.subplots(1, 1, figsize=(20, 20));
    ax.imshow(H, interpolation='bilinear', cmap=cm.gnuplot, clim=(contrast[0], contrast[1]));
    ax.imshow(info_with_aoi.indices[:,:,config-1].T, alpha = 0.2, cmap = cm.gray)
    ax.set_xticks([]);
    ax.set_yticks([]);

    return fig, ax
