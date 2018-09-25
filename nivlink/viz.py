import os

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
    
plot_raw_blinks("test", raw, show=True)