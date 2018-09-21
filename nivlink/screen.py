import numpy as np

def _ellipse_in_shape(shape, center, radii, rotation=0.):
    """Generate coordinates of points within ellipse bounded by shape.
    
    Parameters
    ----------
    shape :  iterable of ints
        Shape of the input image.  Must be length 2.
    center : iterable of floats
        (row, column) position of center inside the given shape.
    radii : iterable of floats
        Size of two half axes (for row and column)
    rotation : float, optional
        Rotation of the ellipse defined by the above, in radians
        in range (-PI, PI), in contra clockwise direction,
        with respect to the column-axis.
    
    Returns
    -------
    rows : iterable of ints
        Row coordinates representing values within the ellipse.
    cols : iterable of ints
        Corresponding column coordinates representing values within the ellipse.
    """
    # https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
                + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    return np.nonzero(distances < 1)

def _ellipse(x, y, x_radius, y_radius, shape=None, rotation=0.):
    """Generate coordinates of pixels within ellipse.
    
    Parameters
    ----------
    x, y : int
        Centre coordinate of ellipse.
    x_radius, y_radius : int
        Axes along the x- and y-dimensions. ``(x/x_radius)**2 + (y/y_radius)**2 = 1``.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for ellipses which exceed the image size.
        By default the full extent of the ellipse are used.
    rotation : float, optional (default 0.)
        Set the ellipse rotation (rotation) in range (-PI, PI)
        in contra clock wise direction, so PI/2 degree means swap ellipse axis
    
    Returns
    -------
    xx, yy : ndarray of int
        Pixel coordinates of ellipse. May be used to directly index into an array, 
        e.g. ``img[rr, cc] = 1``.
    
    Notes
    -----
    The ellipse equation::
        ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
        ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1
    Note that the positions of `ellipse` without specified `shape` can have
    also, negative values, as this is correct on the plane. On the other hand
    using these ellipse positions for an image afterwards may lead to appearing
    on the other side of image, because ``image[-1, -1] = image[end-1, end-1]``
    """
    # https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py
    center = np.array([x, y])
    radii = np.array([x_radius, y_radius])
    
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    y_radius_rot = abs(y_radius * np.cos(rotation)) \
                   + x_radius * np.sin(rotation)
    x_radius_rot = y_radius * np.sin(rotation) \
                   + abs(x_radius * np.cos(rotation))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([y_radius_rot, x_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc

class ScreenInfo(object):
    """Container for visual stimuli information.

    Parameters
    ----------
    xdim : int
        Screen size along horizontal axis (in pixels).
    ydim : int
        Screen size along vertical axis (in pixels).
    sfreq : float
        Sampling rate of eyetracker.
    n_screens: int
        Number different screens corresponding to different
        AoI distributions. Defauls to 1.

    Attributes
    ----------
    labels : array
        List of unique AoIs.
    indices : array, shape (xdim, ydim)
        Look-up table matching pixels to AoIs.          
    """
    
    def __init__(self, xdim, ydim, sfreq, n_screens=1):

        
        self.sfreq = sfreq        
        self.xdim = xdim
        self.ydim = ydim
        self.n_screens = n_screens

        self.labels = ()
        self.indices = np.zeros((xdim,ydim,n_screens))
        
    def _update_aoi(self):
        """Convenience function for updating AoI indices."""

        values, indices = np.unique(self.indices, return_inverse=True)

        if np.all(values): indices += 1
        self.indices = indices.reshape(self.xdim, self.ydim, self.n_screens)
        self.labels = tuple(range(1,int(self.indices.max())+1))

    def add_rectangle_aoi(self, xmin, xmax, ymin, ymax, screen_id=1):

        """Add rectangle area of interest to screen. 
        
        Parameters
        ----------
        xmin, ymin : int or float
            Coordinates of top-left corner of AoI. Accepts absolute
            or fractional [0-1] position. 
        xmax, ymax : int or float
          Coordinates of bottom-right corner of AoI.
        screen_id: int
          Which screen to add AoI to. Defaults to 1.
          
        Returns
        -------
        None
            `indices` and `labels` modified in place.
        """

        isfrac = lambda v: True if v < 1 and v > 0 else False
        xmin, xmax = [int(self.xdim * x) if isfrac(x) else int(x) for x in [xmin,xmax]]
        ymin, ymax = [int(self.ydim * y) if isfrac(y) else int(y) for y in [ymin,ymax]]
        
        self.indices[xmin:xmax,ymin:ymax,screen_id - 1] = self.indices.max() + 1
        self._update_aoi()
    
    def add_ellipsoid_aoi(self, x, y, x_radius, y_radius, rotation=0., screen_id=1, mask=None):
        """Generate coordinates of pixels within ellipse.

        Parameters
        ----------
        x, y : int
            Center coordinate of ellipse.
        x_radius, y_radius : int
            Axes along the x- and y-dimensions. 
        rotation : float
            Set the ellipse rotation (rotation) in range :math:`[-\pi, \pi]`
            in contra-clockwise direction, so :math:`\pi / 2` degree means swap ellipse axis.
        screen_id: int
          Which screen to add AoI to. Defaults to 1.
        mask: int    
          Screen-sized array of 0s and 1s used to mask out parts of the display. Defaults to none.

        Returns
        -------
        None
            `indices` and `labels` modified in place.
        """
        # https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py
        xx, yy = _ellipse(x, y, x_radius, y_radius, shape=(self.xdim,self.ydim), rotation=rotation)
        
        if mask is not None: 

            # Flatten mask indices. 
            (xx_mask,yy_mask) = mask.nonzero()
            mi = np.ravel_multi_index(np.array([xx_mask,yy_mask]),(self.xdim, self.ydim))

            # Flatten ellipse indices.     
            ei = np.ravel_multi_index(np.array([xx,yy]),(self.xdim, self.ydim))  
        
            # Intersect indices.
            idx = np.intersect1d(mi,ei)

            # Unravel into 2D again.
            [xxf,yyf] = np.unravel_index(idx,(self.xdim, self.ydim))

        else: 
            xxf = xx
            yyf = yy

        self.indices[xxf,yyf,screen_id - 1] = self.indices.max() + 1
        self._update_aoi()
        
    def plot_aoi(self, screen_id, height=3, ticks=False, cmap=None):
        """Plot areas of interest.
        
        Parameters
        ----------
        screen_id: int
          Set of AoIs to plot. 
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
          
        Notes
        -----
        Requires matplotlib.
        """

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib as matplotlib
        from matplotlib.colors import ListedColormap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ## Initialize plot.
        ratio = float(self.xdim) / float(self.ydim)
        fig, ax = plt.subplots(1,1,figsize=(ratio*height, height))            
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
            
        ## Initialize colormap.
        if cmap is None:
            
            # Collect hex values from standard colormap.
            cmap = cm.get_cmap('tab20', 20)
            
            colors = []
            for i in range(cmap.N):
                rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
                colors.append(matplotlib.colors.rgb2hex(rgb))

            colors = colors[:len(self.labels)]

            # Add black.
            if np.any(self.indices==0): colors = np.insert(colors, 0, 'k')

            # Construct new colormap.
            cmap = ListedColormap(colors)
            
        ## Plotting.
        cbar = ax.imshow(self.indices[:,:,screen_id-1].T, cmap=cmap, aspect='auto', vmin=0, vmax=len(self.labels))
        fig.colorbar(cbar, cax, ticks=np.arange(len(cmap.colors)))
        if not ticks: ax.set(xticks=[], yticks=[])        

        return fig, ax
