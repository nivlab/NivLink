import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import measurements

def _moving_average(data_set, window):
    """Simple moving average."""
    weights = np.ones(window) / window
    return np.convolve(data_set, weights, mode='valid')

def correct_blinks(self, interp='nan', window=0.05):
        """Correct blinks in pupillometry data.
        
        Parameters
        ----------
        interp : str | int
            Specifies the kind of interpolation as a string ('nan','linear', 'nearest', 
            'zero', 'slinear', ‘quadratic’, ‘cubic’) or as an integer specifying the 
            order of the spline interpolator to use. If 'nan', blink periods are replaced
            by NaNs. See scipy.interpolate.interp1d for details.
        window : int | float
            Interpolation window. If int, number of samples. If float, 
            number of seconds. Ignored if interp = 'nan'.
        """
        if isinstance(window, float): window = int(window * self.info['sfreq'])
        if interp == 'nan': window = 0
        assert isinstance(window, int)
        
        for i, j in deepcopy(self.blinks):
            
            ## Mask blink.
            self.data[i:j,-1] = np.nan
            if interp == 'nan': continue
                
            ## Perform interpolation.
            i, j = i - window, j + window
            y = self.data[i:j,-1]
            x = np.arange(j-i)
            mask = np.invert(np.isnan(y))
            f = interp1d(x[mask], y[mask], interp)
            self.data[i:j,-1] = f(x)
        
    def detect_blinks(self, min_dist=0.1, window=0.01, overwrite=True, verbose=False):
        """Detect blinks in pupillometry data.
        
        Parameters
        ----------
        min_dist : float
            Minimum length (in seconds) between two successive blinks. 
            If shorter, blinks are merged into single event.
        window : int | float
            Length of smoothing window. If int, number of samples. If float, 
            number of seconds.
        overwrite : bool
            Overwrite EyeLink-detected blinks. If False, returns blinks.
        verbose : bool
            Print number of detected blinks.
        
        Notes
        -----
        Blink detection algorithm presented in Hershman et al. (2018). Briefly, 
        the algorithm:
        
        1. Detects blink periods (pupil = 0).
        2. Merges nearby blinks, defined as occurring within :code:`min_dist`.
        3. Extends blink period to start/end of slope. 
        
        See paper for details. Usually preferable to EyeLink default blinks. 
        
        References
        ----------
        [1] Hersman et al. (2018). https://doi.org/10.3758/s13428-017-1008-1.
        """
        if isinstance(window, float): window = int(window * self.info['sfreq'])
        assert isinstance(window, int)
        assert min_dist >= 0
        
        ## STEP 1: Identify blinks.
        blinks, n_blinks = measurements.label(self.data[:,-1] == 0)
        onsets = measurements.minimum(np.arange(blinks.size), labels=blinks, index=np.arange(n_blinks)+1)
        offsets = measurements.maximum(np.arange(blinks.size), labels=blinks, index=np.arange(n_blinks)+1)
        blinks = np.column_stack((onsets, offsets))
        
        ## STEP 2: Merge blinks.
        while True:

            ## Check if any blinks within min dist.
            adjacency = np.diff(self.times[blinks.flatten()])[1::2] < min_dist
            if not np.any(adjacency): break

            ## Update blinks.
            i = np.argmax(adjacency)
            blinks[i,-1] = blinks[i+1,-1]            # Store second blink offset as first.
            blinks = np.delete(blinks, i+1, axis=0)  # Remove redundant blink.

        ## STEP 3: Extend windows.
        for i, (onset, offset) in enumerate(blinks):

            ## Extend onset.
            smooth = _moving_average(self.data[onset-window*10:onset,-1], window)
            first_samp = np.argmax(np.diff(smooth)[::-1] >= 0)
            blinks[i,0] -= first_samp

            ## Extend offset.
            smooth = _moving_average(self.data[offset:offset+window*10:,-1], window)
            first_samp = np.argmax(np.diff(smooth) <= 0)
            blinks[i,1] += first_samp
            
        if verbose: print('%s blinks detected.' %blinks.shape[0])
        if overwrite: self.blinks = blinks
        else: return blinks