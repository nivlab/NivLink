import numpy as np
import os.path as op
from .edf import edf_read

class Raw(object):
    
    def __init__(self, fname):
        
        ## Read file.
        _, ext = op.splitext(fname.lower())
        if ext == '.edf':
            info, samples, blinks, saccades, messages = edf_read(fname)
                
        ## Store metadata.
        self.info = info
        
        ## Prepare data.
        samples = np.array(samples, dtype=np.float32)
        self.times, self.data = samples[:,0], samples[:,1:]
        del samples
        
        ## Prepare events.
        self.blinks = np.array(blinks, dtype=np.float32)
        self.saccades = np.array(saccades, dtype=np.float32)
        self.messages = np.array(messages, dtype=[('time',np.float32),('message',np.unicode_, 80)])
        
        ## Update times.
        start_time = self.times[0]
        self.times = (self.times - start_time) / self.info['sfreq']
        self.blinks = (self.blinks - start_time) / self.info['sfreq']
        self.saccades = (self.saccades - start_time) / self.info['sfreq']
        self.messages['time'] = (self.messages['time'] - start_time) / self.info['sfreq']
        