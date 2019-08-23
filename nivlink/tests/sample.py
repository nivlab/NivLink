import numpy as np
from numpy.random import uniform as runif
np.random.seed(47404)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define metadata.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define metadata.
n_trials = 200

## Define timing info.
tmin = -0.5
tmax = 3.0
sfreq = 100

## Define pupillometry info.
baseline = 4000
noise = lambda size: np.random.normal(0,baseline * 5e-5,size)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define useful functions.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def inv_logit(arr):
    return 1 / (1 + np.exp(-arr))

def pupil_sim(t, h1, t1, d1, h2, t2, d2):
    return h1 * inv_logit((t-t1)/d1) + h2 * inv_logit((t-t2)/d2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Simulate metadata.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define info.
info = dict(sfreq = sfreq)

## Define messages.
onsets = np.arange(n_trials) * int((tmax - tmin) * sfreq) + int(-tmin * sfreq)
messages = ['Cond %s' %(i % 2 + 1) for i in np.arange(n_trials)]
messages = np.array(list(zip(onsets,messages)), dtype=[('sample',int),('message',np.unicode_, 80)])

## Define blinks.
onsets = runif(tmin + 0.1, tmax - 0.5, n_trials)
offsets = onsets + 0.02 + runif(0.05, 0.15, n_trials)
blinks = np.column_stack([onsets, offsets])
blinks = (blinks.T + np.arange(n_trials) * (tmax - tmin)).T
blinks = ((blinks - tmin) * sfreq).astype(int)

## Define saccades.
saccades = np.empty((0,2))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Simulate pupillometry.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define epoch times.
times = np.arange(-0.5,3,1/sfreq).round(2)

## Simulate pupillometry.
pupil = np.zeros((n_trials,times.size))
for i in range(n_trials):
    pupil[i] = pupil_sim(times, 1, runif(0.70,0.80), runif(0.15,0.25), 
                               -1, runif(2.20,2.30), runif(0.05,0.15))
    pupil[i] /= pupil[i].max()
    
## Scale pupillometry data.
pupil[::2] *= 1.5                              # Half trials 1.5x amplitude
pupil = baseline + (pupil * baseline / 100)    # Scale as PSC
pupil += noise(pupil.shape)                    # Add "fuzzy" noise
pupil = pupil.flatten()               

## Corrupt pupillometry with blinks.
for i, j in blinks: pupil[i:j] = 0
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Simulate eyetracking.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## TODO: more realistic eyetracking.

eyetrack = np.random.randint(0,1000,(pupil.size,2))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define times & data.
times = np.arange(pupil.size) / sfreq
data = np.column_stack([eyetrack,pupil])
ch_names = ['gaze_x','gaze_y','pupil']

## Save.
np.savez_compressed('sample', info=info, ch_names=ch_names, times=times, data=data, 
                    blinks=blinks, saccades=saccades, messages=messages)