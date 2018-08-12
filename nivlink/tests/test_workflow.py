import numpy as np
from nivlink import ScreenInfo, align_to_aoi, compute_fixations

'''
NOTE: We do not test any epoching functions. This would require storing
sample data as part of the package, which we currently do not want to do.
If sample data is submitted in the future, this may change.
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Test ScreenInfo.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define metadata.
xdim, ydim, sfreq = 100, 100, 1

## Initialize ScreenInfo object.
info = ScreenInfo(xdim, ydim, sfreq)

assert info.xdim == xdim            # Test screen storing values properly.
assert info.ydim == ydim            # Test screen storing values properly.
assert info.sfreq == 1              # Test screen storing values properly.
assert len(info.labels) == 0        # Test screen initialized to empty list.
assert np.all(info.indices == 0)    # Test screen initialized to all zeros.
assert np.all(np.equal(info.indices.shape, (xdim,ydim)))

## Add areas of interest.
info.add_rectangle_aoi(1, 0, xdim/2, 0, ydim)
info.add_rectangle_aoi(2, xdim/2, xdim, 0, ydim)

assert np.all(info.indices[:xdim//2] == 1)    # Test screen indices update.
assert np.all(info.indices[xdim//2:] == 2)    # Test screen indices update.
assert np.all(np.equal(info.labels, [1,2]))   # Test screen indices update.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Test preprocessing.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Simulate data.
n_times = 50
epochs = np.array([np.repeat([0.25*xdim,ydim/2], n_times).reshape(n_times,2,order='F'),
                   np.repeat([0.75*xdim,ydim/2], n_times).reshape(n_times,2,order='F')])

## Align data to areas of interest.
aligned = align_to_aoi(epochs, info)

assert np.all(np.in1d(aligned, info.labels))
assert np.all(np.equal(aligned.shape, [2,n_times]))

## Compute fixations.
fixations = compute_fixations(aligned, info)

assert np.all(fixations['AoI'] == [1,2])
assert np.all(fixations['Duration'] == n_times - info.sfreq)