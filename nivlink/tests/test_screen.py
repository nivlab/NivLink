import numpy as np
from nivlink import Screen

'''
NOTE: We do not test any epoching functions. This would require storing
sample data as part of the package, which we currently do not want to do.
If sample data is submitted in the future, this may change.
'''

def test_workflow():

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Test ScreenInfo.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Define metadata.
    xdim, ydim, n_screens = 100, 100, 1

    ## Initialize Screen object.
    info = Screen(xdim, ydim, n_screens)

    assert info.xdim == xdim            # Test screen storing values properly.
    assert info.ydim == ydim            # Test screen storing values properly.
    assert len(info.labels) == 0        # Test screen initialized to empty list.
    assert np.all(info.indices == 0)    # Test screen initialized to all zeros.
    assert np.all(np.equal(info.indices.shape, (xdim,ydim,n_screens)))

    ## Add areas of interest.
    info.add_rectangle_aoi(0, xdim/2, 0, ydim)
    info.add_rectangle_aoi(xdim/2, xdim, 0, ydim)

    assert np.all(info.indices[:xdim//2] == 1)    # Test screen indices update.
    assert np.all(info.indices[xdim//2:] == 2)    # Test screen indices update.
    assert np.all(np.equal(info.labels, [1,2]))   # Test screen indices update.