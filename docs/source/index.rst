NivLink
=======
NivLink is an open-source Python package developed in the `Niv Lab <https://www.princeton.edu/~nivlab/>`_ to preprocess EyeLink eyetracking data.

Example
^^^^^^^

.. code-block:: python

    import numpy as np
    from nivlink import ScreenInfo, align_to_aoi, compute_fixations

    ## Load example data.
    eyepos = np.loadtxt('eyepos.txt')

    ## Initialize ScreenInfo object.
    info = ScreenInfo(100, 100, 500)
    info.add_rectangle_aoi(0, 50, 0, 100)
    info.add_rectangle_aoi(50, 100, 0, 100)

    ## Align eye positions to areas of interest (AoI).
    aligned = align_to_aoi(eyepos, info)

    ## Compute fixation times.
    fixations = compute_fixations(aligned, info)

For more detailed example use cases, please see the `demos <https://github.com/nivlab/NivLink/tree/master/demos>`_ folder.


License
^^^^^^^

The project is licensed under the MIT license.
