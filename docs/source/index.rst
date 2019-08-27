NivLink
=======
NivLink is an open-source Python package developed in the `Niv Lab <https://www.princeton.edu/~nivlab/>`_ to preprocess EyeLink eyetracking data. 

Example
^^^^^^^

.. code-block:: python

    from nivlink import Raw, Epochs, Screen
    from nivlink.gaze import align_to_aoi, compute_fixations

    ## Load EDF file.
    raw = Raw('example.edf')

    ## Find events.
    events = raw.find_events('STARTTRIAL')

    ## Epochs.
    epochs = Epochs(raw, events, tmin=0, tmax=1, picks='gaze', eyes='LEFT')

    ## Initialize screen.
    screen = Screen(1600, 1200)
    screen.add_rectangle_aoi(0.25, 0.75, 0.25, 0.75)

    ## Aligned.
    aligned = align_to_aoi(epochs, screen)

    compute_fixations(aligned, epochs.times)

For more detailed example use cases, please see the `demos <https://github.com/nivlab/NivLink/tree/master/demos>`_ folder.


License
^^^^^^^

The project is licensed under the MIT license.
