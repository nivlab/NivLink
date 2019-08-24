Contributing
------------

Goals
^^^^^

NivLink is a lightweight toolkit originally built to support eyetracking experiments in the Niv Lab. The overarching goal is to develop fast, flexible, and robust software for preprocesisng EyeLink data collected as part of ongoing and future experiments. The design goals for this package are straightforward:

* Do not duplicate existing software. If there is software available for preprocessing your data as needed, use that instead. This improves replicability. 

* Do not write superfluous functions. If a routine can be accomplished in 1-2 lines of code (e.g. removing short fixations), it does not need its own function. This reduces clutter.

* Do the bare minimum. In preprocessing, preserve the data as best as possible leaving data reduction and transformation for postprocessing. When transformation is necessary, be explicit in the justification and assumptions.

With that in mind, we are open to contributions. NivLink is meant to benefit anyone analyzing EyeLink data, and as such, we seek enhancements that will benefit users of the package. Before starting new code, we highly recommend opening an issue on `NivLink GitHub <https://github.com/nivlab/NivLink>`_ to discuss potential changes.


Package layout
^^^^^^^^^^^^^^

NivLink follows a simple organizational layout. The ``nivlink`` package itself is comprised of two primary modules, ``Screen`` and ``preprocessing``, and secondary modules particular to specific datasets, e.g. ``fht``.

Functions for representing experimental stimuli and their associated spatial areas of interest belong in the ``Screen`` module. Functions for preprocessing eyetracking data that are likely to generalize across experiments belong in the ``preprocessing`` module.

Preprocessing functions required by and particular to specific datasets (e.g. ``fht_epoching``) should be stored in a separate ``*.py`` file. This organizational structure helps to demarcate which tools are appropriate for some or all of the datasets collected by the lab.


Coding Guidelines
^^^^^^^^^^^^^^^^^

* All public functions must have documentation that explains what the code does, what its parameters mean, and what its return values can be. Docstrings should be formatted according to the `NumPy docstring <https://numpydoc.readthedocs.io/en/latest/format.html>`_ standard (`example <http://www.sphinx-doc.org/en/stable/ext/example_numpy.html>`_).

* All code contributed to ``Screen`` and ``preprocessing`` must have unit tests executable with pytest and Travis-CI. Unit tests should be put in a ``test_*.py`` file in the test folder.

* Demonstrations of how to use new functions is strongly encouraged and should be put in the `demos folder <https://github.com/nivlab/NivLink/tree/master/demos>`_.
