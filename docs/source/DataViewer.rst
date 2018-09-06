DataViewer
----------
DataViewer is the licensed eyetracking data analysis software developed by SR Research for analysis of data collected using the EyeLink system. 

Generating reports with DataViewer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Adapted from `Psychwire <https://wiki.psychwire.co.uk/?page_id=187>`_)

Given an EyeLink recording, DataViewer can generate a large variety of sample reports for subsequent use in analysis. To view the available report types in DataViewer, go to:

.. code::

    Analysis > Reports  
    
You can then select which types of data to output into the report. Each report type in DataViewer has many different data types available for output into a sample report.

Sample Report
^^^^^^^^^^^^^
Sample reports outputted by DataViewer are structured such that there is one row of data for every sample recorded by the eye-tracker during the experiment. If the Eyelink is recording at 1000 Hz, for example, the sample report will have 1000 rows of data per second of recording. As a consequence, sample reports are typically large (~400MB). Though memory intensive, this level of granularity is necessary for measuring dynamic processes elicated by complex experimental designs (e.g. moving displays); for measuring fast changes in pupil size; and/or for advanced algorithms for detecting fixations, saccades, blinks, etc. 

NivLink currently requires sample reports as input. At present, however, NivLink requires only the following three data types: ``SAMPLE_MESSAGE``, ``RIGHT_GAZE_X``, and ``RIGHT_GAZE_Y``. As such, disk space can be conserved by subselecting only these three columns for inclusion in the sample report.

Fixation Report
^^^^^^^^^^^^^^^
Similar to smaple reports, fixation reports are comprised of one row of data for each detected eye fixation in an experiment. Though not strictly necessary, fixation reports are invaluable for visual search and scene perception studies. This is because these types of studies benefit from filtering fixations coinciding with stimulus events, such as display changes, button-press responses, etc.