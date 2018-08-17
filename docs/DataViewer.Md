
## DataViewer
DataViewer is the licensed eyetracking data analysis software developed by SR Research for analysis of data collected using the Eyelink system. 

### Generating reports with DataViewer
(adapted from https://wiki.psychwire.co.uk/?page_id=187)

Data Viewer can generate a wide variety of different reports which can be used for analyzing your data in different ways. To generate a report, go Analysis>Reports>  and you will see the different report types. 

You can then select which columns to use as output for each report. There are a huge number of different columns that are automatically generated for each report type. 

##### Sample Report
NivLink currently takes Sample Reports as input. 

The Sample Report gives you one row of data for every sample recorded by the eye-tracker during the study. If you have your Eyelink running at 1000Hz, that gives you 1,000 rows of data per second of recording. As a consequence, sample reports are fairly large (~400MB). Sample reports are useful for highly customised setups (e.g., moving displays), if you want to get an idea of millisecond-by-millisecond pupil size (as is the case in pupillometry), or if you want to write your own algorithms for detecting fixations, saccades, blinks etc. You can reduce file size by subselecting only certain columns to include in the report, but to run NivLink you will at the very least need: SAMPLE_MESSAGE, RIGHT_GAZE_X, and RIGHT_GAZE_Y.

##### Fixation Report
The Fixation Report gives you one row of data for every fixation in your study. You will typically need to use the fixation report if you are running visual search/scene perception studies. This is because, for studies of this type, you’ll often want to filter out fixations that coincide with other events, such as display changes, button-press responses, etc.

