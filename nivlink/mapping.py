import numpy as np
import pandas as pd 

def map_to_feats(fixations, featmap):
	'''Map fixations (stored in pandas dataframe) to visual task features.

	Parameters
	----------
	fixations : Pandas dataframe 
		One line per fixation where the "AoI" column indicates 
		the AoI corresponding to each fixation 
	featmap	: array, shape (n_trials, n_aois)
		Mapping between AoIs and task features. Each row represents a trial. 
		There are as many columns as there are AoIs 

	Returns
	-------
	mapped_fixations : pandas dataframe with one line per event where the "Feature" column 
		indicates the visual task feature corresponding to each fixation 
	'''

	# Retrieve number of trials and number of fixations
	[n_trials, dummy] = np.shape(featmap)
	[n_fixations, dummy] = fixations.shape

	# Initialize empty feature array.
	feats = np.empty([n_fixations, ])

	# Iterate through trials to get remap AoIs to task features
	# Can this be made any simpler? 
	for t in range(n_trials):
	    
	    # Subset events for this trial.
	    this_trial_data = fixations.loc[fixations.Trial == t+1]
	    
	    # Grab trials. 
	    trials = this_trial_data.index.values.astype(int)
	    
	    # Get AoIs for this trial.
	    this_trial_aoi = this_trial_data.AoI.values.astype(int)
	    
	    # Get AoIs to features map for this trial.
	    this_trial_map = featmap[t]
	    
	    # Convert to task feature space. 
	    this_trial_feats = this_trial_map[this_trial_aoi-1]
	    
	    # Store in array
	    feats[trials, ] = this_trial_feats 

	# Append to dataframe.
	fixations['Feature'] = feats  
	mapped_fixations = fixations

	return mapped_fixations.astype(float)




