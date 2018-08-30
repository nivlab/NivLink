
% Example of an experiment specific script which generates an "events" file
% which NivLink needs for the epoching step   
% 
% Returns
% -------
% events_array: each row represents a trial. The first column
% denotes the block. The second column denotes the trial onset
% relative to the block. The third column denotes the trial length.

% Load behavioral data. 
behav_data_path = '/Users/angelar/Dropbox/research/attention_modeling/data/NewmanThesis/AllData.mat';
load(behav_data_path);

% Predefine some variables. 
subj_num = 1; % this corresponds to subject 2 in test_epoching_FHT
n_blocks = 6; % blocks = runs, keeping with terminology in NivLink 
n_trials_per_block = 60;

events_array = NaN(n_blocks * n_trials_per_block,3);

% Iterate over runs/blocks.
for b = 1:n_blocks

    % Grab trials.
    trials = [(b-1)*n_trials_per_block]+1:b*n_trials_per_block;
    
    % StimOn tells us when the stimuli were shown relative to the
    % start of each run/block. Key assumption is that each Start of Run message
    % in the raw eye tracking daya is aligned to the first stimulus onset within that run/block.
    this_block = b*ones(n_trials_per_block,1);
    this_block_trial_onsets = AllData{subj_num,2}.b.StimOn(trials);
    
    % Each trial is 4s long.
    this_block_trial_length = 4*ones(n_trials_per_block,1);
    
    % Record block number, onsets and trial length.
    events_array(trials,1) = this_block;
    events_array(trials,2) = this_block_trial_onsets;
    events_array(trials,3) = this_block_trial_length;

end

save('events.mat','events_array');