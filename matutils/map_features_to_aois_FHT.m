
% Example of an experiment specific script which generates a "feature map" file
% which NivLink needs for the mapping step. Remaps the task features (9 in this case) to the 
% position they were presented on screen.   

% Load behavioral data. 
behav_data_path = '/Users/angelar/Dropbox/research/attention_modeling/data/NewmanThesis/AllData.mat';
load(behav_data_path);

% Grab relevant subject variables. 
subj_num = 1; 
n_trials = length(AllData{subj_num,2}.b.RT);
this_sub_stim = AllData{subj_num,2}.b.Stimuli;
this_sub_row_order = AllData{subj_num,2}.b.RowOrder(1,:);

% Pre-allocate array for trialwise AOI mapping. 
features_aoi_map = NaN(n_trials,9);

% Iterate over trials.
for t = 1:n_trials
    
    % Grab the task features and convert to 1-9 feature notation. 
    % 1-3 are Faces, 4-6 are Houses, 7-9 are Tools.
    stimuli_trial = squeeze(this_sub_stim(t,:,:))';
    hlp = [zeros(1,3); 3*ones(1,3); 6*ones(1,3)];
    features_trial_aux = stimuli_trial + hlp; 
    
    % Use row order to remap dimensions to their correct position.
    features_trial = NaN(3,3);
    for f = 1:3
        features_trial(f,:) = (features_trial_aux(find(this_sub_row_order == f),:));
    end
    
    % Map features to AOIs.
    % AOI map for FHT experiment: 
    % [1][4][7] 
    % [2][5][8]
    % [3][6][9]
    
    features_aoi_map(t,:) = [features_trial(:,1)' features_trial(:,2)' features_trial(:,3)']; 
    
end

save('featmap.mat','features_aoi_map');