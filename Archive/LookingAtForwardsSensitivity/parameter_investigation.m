close all
clear all
load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\45_epoch_checkpoint_Forwards_Sensitivity.mat") %Latest Forwards Sensitivity Run
%load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\80_Epoch_coninuous.mat") %Last eprop Run
%load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_post_strf_fix_4.mat") %Old Advanced Run

params_names = fields(params);
figure;
cell = 7;
batch_size = size(output,1);
for k = 1:12
    pram = params_names{k};
    subplot(4,3,k)
    plot(squeeze(mean(params.(pram)(:,cell,:),1)),'LineWidth',3); hold on
    for m = 1:batch_size
        if m == 6
            plot(squeeze(params.(pram)(m,cell,:)),'Color',[0.7,0.2,0.5,0.5],'LineWidth',2); hold on
        else
            plot(squeeze(params.(pram)(m,cell,:)),'Color',[0.5,0.5,0.5,0.5]); hold on
        end
    end
    title(pram)
end

%% Construct the experimental PSTHs
data_object = load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat");
sample_rate_hz = 10000;
n_batches = size(output,1);
n_trials = size(output,2);
n_cells = size(output,3);
simlen = size(output,4);
PSTH_gran = 100; 
n_bins = (simlen-mod(simlen,PSTH_gran))/PSTH_gran;
Rasters_data = false(n_cells,n_trials,simlen);
PSTHs_data = zeros(n_cells,n_bins);
used_timesteps = n_bins*PSTH_gran;
bin_edges_seconds = (0:PSTH_gran:used_timesteps)/sample_rate_hz;

for k = 1:n_cells
    tuning = lower(char(string(data_object.all_data(k).tuning_type)));
    if contains(tuning,'contra')
        focus = 1;
    elseif contains(tuning,'45')
        focus = 2;
    elseif contains(tuning,'center')
        focus = 3;
    elseif contains(tuning,'ipsi')
        focus = 4;
    else
        error('Unrecognized tuning type for cell %d: %s',k,tuning);
    end

    SpikeTimes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
    picture = false(n_trials,simlen);
    for m = 1:n_trials
        % Match Data/data_handler.py by binning the original timestamps.
        % Rounding to the 0.1-ms raster first can move spikes across 10-ms bins.
        stim_mask = (SpikeTimes{m} >= 0) & ...
            (SpikeTimes{m} <= simlen/sample_rate_hz);
        cut_times = SpikeTimes{m}(stim_mask);
        PSTHs_data(k,:) = PSTHs_data(k,:) + ...
            histcounts(cut_times,bin_edges_seconds);

        trial_indices = floor(cut_times*sample_rate_hz);
        trial_indices = trial_indices( ...
            trial_indices >= 0 & trial_indices < simlen);
        picture(m,trial_indices+1) = true;
    end
    Rasters_data(k,:,:) = picture;
end

%%
sim_PSTHs = squeeze(sum(sum(reshape( ...
    permute(output(:,:,:,1:used_timesteps),[3,1,2,4]), ...
    [n_cells,n_batches,n_trials,PSTH_gran,n_bins]),4),3));

%%
figure;
subplot(2,1,2)
spy(squeeze(Rasters_data(cell,:,:)));
subplot(2,1,1)
plot(squeeze(PSTHs_data(cell,:)));


figure;
for k = 1:batch_size
    subplot(batch_size,2,(k-1)*2 + 1)
    plot(squeeze(sim_PSTHs(cell,k,:)));
    subplot(batch_size,2,(k-1)*2 + 2)
    spy(squeeze(output(k,:,cell,:)));
   
end

%%
cell_PSTH = squeeze(sim_PSTHs(cell,:,:));
cell_data_PSTH = PSTHs_data(cell,:);

sum((cell_PSTH-cell_data_PSTH).^2,2)