%% Find top 5 highest concordance cells
load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_wide_forwards_sensitivity_cell_7_fr_corrections_lamda_10_lr_0.01.mat")

n_batches = 1200;
target_cell = 7;
PSTH_gran = 100; 
simlen = 29801;
n_trials = 10;
n_cells = 1;
n_bins = (simlen-mod(simlen,PSTH_gran))/PSTH_gran;
used_timesteps = n_bins*PSTH_gran;
[data_PSTH,data_raster] = get_data_PSTH(target_cell,PSTH_gran,n_bins,used_timesteps,n_trials);
[sim_PSTH,sim_raster]  = get_sim_PSTH(PSTH_gran,n_bins,used_timesteps,n_trials,n_cells,n_batches);

concordance = [];
for k = 1:n_batches
    result = linCCC(data_PSTH,sim_PSTH(k,:));
    ccc_val = result.ccc;
    concordance = [concordance, ccc_val];
end

[val idx] = sort(concordance,'descend');
x = 5;
topx = idx(1:x);

%% Look at thier initializations
%Load in the first epoch output (Initialization)
import_object = load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Model_Outputs\lamda10BPTT_rasters_lr_0.01\rasters_epoch_001.mat");
final_object = load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Model_Outputs\lamda10BPTT_rasters_lr_0.01\rasters_epoch_100.mat");
figure;
for k = 1:x
    subplot(x,1,k)
    spy(squeeze(import_object.output(topx(k),:,:,:)))
end
sgtitle('TOP 5 cells initialization')

figure;
for k = 1:x    
    subplot(x,1,k)
    spy(squeeze(final_object.output(topx(k),:,:,:)))
end
sgtitle('TOP 5 cells after 100 epochs')



%% Look at their underlying parameters at the start and at the end.
params_names = fields(import_object.params);
figure;
cell = 7;
batch_size = size(output,1);
figure;

for k = 1:12
    pram = params_names{k};
    disp(pram)
    disp(import_object.params.(pram)(topx(1)))
    noramaliazed_pram = import_object.params.(pram)/max(import_object.params.(pram));
    plot(ones(1,n_batches)*k,noramaliazed_pram,'k.','MarkerSize',5); hold on
    plot(ones(1,x)*k,noramaliazed_pram(topx),'r.','MarkerSize',20); hold on
end
sgtitle('Parameter Initializations normalized')

figure;
for k = 1:12
    pram = params_names{k};
    noramaliazed_pram = final_object.params.(pram)(:,:,100)/max(final_object.params.(pram)(:,:,100));
    plot(ones(1,n_batches)*k,noramaliazed_pram,'k.','MarkerSize',5); hold on
    plot(ones(1,x)*k,noramaliazed_pram(topx),'r.','MarkerSize',20); hold on
end
sgtitle('Parameters after 100 epochs normalized')

%% Functions
function [data_PSTH,data_raster] = get_data_PSTH(data_cell,PSTH_gran,n_bins,used_timesteps,n_trials)
    data_object = load('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat');
    tuning = lower(char(string(data_object.all_data(data_cell).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    %Bring in the data spike times
    spike_times = data_object.all_data(data_cell).ctrl_tar1_timestamps(:,focus);
    data_raster = zeros(10,29801);
    for k = 1:10
        spike_times_data_batch = spike_times{k};
        for m = 1:length(spike_times_data_batch)
            if (spike_times_data_batch(m)>0) & (spike_times_data_batch(m)<=2.9801)
                data_raster(k,round(spike_times_data_batch(m)*10000)) = 1;
            end
        end
    end
    data_PSTH = squeeze(sum(sum(reshape( ...
    data_raster(:,1:used_timesteps), ...
    [n_trials,PSTH_gran,n_bins]),2),1));
end

function [sim_PSTH,sim_raster] = get_sim_PSTH(PSTH_gran,n_bins,used_timesteps,n_trials,n_cells,n_batches)
    sim_object = load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Model_Outputs\lamda10BPTT_rasters_lr_0.01\rasters_epoch_100.mat");
    sim_raster = squeeze(sim_object.output);
    sim_PSTH = squeeze(sum(sum(reshape( ...
    permute(sim_object.output(:,:,:,1:used_timesteps),[3,1,2,4]), ...
    [n_cells,n_batches,n_trials,PSTH_gran,n_bins]),4),3));
end


