clear all; close all;
PSTH_gran = 100; 
simlen = 29801;
n_trials = 10;
n_cells = 1;
n_bins = (simlen-mod(simlen,PSTH_gran))/PSTH_gran;
used_timesteps = n_bins*PSTH_gran;

data_PSTH_holder = zeros([220,n_bins]);
data_raster_holder = zeros([220,n_trials,simlen]);
data_object = load('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat');
[y, Fs] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target1.wav');
[y2, Fs2] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target2.wav');

cvs = [];
psuedo_densities = [];
onsetness = [];
for k = 1:220
    [data_PSTH,data_raster] = get_data_PSTH(k,PSTH_gran,n_bins,used_timesteps,n_trials,data_object);
    data_PSTH_holder(k,:) = data_PSTH;
    data_raster_holder(k,:,:) = data_raster;
    cvs = [cvs,calc_cv(data_raster)];
    psuedo_densities = [psuedo_densities,calc_pseudo_density(data_PSTH,n_bins)];
    onsetness = [onsetness,calculate_onsetness(data_raster)];
end

[val idx] = sort(onsetness,'descend');

data_raster_holder_sorted = data_raster_holder(idx,:,:);
data_raster_holder_sorted_monolith = reshape(permute(data_raster_holder_sorted,[2,1,3]),[220*10, 29801]);

figure('Position',[0,0,600,1200]);
spy(data_raster_holder_sorted_monolith(1:2200,:))
axis fill;
ylim([0,2200])

%Save values to influence parameter initialization
save_onsetness = onsetness + 0.1; % I think this helps balance things
save('onsetness.mat',"save_onsetness")
save('psuedo_densities.mat',"psuedo_densities")

function [data_PSTH,data_raster] = get_data_PSTH(data_cell,PSTH_gran,n_bins,used_timesteps,n_trials,data_object)
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

function pd = calc_pseudo_density(data_PSTH,n_bins)
    %auc = sum((data_PSTH/max(data_PSTH))/n_bins);
    %pd = 0.0001*sum(data_PSTH)+(1-auc);
    %auc = sum(data_PSTH);
    %pd = (1-auc);
    pd = 0;
    smooth_signal = movmean(data_PSTH,5);
    [val,start_idx] = max(smooth_signal);
    for k = start_idx:length(data_PSTH)
        val_compare = smooth_signal(k);
        if (val_compare/val) < 0.5
            pd = k - start_idx;
            break;
        end
    end
    
    if pd == 0 
        pd = 40;
    end

end

function cv = calc_cv(data_raster)
    isi_holder = [];
    for k = 1:10
        isi_holder = [isi_holder, diff(find(data_raster(k,:)))];
    end
    cv = std(isi_holder)/mean(isi_holder);
end

function onsetness = calculate_onsetness(data_raster)

    onset_regions = [1700,2450;
                     4050,5250;
                     7000,7500;
                     8500,10100;
                     12200,12500;
                     14300,16450;
                     17400,19600;
                     21500,21800;
                     22900,24600;
                     28200,28600;];

    total_spikes = 0;
    for k = 1:length(onset_regions)
        total_spikes = total_spikes + sum(sum(data_raster(:,(onset_regions(k,1):(onset_regions(k,2))))));
    end

    onsetness = total_spikes/sum(sum(data_raster));

end
