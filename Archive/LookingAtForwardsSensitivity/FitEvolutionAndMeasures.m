%The goal of this script is to visualize the evolution of the
%fit over time.

%Start with the following:
% Plot the raster at every epoch for a few candidate cells 
% and see how things evolve.

% How should I visualize 100 rasters?

% Ideas:

% 1. Giant spy plot with lines at every 10 rows to split them up.
% 2. scrollable plot
% 3. animation
% 4. All on one plot but change color by epoch
clear all; close all;
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")

file_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Model_Outputs\lamda10EPROP_rasters';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_cell = 7;
for k = 1:10
    disp(k)
    target_batch = 1000+k;
    %%
    sim_raster_object = Extract_Sim_Raster(file_location,target_batch);
    %%
    spike_distances = [];
    sse_distances = [];
    for epoch = 1:200
        spike_times = calc_spike_times(sim_raster_object,data_location,data_cell,epoch);
        spike_distances = [spike_distances, calculate_spike_distance(spike_times)];
        sse_distances = [sse_distances, calculate_sse_dist(spike_times)];
    end
    %% Trying 1: Giant spy plot
    figure('Position',[0,0,1250,1250]);
    subplot(1,5,1:3);
    monolith_spy = reshape(permute(sim_raster_object,[2,1,3]),[2000,29801]);
    spy(monolith_spy)
    axis fill
    title('evolution of raster')
    %Plot measures per epoch
    subplot(1,5,4)
    plot(spike_distances,1:200); hold on
    [best_SPIKE_dist_val,best_SPIKE_dist_idx] = min(spike_distances);
    plot(best_SPIKE_dist_val,best_SPIKE_dist_idx,'kx','MarkerSize',15,'LineWidth',2)
    title('evolution of spike distance')
    ylabel('epoch')
    set(gca, 'YDir', 'reverse');
    ylim([1 200])
    subplot(1,5,5)
    plot(sse_distances,1:200); hold on
    [best_sse_dist_val,best_sse_dist_idx] = min(sse_distances);
    plot(best_sse_dist_val,best_sse_dist_idx,'kx','MarkerSize',15,'LineWidth',2)
    set(gca, 'YDir', 'reverse');
    title('evolution of sse distance')
    ylabel('epoch')
    ylim([1 200])

    %Visualization plot
    data_raster = get_data_raster(spike_times);
    figure;
    subplot(3,1,1);
    spy(data_raster)
    title('data Raster')
    subplot(3,1,2);
    spy(squeeze(sim_raster_object(best_SPIKE_dist_idx,:,:)))
    title('Best Spike-Distance')
    subplot(3,1,3);
    spy(squeeze(sim_raster_object(best_sse_dist_idx,:,:)))
    title('Best SSE distance')
end


%%

function raster_holder = Extract_Sim_Raster(f_loc,target_batch)
    raster_holder = zeros([200,10,29801]);
    for k = 1:200
        saved_epoch_object = load([f_loc, '\rasters_epoch_',sprintf( '%03d', k ),'.mat']);
        raster_object = squeeze(saved_epoch_object.output);
        raster_holder(k,:,:) = squeeze(raster_object(target_batch,:,:));
    end
end

function spike_times = calc_spike_times(spike_object,data_location,data_cell,epoch)
    data_object = load(data_location);
    tuning = lower(char(string(data_object.all_data(data_cell).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    %Bring in the data spike times
    spike_times = data_object.all_data(data_cell).ctrl_tar1_timestamps(:,focus);
    %Append a simulations spike times
    for k = 1:10
        spike_times{k+10} = (find(squeeze(spike_object(epoch,k,:)))-1)/10000; %Putting this in real time. The - 1 accounts for find being matlab indexing and the /10000 converts it into seconds
    end
    %Transpose things so they work with cSPIKE
    for k = 1:20
        spike_times{k} = transpose(spike_times{k});
    end
    spike_times = transpose(spike_times);
end

function spike_distance = calculate_spike_distance(spike_times)
    STS = SpikeTrainSet(spike_times,0,3);
    dist_mat = STS.SPIKEdistanceMatrix();
    %Currently going to use the average over all pairwise comparisons
    %between data-sim spike trains. SPIKE-dist is calculated per spike
    %train pair in SPIKEdistanceMatrix.
    spike_distance = mean(mean(dist_mat(1:10,11:20)));
end

function sse_dist = calculate_sse_dist(spike_times)
    
    data_times = [];
    for k = 1:10
        data_times = [data_times,spike_times{k}];
    end
    sim_times = [];
    for k = 11:20
        sim_times = [sim_times,spike_times{k}];
    end
    bin_edges = (0:100:29799)/10000;
    data_PSTH = histcounts(data_times,bin_edges);
    sim_PSTH = histcounts(sim_times,bin_edges);
    sse_dist = sum((data_PSTH-sim_PSTH).^2); 
end

function data_raster = get_data_raster(spike_times)
    data_raster = zeros(10,29801);
    for k = 1:10
        spike_times_data_batch = spike_times{k};
        for m = 1:length(spike_times_data_batch)
            if (spike_times_data_batch(m)>0) & (spike_times_data_batch(m)<2.9802)
                data_raster(k,round(spike_times_data_batch(m)*10000)) = 1;
            end
        end
    end
end