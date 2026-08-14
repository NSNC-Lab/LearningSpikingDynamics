load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_wide_eprop_cell_7_fr_corrections_lamda_0.1.mat") %Latest Forward

%% Construct the experimental PSTHs
data_object = load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat");
sample_rate_hz = 10000;
n_batches = size(output,1);
n_trials = size(output,2);
n_cells = 220;
simlen = size(output,4);
PSTH_gran = 100; 
n_bins = (simlen-mod(simlen,PSTH_gran))/PSTH_gran;
Rasters_data = false(n_cells,n_trials,simlen);
PSTHs_data1 = zeros(n_cells,n_bins);
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
        PSTHs_data1(k,:) = PSTHs_data1(k,:) + ...
            histcounts(cut_times,bin_edges_seconds);

        trial_indices = floor(cut_times*sample_rate_hz);
        trial_indices = trial_indices( ...
            trial_indices >= 0 & trial_indices < simlen);
        picture(m,trial_indices+1) = true;
    end
    Rasters_data(k,:,:) = picture;
end

cell = 7;
n_cells = 1;
sim_PSTHs1 = squeeze(sum(sum(reshape( ...
    permute(output(:,:,:,1:used_timesteps),[3,1,2,4]), ...
    [n_cells,n_batches,n_trials,PSTH_gran,n_bins]),4),3));

correlations1 = [];
for k = 1:n_batches
    %correlations = [correlations, max(xcorr(PSTHs_data(cell,:), sim_PSTHs(k,:),'normalized'))];
    %corr_val = corr(transpose([PSTHs_data(cell,:);sim_PSTHs(k,:)]));
    %correlations = [correlations, corr_val(1,2)];

    result = linCCC(PSTHs_data1(cell,:),sim_PSTHs1(k,:));
    ccc_val = result.ccc;
    correlations1 = [correlations1, ccc_val];
end

[vals1,idx1] = sort(correlations1,'descend');

load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_wide_forwards_sensitivity_cell_7.mat") %Latest Forward

sample_rate_hz = 10000;
n_batches = size(output,1);
n_trials = size(output,2);
n_cells = 220;
simlen = size(output,4);
PSTH_gran = 100; 
n_bins = (simlen-mod(simlen,PSTH_gran))/PSTH_gran;
Rasters_data = false(n_cells,n_trials,simlen);
PSTHs_data2 = zeros(n_cells,n_bins);
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
        PSTHs_data2(k,:) = PSTHs_data2(k,:) + ...
            histcounts(cut_times,bin_edges_seconds);

        trial_indices = floor(cut_times*sample_rate_hz);
        trial_indices = trial_indices( ...
            trial_indices >= 0 & trial_indices < simlen);
        picture(m,trial_indices+1) = true;
    end
    Rasters_data(k,:,:) = picture;
end

cell = 7;
n_cells = 1;
sim_PSTHs2 = squeeze(sum(sum(reshape( ...
    permute(output(:,:,:,1:used_timesteps),[3,1,2,4]), ...
    [n_cells,n_batches,n_trials,PSTH_gran,n_bins]),4),3));

correlations2 = [];
for k = 1:n_batches
    %correlations = [correlations, max(xcorr(PSTHs_data(cell,:), sim_PSTHs(k,:),'normalized'))];
    %corr_val = corr(transpose([PSTHs_data(cell,:);sim_PSTHs(k,:)]));
    %correlations = [correlations, corr_val(1,2)];

    result = linCCC(PSTHs_data2(cell,:),sim_PSTHs2(k,:));
    ccc_val = result.ccc;
    correlations2 = [correlations2, ccc_val];
end


[vals2,idx2] = sort(correlations2,'descend');

%% MDS based on loss
PSTHs_all = [sim_PSTHs1(:,:);sim_PSTHs2(:,:)];
comp_matrix_all = zeros(2400, 2400);

for m = 1:2400
    for k = 1:2400
        %RMSE
        r = sqrt(mean((PSTHs_all(m,:) - PSTHs_all(k,:)).^2));
        comp_matrix_all(m,k) = r;
        %SSE
        %r = sum((PSTHs_all(m,:) - PSTHs_all(k,:)).^2);
        %comp_matrix_all(m,k) = r;
    end
end

[Y,stress] = mdscale(comp_matrix_all,2, 'criterion','metricsstress');
%%

%mds_corr = [1,correlations]';
%mds_sse = [0;sse_losses_all];

figure;
topx = 20;
%scatter(Y(:,1),Y(:,2),10,mds_corr,'filled')
scatter(Y(1:1200,1),Y(1:1200,2),10,[0,0,0],'filled'); hold on
scatter(Y(1200:2400,1),Y(1200:2400,2),10,[0.5,0.5,0.5],'filled'); hold on
scatter(Y(idx1(1:topx),1),Y(idx1(1:topx),2),20,[0,1,1],'filled'); hold on
scatter(Y(idx2(1:topx)+1200,1),Y(idx2(1:topx)+1200,2),20,[1,0,1],'filled'); hold on
title(['Dual MDS space ' , 'Top: ' , num2str(topx)])
%title(['Dual MDS space'])
legend({'Eprop','BPTT',['Top ' , num2str(topx),' Eprop'],['Top ' , num2str(topx) ,  ' BPTT']})
%legend({'Eprop','BPTT'})
%xlim([-2 4])


%% Look at parameter bar graphs
load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_wide_forwards_sensitivity_cell_7_fr_corrections_lamda_0.1.mat") %Latest Forward
disp(sum(sum(sum(output))))
disp(sum(sum(sum(output(idx1(1:20),:,:,:))))/(sum(sum(Rasters_data(7,:,:)))*20))
disp(sum(sum(sum(output(idx1(1:20),:,:,:)))))
figure;
topx = 1200;
pram_store1 = ones(1,12);
for k = 1:12
    pram = params_names{k};
    pram_store1(k) = mean(params.(pram)(idx1(1:topx),1,end));
end


load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_wide_forwards_sensitivity_cell_7.mat") %Latest Forward
disp(sum(sum(sum(output))))
disp(sum(sum(sum(output(idx2(1:20),:,:,:))))/(sum(sum(Rasters_data(7,:,:)))*20))
disp(sum(sum(sum(output(idx1(1:20),:,:,:)))))
pram_store2 = ones(1,12);
for k = 1:12
    pram = params_names{k};
    pram_store2(k) = mean(params.(pram)(idx2(1:topx),1,end));
end

figure;
subplot(2,1,1)
bar(pram_store1)
subplot(2,1,2)
bar(pram_store2)

figure;
bar(pram_store1./pram_store2); hold on
plot(0:13,ones(14),'k')

%top1  (Eprop/BPTT)
%Parameter 1,9,12 (strf_gain 0.1x, off_ron 10x, off_sonoff 50x)