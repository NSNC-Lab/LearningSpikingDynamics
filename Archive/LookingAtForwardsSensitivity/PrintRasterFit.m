%clear all; close all;
PSTH_gran = 100; 
simlen = 29801;
n_trials = 10;
n_cells = 220;
n_batches = 12;
n_bins = (simlen-mod(simlen,PSTH_gran))/PSTH_gran;
used_timesteps = n_bins*PSTH_gran;

data_PSTH_holder = zeros([220,n_bins]);
data_raster_holder = zeros([220,n_trials,simlen]);
sim_raster_holder = zeros([220,n_trials,simlen]);
data_object = load('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat');
[y, Fs] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target1.wav');
[y2, Fs2] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target2.wav');

load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_all_cells_Eprop_lamda2_10.mat")
sim_PSTH = squeeze(sum(sum(reshape(permute(output(:,:,:,1:used_timesteps),[3,1,2,4]),[n_cells,n_batches,n_trials,PSTH_gran,n_bins]),4),3));

for k = 1:220
    [data_PSTH,data_raster] = get_data_PSTH(k,PSTH_gran,n_bins,used_timesteps,n_trials,data_object);
    data_PSTH_holder(k,:) = data_PSTH;
    data_raster_holder(k,:,:) = data_raster;

    %Create a plot
    %figure('Position',[500,0,400,1300]);
    subplot(13,1,1);
    spy(data_raster)
    title(['Cell ', num2str(k), ' data'])
    %subplot(14,1,14);
    
    for m = 2:13
        subplot(13,1,m)
        spy(squeeze(output(m-1,:,k,:)))
        title(['batch: ',num2str(m-1)])
        %subplot(14,1,14);
        
    end
    %figure;
    [val,idx] = min(sum((data_PSTH' - squeeze(sim_PSTH(k,:,:))).^2,2));
    for m = 1:12
        if m == idx
            plot(squeeze(sim_PSTH(k,m,:)),'g-','LineWidth',2); hold on;
        else
            %plot(squeeze(sim_PSTH(k,m,:)),'Color',[0.5,0.5,0.5,0.5],'LineWidth',0.5); hold on;
        end
    end
    plot(data_PSTH,'r-','LineWidth',2); hold on
    title(['PSTH comparison Cell ',num2str(k), ' -- Best match: Batch ', num2str(idx)])

    sim_raster_holder(k,:,:) = output(idx,:,k,:);
end

%%
figure('Position',[0,0,1000,1300]);
subplot(1,2,1);
spy(reshape(permute(data_raster_holder,[2,1,3]),2200,29801))
axis fill
ylim([0 2200])
subplot(1,2,2);
spy(reshape(permute(sim_raster_holder,[2,1,3]),2200,29801))
axis fill
ylim([0 2200])


%% Looking at raw loss comparison
long_object = load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Archive\LongRunResults.mat");
long_output_inform = permute(squeeze(long_object.output),[2,3,1,4]);
n_batches_long = 15;
sim_PSTH_long = squeeze(sum(sum(reshape(permute(long_output_inform(:,:,:,1:used_timesteps),[3,1,2,4]),[n_cells,n_batches_long,n_trials,PSTH_gran,n_bins]),4),3));


long_loss = [];
modern_loss = [];
count = 0;
count2 = 0;
count3 = 0;

%Append these stats given the best sse example
cv_dist = [];
fr_dist = [];
cv_dist_long = [];
fr_dist_long = [];

data_cv = [];
data_fr = [];

cv_dist_cv = [];
cv_dist_cv_long = [];

cv_dist_SPIKE = [];
cv_dist_SPIKE_long = [];
sse_dist_SPIKE = [];
sse_dist_SPIKE_long = [];
SPIKE1 = [];
SPIKE2 = [];
fr_dist_SPIKE = [];
fr_dist_SPIKE_long = [];


for k = 1:220


    %Calc SPIKE distance
    
    % spike_distances1 = [];
    % spike_distances2 = [];
    % for m = 1:n_batches
    %     sim_raster_object1 = squeeze(output(m,:,k,:));
    %     sim_raster_object2 = squeeze(long_output_inform(m,:,k,:));
    %     spike_times1 = calc_spike_times(sim_raster_object1,'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat',k,100);
    %     spike_times2 = calc_spike_times(sim_raster_object2,'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat',k,100);
    %     spike_distances1 = [spike_distances1, calculate_spike_distance(spike_times1)];
    %     spike_distances2 = [spike_distances2, calculate_spike_distance(spike_times2)];
    % end
    % 
    % [valSPIKE1,idx_SPIKE] = min(spike_distances1);
    % [valSPIKE2,idx_SPIKE_long] = min(spike_distances2);
    % 
    % SPIKE1 = [SPIKE1, valSPIKE1];
    % SPIKE2 = [SPIKE2, valSPIKE2];

    %Lookin at sse
    [val,idx_sse] = min(sum((data_PSTH_holder(k,:) - squeeze(sim_PSTH(k,:,:))).^2,2));
    [val_long,idx_long_sse] = min(sum((data_PSTH_holder(k,:) - squeeze(sim_PSTH_long(k,:,:))).^2,2));
    modern_loss = [modern_loss, val];
    long_loss = [long_loss, val_long];

    %sse_dist_SPIKE = [sse_dist_SPIKE, sum((data_PSTH_holder(k,:) - squeeze(sim_PSTH(k,idx_SPIKE,:))').^2,2)];
    %sse_dist_SPIKE_long = [sse_dist_SPIKE_long, sum((data_PSTH_holder(k,:) - squeeze(sim_PSTH_long(k,idx_SPIKE_long,:))).^2,2)];

    %15 out of 220 cells improved with the latest run.
    if val < val_long
        count = count + 1;
    end

    %Looking at CV
    cvs = [];
    for m = 1:n_batches
        cvs = [cvs,(calc_cv(squeeze(output(m,:,k,:)))-calc_cv(squeeze(data_raster_holder(k,:,:))))^2];
    end

    cv_dist = [cv_dist, calc_cv(squeeze(output(idx_sse,:,k,:)))];
    data_cv = [data_cv, calc_cv(squeeze(data_raster_holder(k,:,:)))];
    
    cvs_long = [];
    for m = 1:n_batches_long
        cvs_long = [cvs_long,(calc_cv(squeeze(long_output_inform(m,:,k,:)))-calc_cv(squeeze(data_raster_holder(k,:,:))))^2];
    end

    cv_dist_long = [cv_dist_long, calc_cv(squeeze(long_output_inform(idx_long_sse,:,k,:)))];

    [val,idx_cv] = min(cvs);
    [val_long,idx_cv_long] = min(cvs_long);

    cv_dist_cv = [cv_dist_cv, calc_cv(squeeze(output(idx_cv,:,k,:)))];
    cv_dist_cv_long = [cv_dist_cv_long, calc_cv(squeeze(long_output_inform(idx_cv_long,:,k,:)))];
    
    %cv_dist_SPIKE = [cv_dist_SPIKE,  calc_cv(squeeze(output(idx_SPIKE,:,k,:)))];
    %cv_dist_SPIKE_long = [cv_dist_SPIKE_long, calc_cv(squeeze(long_output_inform(idx_SPIKE_long,:,k,:)))];
    
    %170 out of 220 cells improved with the latest run.
    if val < val_long
        count2 = count2 + 1;
    end

    %Lookin at rate
    difs = (sum(sum(data_raster_holder(k,:,:))) - sum(sum(squeeze(output(:,:,k,:)),2),3)).^2;
    difs_long = (sum(sum(data_raster_holder(k,:,:))) - sum(sum(squeeze(long_output_inform(:,:,k,:)),2),3)).^2;

    [val,idx] = min(difs);
    [val_long,idx] = min(difs_long);

    data_fr = [data_fr, sum(sum(data_raster_holder(k,:,:)))];
    fr_dist = [fr_dist, sum(sum(squeeze(output(idx_sse,:,k,:))))];
    fr_dist_long = [fr_dist_long, sum(sum(squeeze(long_output_inform(idx_long_sse,:,k,:))))];
    
    %fr_dist_SPIKE = [fr_dist_SPIKE, sum(sum(squeeze(output(idx_SPIKE,:,k,:))))];
    %fr_dist_SPIKE_long = [fr_dist_SPIKE_long,sum(sum(squeeze(long_output_inform(idx_SPIKE_long,:,k,:))))];
    
    %105 out of 220 cells improved
    if val < val_long
        count3 = count3 + 1;
    end


end
%%

%Choosing wrt best sse
n_bins_gran = 50;
figure;
subplot(2,1,1);
histogram(data_cv,n_bins_gran); hold on
histogram(cv_dist,n_bins_gran); hold on
histogram(cv_dist_long,n_bins_gran); hold on
title('cv distribution')
legend({'data','new run', 'old run'})
subplot(2,1,2);
histogram(data_fr,n_bins_gran); hold on
histogram(fr_dist,n_bins_gran); hold on
histogram(fr_dist_long,n_bins_gran); hold on
title('fr distribution')
legend({'data','new run', 'old run'})


figure;
boxchart([modern_loss',long_loss']); hold on
xticklabels({'new', 'old'})

%Choosing wrt best cv
figure;
histogram(data_cv,n_bins_gran); hold on
histogram(cv_dist_cv,n_bins_gran); hold on
histogram(cv_dist_cv_long,n_bins_gran); hold on
title('cv distribution (choosing by cv)')
legend({'data','new run', 'old run'})

figure;
subplot(2,2,1);
histogram(data_cv,n_bins_gran); hold on
histogram(cv_dist_SPIKE,n_bins_gran); hold on
histogram(cv_dist_SPIKE_long,n_bins_gran); hold on
title('cv distribution (choosing by SPIKE)')
legend({'data','new run', 'old run'})
subplot(2,2,2);
histogram(data_fr,n_bins_gran); hold on
histogram(fr_dist_SPIKE,n_bins_gran); hold on
histogram(fr_dist_SPIKE_long,n_bins_gran); hold on
title('fr distribution (choosing by SPIKE)')
legend({'data','new run', 'old run'})
subplot(2,2,3);
boxchart([sse_dist_SPIKE',sse_dist_SPIKE_long']); hold on
xticklabels({'new', 'old'})
title('sse distribution (choosing by SPIKE)')
subplot(2,2,4);
boxchart([SPIKE1',SPIKE2']); hold on
xticklabels({'new', 'old'})
title('SPIKE distribution (choosing by SPIKE)')

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

function cv = calc_cv(data_raster)
    isi_holder = [];
    for k = 1:10
        isi_holder = [isi_holder, diff(find(data_raster(k,:)))];
    end
    cv = std(isi_holder)/mean(isi_holder);
end

function spike_distance = calculate_spike_distance(spike_times)
    STS = SpikeTrainSet(spike_times,0,3);
    dist_mat = STS.SPIKEdistanceMatrix();
    %Currently going to use the average over all pairwise comparisons
    %between data-sim spike trains. SPIKE-dist is calculated per spike
    %train pair in SPIKEdistanceMatrix.
    spike_distance = mean(mean(dist_mat(1:10,11:20)));
end

function spike_times = calc_spike_times(spike_object,data_location,data_cell,epoch)
    data_object = load(data_location);
    tuning = lower(char(string(data_object.all_data(data_cell).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    %Bring in the data spike times
    spike_times = data_object.all_data(data_cell).ctrl_tar1_timestamps(:,focus);
    %Append a simulations spike times
    for k = 1:10
        spike_times{k+10} = transpose((find(squeeze(spike_object(k,:)))-1)/10000); %Putting this in real time. The - 1 accounts for find being matlab indexing and the /10000 converts it into seconds
    end
    %Transpose things so they work with cSPIKE
    for k = 1:20
        spike_times{k} = transpose(spike_times{k});
    end
    spike_times = transpose(spike_times);
end
