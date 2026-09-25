close all; clear all;
script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir); addpath(script_dir); end
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\tar2_test.mat';
sim_location2 = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\all_inits_100_epochs_sub_batch_5_split.mat';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
%sim_object = split_and_load_large_sim_object(sim_location);

Data_Rasters = zeros([220,10,29801]);
for k = 1:220
    tuning = lower(char(string(data_object.all_data(k).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
    for m = 1:10
        spike_trial = spikes{m};
        valid_spikes = round(spike_trial((spike_trial>0) & (spike_trial<2.9801))*10000);
        Data_Rasters(k,m,valid_spikes) = 1;
    end
end

[y, Fs] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target1.wav');
y = y((0.25*Fs)-1:end);
time = (0:length(y)-1) / Fs;

data_PSTHs = [];
psth_bin_edges = (0:200:29800)/10000;  % 20 ms bins, in seconds
psth_time = psth_bin_edges(1:end-1);   % plot each bin at its left edge

re_vals = [];

for k = 1:220
    tuning = lower(char(string(data_object.all_data(k).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
    data_times = [];
    for m = 1:10
        data_times = [data_times;spikes{m}];
    end
    data_PSTHs = [data_PSTHs;movmean(histcounts(data_times,psth_bin_edges),3)];

    spikes = cellfun(@transpose, spikes, 'UniformOutput', false)';
    STS = SpikeTrainSet(spikes,0,3);
    dist_mat = STS.SPIKEdistanceMatrix();
    re_vals = [re_vals, 1 - mean(dist_mat(triu(true(size(dist_mat)),1)),'omitnan')];
end

%%

close all;

for cell = 130

    figure(Position=[0,0,800,700]);
    outer = tiledlayout(3,1, 'TileSpacing','none','Padding','none');
    ax(1) = nexttile(outer);
    Raster_matrix = squeeze(Data_Rasters(cell,:,:));
    raster_plot_func(Raster_matrix)
    ax(2) = nexttile(outer);
    plot(psth_time,data_PSTHs(cell,:),'b','LineWidth',1);
    xlim([0 2.9801])
    xticks([])
    yticklabels('')
    ax(3) = nexttile(outer);
    plot(time,y,'k')
    xlim([0, time(end)])
    xlabel('Time (s)')
    yticklabels('')
    yticks([])
    xticks(0:0.5:time(end))
    sgtitle(['Cell: ',num2str(cell),' RE = ',num2str(re_vals(cell))])


    for a = ax([1,2,3])
    xline(a, 0.38, 'LineWidth', 1,'Color',[0.5,0,0], ...
        'HandleVisibility','off');
    xline(a, 0.5, 'LineWidth', 1,'Color',[0.5,0,0], ...
        'HandleVisibility','off');
    end
    
    for a = ax([1,2,3])
        xline(a, 0.96, 'LineWidth', 1,'Color',[0.5,0,0], ...
            'HandleVisibility','off');
        xline(a, 1.2, 'LineWidth', 1,'Color',[0.5,0,0], ...
            'HandleVisibility','off');
    end

end


function raster_plot_func(Raster_matrix)
    
    [trial,sample] = find(Raster_matrix);
    time = sample/10000;

    line([time time]', ...
         [trial-0.5 trial+0.5]', ...
         'Color','k','LineWidth',0.6)

    xlim([0 2.9801])
    ylim([0.5 size(Raster_matrix,1)+0.5])
    yticklabels('')
    xticklabels(' ')
    yticks([])
    box off
end