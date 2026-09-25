close all; clear all;
script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir); addpath(script_dir); end
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\tar2_test.mat';
sim_location2 = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\all_inits_100_epochs_sub_batch_5_split.mat';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = load(sim_location);

sim_object2 = load_large_sim_object(sim_location2);

% Do selection
Choose_by = 'SPIKE';
selection = calculate_selction(Choose_by, sim_object2, data_object);

Data_Rasters = zeros([220,10,29801]);
for k = 1:220
    tuning = lower(char(string(data_object.all_data(k).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    spikes = data_object.all_data(k).ctrl_tar2_timestamps(:,focus);
    for m = 1:10
        spike_trial = spikes{m};
        valid_spikes = round(spike_trial((spike_trial>=0.0001) & (spike_trial<2.9801))*10000);
        Data_Rasters(k,m,valid_spikes) = 1;
    end
end

Data_Rasters1 = zeros([220,10,29801]);
for k = 1:220
    tuning = lower(char(string(data_object.all_data(k).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
    for m = 1:10
        spike_trial = spikes{m};
        valid_spikes = round(spike_trial((spike_trial>=0.0001) & (spike_trial<2.9801))*10000);
        Data_Rasters1(k,m,valid_spikes) = 1;
    end
end

%%
close all;
good = [1	15	19	52	53	64	92	96	124	127	128	129	132	186	200	210];
great = [7  68  102 133];

Labeled_cell = 11;
Labeled_cell2 = 133;
Labeled_cell3 = 102;
Labeled_cell4 = 52;

figure('Position',[0,0,1800,1600]);
subplot(6,4,1);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell,:,:));
raster_plot_func(Raster_matrix)
ylabel('Data')
title(['Cell : ', num2str(Labeled_cell)])
subplot(6,4,2);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell2,:,:));
raster_plot_func(Raster_matrix)
title(['Cell : ', num2str(Labeled_cell2)])
subplot(6,4,3);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell3,:,:));
raster_plot_func(Raster_matrix)
title(['Cell : ', num2str(Labeled_cell3)])
subplot(6,4,4);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell4,:,:));
raster_plot_func(Raster_matrix)
title(['Cell : ', num2str(Labeled_cell4)])
subplot(6,4,5);
Raster_matrix = squeeze(sim_object.output(1,:,Labeled_cell,:));
raster_plot_func(Raster_matrix)
ylabel('Model')
subplot(6,4,6);
Raster_matrix = squeeze(sim_object.output(1,:,Labeled_cell2,:));
raster_plot_func(Raster_matrix)
subplot(6,4,7);
Raster_matrix = squeeze(sim_object.output(1,:,Labeled_cell3,:));
raster_plot_func(Raster_matrix)
subplot(6,4,8);
Raster_matrix = squeeze(sim_object.output(1,:,Labeled_cell4,:));
raster_plot_func(Raster_matrix)


%subplot(10,3,[25,28])
[y, Fs] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target2.wav');
y = y((0.25*Fs)-1:end);
time = (0:length(y)-1) / Fs;

for k = 1:4
    
    subplot(6,4,8+k)
    plot(time,y,'k')
    xlim([0, time(end)])
    xlabel('Time (s)')
    yticklabels('')
    yticks([])
    xticks(0:0.5:time(end))
    if k == 1
        ylabel('Target 2')
    end

end


subplot(6,4,13);
Raster_matrix = squeeze(Data_Rasters1(Labeled_cell,:,:));
raster_plot_func(Raster_matrix)
ylabel('Data')
title(['Cell : ', num2str(Labeled_cell)])
subplot(6,4,14);
Raster_matrix = squeeze(Data_Rasters1(Labeled_cell2,:,:));
raster_plot_func(Raster_matrix)
title(['Cell : ', num2str(Labeled_cell2)])
subplot(6,4,15);
Raster_matrix = squeeze(Data_Rasters1(Labeled_cell3,:,:));
raster_plot_func(Raster_matrix)
title(['Cell : ', num2str(Labeled_cell3)])
subplot(6,4,16);
Raster_matrix = squeeze(Data_Rasters1(Labeled_cell4,:,:));
raster_plot_func(Raster_matrix)
title(['Cell : ', num2str(Labeled_cell4)])
subplot(6,4,17);
Raster_matrix = squeeze(sim_object2.output(selection(Labeled_cell),:,Labeled_cell,:));
raster_plot_func(Raster_matrix)
ylabel('Model')
subplot(6,4,18);
Raster_matrix = squeeze(sim_object2.output(selection(Labeled_cell2),:,Labeled_cell2,:));
raster_plot_func(Raster_matrix)
subplot(6,4,19);
Raster_matrix = squeeze(sim_object2.output(selection(Labeled_cell3),:,Labeled_cell3,:));
raster_plot_func(Raster_matrix)
subplot(6,4,20);
Raster_matrix = squeeze(sim_object2.output(selection(Labeled_cell4),:,Labeled_cell4,:));
raster_plot_func(Raster_matrix)


[y, Fs] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target1.wav');
y = y((0.25*Fs)-1:end);
time = (0:length(y)-1) / Fs;

for k = 1:4
    
    subplot(6,4,20+k)
    plot(time,y,'k')
    xlim([0, time(end)])
    xlabel('Time (s)')
    yticklabels('')
    yticks([])
    xticks(0:0.5:time(end))
    if k == 1
        ylabel('Target 1')
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
    %set(gca,'YDir','reverse','YTick',1:size(Raster_matrix,1))
    %xlabel('Time (s)')
    %ylabel('Trial')
    %title(['Cell ',cell_num,' — ',model_or_data])
    yticklabels('')
    xticklabels(' ')
    yticks([])
    box off
end


function selection = calculate_selction(choose_by, sim_object, data_object)
    
    output_size = size(sim_object.output);
    n_batches = output_size(1);

    selection = [];
    if strcmp(choose_by, 'none')
        selection = repmat(1:12,220,1);
    elseif strcmp(choose_by, 'SSE')
        for k = 1:220
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
            sses = [];
            for m = 1:n_batches
                for z = 1:10
                    spikes{z+10} = find(sim_object.output(m,z,k,:))/10000;
                end
                data_times = [];
                for m = 1:10
                    data_times = [data_times;spikes{m}];
                end
                sim_times = [];
                for d = 11:20
                    sim_times = [sim_times;spikes{d}];
                end
                bin_edges = (0:100:29799)/10000;
                data_PSTH = histcounts(data_times,bin_edges);
                sim_PSTH = histcounts(sim_times,bin_edges);
                sses = [sses, sum((data_PSTH-sim_PSTH).^2)]; 
            end
            [val,idx] = min(sses);
            selection = [selection, idx];
        end

    elseif strcmp(choose_by, 'NCCC')

        for k = 1:220
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
            NCCCs = [];
            for m = 1:n_batches
                for z = 1:10
                    spikes{z+10} = find(sim_object.output(m,z,k,:))/10000;
                end
                data_times = [];
                ris = [];
                bin_edges = (0:100:29799)/10000;
                for m = 1:10
                    data_times = [data_times;spikes{m}];
                    ris = [ris;histcounts(spikes{m},bin_edges)];
                end
                
                sim_times = [];
                
                for d = 11:20
                    sim_times = [sim_times;spikes{d}];
                    
                end
                
                data_PSTH = histcounts(data_times,bin_edges);
                sim_PSTH = histcounts(sim_times,bin_edges);
                
                cor_vals = zeros([10, 10]);
                for mm = 1:10
                   for zz = 1:10
                       corr_mat = corrcoef(ris(mm,:),ris(zz,:));
                       cor_vals(mm,zz) = corr_mat(2,1);
                   end
                end
                
                TTRC = mean(cor_vals(triu(true(size(cor_vals)), 1)), 'omitnan');
                
                pred_cor = zeros([10, 1]);
                for mm = 1:10
                    corr_mat = corr(ris(mm,:),sim_PSTH);
                    pred_cor(mm) = corr(ris(mm,:)', sim_PSTH','Rows', 'complete');
                end

                NCCCs = [NCCCs, (1/sqrt(TTRC))*mean(pred_cor)];

            end
            [val,idx] = max(NCCCs);
            selection = [selection, idx];
        end
    elseif strcmp(choose_by, 'SPIKE')
        for k = 1:220
            distances = [];
            for m = 1:n_batches
                tuning = lower(char(string(data_object.all_data(k).tuning_type)));
                if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
                spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
                for z = 1:10
                    spikes{z+10} = find(sim_object.output(m,z,k,:))/10000;
                end
                spikes = cellfun(@transpose, spikes, 'UniformOutput', false)';
                STS = SpikeTrainSet(spikes,0,3);
                dist_mat = STS.SPIKEdistanceMatrix();
                distances = [distances,mean(mean(dist_mat(1:10,11:20)))];
            end
            [val,idx] = min(distances);
            selection = [selection, idx];
        end
    elseif strcmp(choose_by, 'CV')
        for k = 1:220
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
            data_cv = calc_cv_data(spikes);
            cv_val = [];
            for m = 1:n_batches %For all batches
                isi_s = []; 
                for z  = 1:10 % For all trials
                    isi_s = [isi_s,squeeze(diff(find(sim_object.output(m,z,k,:)))/10000)'];
                end
                cv_val = [cv_val,std(isi_s)/mean(isi_s)];
            end
            [val,idx] = min((data_cv - cv_val).^2);
            selection = [selection, idx];
        end
    end

        
end