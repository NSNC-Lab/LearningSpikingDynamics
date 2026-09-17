%Import stuff
close all; clear all;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE")
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_all_cells_Eprop';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = load(sim_location);
Distance_measure = 'RMSE';

%%
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

Raster_matrixs = squeeze(sim_object.output(:,:,186,:));

color_vecs = [];
f_vals = fields(sim_object.params);
for m = 1:12
    sub_vec = [];
    for k = 1:220
        values = sim_object.params.(f_vals{m});
        sub_vec = [sub_vec, squeeze(values(:,k,end))];
    end
    color_vecs = [color_vecs; sub_vec];
end

selected_vec = reshape(squeeze(color_vecs(:,186)),[12,12])';
close all;
figure(Position=[100,100,1600,700]);
outer = tiledlayout(13,2, 'TileSpacing','compact','Padding','compact');
ax(1) = nexttile(outer);
ax(2) = nexttile(outer);
raster_plot_func(squeeze(Data_Rasters(186,:,:)))

for m = 1:2
    for k = 1:12
        ax(2 + (m-1)*12 + k) = nexttile;
        if (mod(k,2) == 0)
            raster_plot_func(squeeze(Raster_matrixs(k,:,:)))
        else
            plot(selected_vec(k,:))
        end
    end
end

%figure;
%histogram(save_onsetness)


%%

%Now lets look at different ways of characterizing sharpness and offsetnes

spikes = data_object.all_data(186).ctrl_tar1_timestamps(:,focus);

data_times = [];
for m = 1:10
    data_times = [data_times;spikes{m}];
end
bin_edges = (0:200:29799)/10000;
data_PSTH186 = histcounts(data_times,bin_edges);

figure;
plot(movmean(data_PSTH186,5))

spikes = data_object.all_data(7).ctrl_tar1_timestamps(:,focus);

data_times = [];
for m = 1:10
    data_times = [data_times;spikes{m}];
end
bin_edges = (0:200:29799)/10000;
data_PSTH7 = histcounts(data_times,bin_edges);

figure;
plot(movmean(data_PSTH7,5))

spikes = data_object.all_data(102).ctrl_tar1_timestamps(:,focus);

data_times = [];
for m = 1:10
    data_times = [data_times;spikes{m}];
end
bin_edges = (0:200:29799)/10000;
data_PSTH102 = histcounts(data_times,bin_edges);

figure;
plot(movmean(data_PSTH102,5))

%%
[y, Fs] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target1.wav');
y = y((0.25*Fs)-1:end);
y = downsample(movmean(envelope(y,11,'analytic'),1111),round(length(y)/length(data_PSTH))+1);
time = (0:length(y)-1) / Fs;
figure;
plot(time,y,'k')
xlim([0, time(end)])
xlabel('Time (s)')
yticklabels('')
yticks([])


%%
% Calculate SSE

sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\simulation_output';
sim_object = load(sim_location);

% Do selection
Choose_by = 'SPIKE';
selection = calculate_selction(Choose_by, sim_object, data_object);



tuning = lower(char(string(data_object.all_data(1).tuning_type)));
if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
spikes = data_object.all_data(1).ctrl_tar1_timestamps(:,focus);

for z = 1:10
    spikes{z+10} = find(sim_object.output(selection(1),z,1,:))/10000;
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

sse = sum((histcounts(data_times,bin_edges) - histcounts(sim_times,bin_edges)).^2);


%data_PSTHs = [data_PSTHs;movmean(histcounts(data_times,bin_edges),3)];
%sim_PSTHs = [sim_PSTHs;movmean(histcounts(sim_times,bin_edges),3)];


%%


sses = [];


tuning = lower(char(string(data_object.all_data(186).tuning_type)));
if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
spikes = data_object.all_data(186).ctrl_tar1_timestamps(:,focus);


for q = 1:21
    %figure;
    %subplot(11,1,1);
    %raster_plot_func(squeeze(Data_Rasters(186,:,:)))


    
    sse_mid = [];
        
    for k = 1:10
        %subplot(11,1,k+1)
        str = sprintf('%02d', k);
        sim_load = load(['C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Archive\july1_local_eligibility_backup_2026-07-15\Epoch_Rasters_Eprop\rasters_epoch_0',str,'.mat']);
        %raster_plot_func(squeeze(sim_load.output(q,:,:,:)))

        for z = 1:10
            spikes{z+10} = find(sim_load.output(q,z,1,:))/10000;
        end
    
        data_times = [];
        for m = 1:10
            data_times = [data_times;spikes{m}];
        end
        sim_times = [];
        for d = 11:20
            sim_times = [sim_times;spikes{d}];
        end
    
        sse = sum((histcounts(data_times,bin_edges) - histcounts(sim_times,bin_edges)).^2);
        sse_mid = [sse_mid,sse];

    end

    sses = [sses; sse_mid];
    
end

%%

sim_load = load('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\simulation_output.mat');

figure;
for k = 1:21
    subplot(21,1,k)
    spy(squeeze(sim_load.output(k,:,:,:)))
end

figure;
subplot(3,1,1)
spy(squeeze(Data_Rasters(186,:,:)))
subplot(3,1,2)
spy(squeeze(Data_Rasters(7,:,:)))
subplot(3,1,3)
spy(squeeze(Data_Rasters(102,:,:)))


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

    selection = [];
    if strcmp(choose_by, 'none')
        selection = repmat(1:12,220,1);
    elseif strcmp(choose_by, 'SSE')
        for k = 1:220
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
            sses = [];
            for m = 1:12
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
            for m = 1:12
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
        for k = 1
            distances = [];
            for m = 1:12
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
            for m = 1:12 %For all batches
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



