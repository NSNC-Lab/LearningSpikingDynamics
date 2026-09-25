

%Import stuff
close all; clear all;
%close all; clear all;
script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir); addpath(script_dir); end
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\200_epoch_Eprop_Subbatch_8.mat';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = split_and_load_large_sim_object(sim_location);
%Method by which to choose the simulation target
% none -- use all batches in the distrubutions
% SPIKE -- choose by spike distance
% SSE -- choose by SSE
% CV -- choose by CV
% NCCC -- choose by noise corrected prediction correlation

%Do selection
Choose_by = 'SPIKE';
selection = calculate_selction(Choose_by, sim_object, data_object);


%%

%Choices
%good_cells
%possibly_usable_cells
%all_cells
cell_choice = "all_cells";
Good_Cells = [7,19,30,33,52,53,61,68,69,77,96,102,106,108,124,126,127,128,129,130,132,133,149,186,200,210,218];
Possibly_usable_cells = [220,217,215,208,203,202,191,189,175,165,150,143,142,141,139,136,134,119,113,110,103,101,97,95,90,89,84,79,78,76,75,74,71,70,63,62,56,50,49,44,38,32,28,25,22,14,6];
Possibly_usable_cells = [Possibly_usable_cells,Good_Cells];

if strcmp(cell_choice, 'good_cells')
    choice_cells = Good_Cells;
    nbins = 10;

elseif strcmp(cell_choice, 'possibly_usable_cells')
    choice_cells = Possibly_usable_cells;
    nbins = 20;

elseif strcmp(cell_choice, 'all_cells')
    choice_cells = 1:220;
    nbins = 50;
end

NCCCs = nan(50, length(choice_cells));
TTRCs = nan(50, length(choice_cells));
mean_prediction_correlations = nan(50, length(choice_cells));

max_lag = 50;

for k = 1:length(choice_cells)
    tuning = lower(char(string(data_object.all_data(choice_cells(k)).tuning_type))) ;
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    spikes = data_object.all_data(choice_cells(k)).ctrl_tar1_timestamps(:,focus);
    
    %for m = 1:n_batches
    for z = 1:10
        spikes{z+10} = find(sim_object.output(selection(choice_cells(k)),z,choice_cells(k),:))/10000;
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


    
    for qq = 1:50

        window = qq:length(sim_PSTH)-max_lag-1+qq;

        [NCCCs(qq,k), TTRCs(qq,k), mean_prediction_correlations(qq,k)] = ...
            calculate_nccc_from_histograms(ris(:,window), sim_PSTH(1:end-max_lag));
    end

    %end
    %[val,idx] = max(NCCCs);
end

best_lags = [];
best_NCCCs = [];

for z = 1:length(choice_cells)
    [val,idx] = max(NCCCs(:,z));
    best_lags = [best_lags,idx];
    best_NCCCs = [best_NCCCs,val];
end



figure;
zero_NCCCs = NCCCs(1,:);
histogram(zero_NCCCs(isfinite(zero_NCCCs)),20); hold on
histogram(best_NCCCs(isfinite(best_NCCCs)),65); hold on
xlabel('Noise-corrected prediction correlation')
ylabel('Cell count')
title(sprintf('Valid cells: %d/%d', nnz(isfinite(best_NCCCs)), numel(best_NCCCs)))
xlim([-0.75,1.5])

figure;
histogram(best_lags);
title('best lags')

% 
% num_nans = nnz(isnan(NCCCs));
% idxs = find(isnan(NCCCs));
% 
% Data_Rasters = zeros([220,10,29801]);
% for k = 1:220
%     tuning = lower(char(string(data_object.all_data(k).tuning_type)));
%     if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
%     spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
%     for m = 1:10
%         spike_trial = spikes{m};
%         valid_spikes = round(spike_trial((spike_trial>0) & (spike_trial<2.9801))*10000);
%         Data_Rasters(k,m,valid_spikes) = 1;
%     end
% end
% 
% %%
% 
% threshold = 0.1;
% num_below = nnz(NCCCs<threshold);
% idx_below = find(NCCCs<threshold);
% 
% plot_panel = scroll_panel(num_nans);
% 
% for k = 1:num_nans
%     plot_axis = subplot(num_nans,2,(k-1)*2+1,'Parent',plot_panel);
%     Raster_matrix = squeeze(Data_Rasters(idxs(k),:,:));
%     raster_plot_func(plot_axis, Raster_matrix)
%     if k == 1
%         title(plot_axis, 'data')
%     end
%     ylabel(plot_axis, num2str(idxs(k)))
%     plot_axis = subplot(num_nans,2,(k-1)*2+2,'Parent',plot_panel);
%     Raster_matrix = squeeze(sim_object.output(selection(idxs(k)),:,idxs(k),:));
%     raster_plot_func(plot_axis, Raster_matrix)
%     if k == 1
%         title(plot_axis, 'sim')
%     end
% 
% end
% 
% plot_panel = scroll_panel(num_below);
% 
% for k = 1:num_below
%     plot_axis = subplot(num_below,2,(k-1)*2+1,'Parent',plot_panel);
%     Raster_matrix = squeeze(Data_Rasters(idx_below(k),:,:));
%     raster_plot_func(plot_axis, Raster_matrix)
%     if k == 1
%         title(plot_axis, 'data')
%     end
%     ylabel(plot_axis, num2str(idx_below(k)))
%     plot_axis = subplot(num_below,2,(k-1)*2+2,'Parent',plot_panel);
%     Raster_matrix = squeeze(sim_object.output(selection(idx_below(k)),:,idx_below(k),:));
%     raster_plot_func(plot_axis, Raster_matrix)
%     if k == 1
%         title(plot_axis, 'sim')
%     end
% 
% end

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
                for qq = 1:10
                    data_times = [data_times;spikes{qq}];
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
            NCCCs = nan(1, n_batches);
            prediction_correlations = nan(1, n_batches);
            for m = 1:n_batches
                for z = 1:10
                    spikes{z+10} = find(sim_object.output(m,z,k,:))/10000;
                end
                data_times = [];
                ris = [];
                bin_edges = (0:100:29799)/10000;
                for qq = 1:10
                    data_times = [data_times;spikes{qq}];
                    ris = [ris;histcounts(spikes{qq},bin_edges)];
                end
                
                sim_times = [];
                
                for d = 11:20
                    sim_times = [sim_times;spikes{d}];
                    
                end
                
                data_PSTH = histcounts(data_times,bin_edges);
                sim_PSTH = histcounts(sim_times,bin_edges);
                
                [NCCCs(m), ~, prediction_correlations(m)] = ...
                    calculate_nccc_from_histograms(ris, sim_PSTH);

            end
            valid = find(isfinite(NCCCs));
            if ~isempty(valid)
                [~, local_idx] = max(NCCCs(valid));
                idx = valid(local_idx);
            else
                % TTRC is a cell-specific scaling factor, so uncorrected
                % prediction correlation preserves the batch ranking when
                % the reliability correction itself is undefined.
                valid = find(isfinite(prediction_correlations));
                if ~isempty(valid)
                    [~, local_idx] = max(prediction_correlations(valid));
                    idx = valid(local_idx);
                else
                    idx = 1;
                    warning('NCCC:NoValidCorrelation', ...
                        'Cell %d has no valid NCCC or prediction correlation; using batch 1.', k);
                end
            end
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


function raster_plot_func(plot_axis, Raster_matrix)
    
    [trial,sample] = find(Raster_matrix);
    time = sample/10000;

    line(plot_axis, [time time]', ...
         [trial-0.5 trial+0.5]', ...
         'Color','k','LineWidth',0.6)

    xlim(plot_axis, [0 2.9801])
    ylim(plot_axis, [0.5 size(Raster_matrix,1)+0.5])
    %set(gca,'YDir','reverse','YTick',1:size(Raster_matrix,1))
    %xlabel('Time (s)')
    %ylabel('Trial')
    %title(['Cell ',cell_num,' — ',model_or_data])
    yticklabels(plot_axis, '')
    xticklabels(plot_axis, ' ')
    yticks(plot_axis, [])
    box(plot_axis, 'off')
end


function content = scroll_panel(num_rows)
    f = uifigure('Position',[100 100 900 700]);
    viewport = uipanel(f,'Position',[1 1 899 699], ...
        'Scrollable','on','BorderType','none');
    content = uipanel(viewport,'Units','pixels', ...
        'Position',[1 1 860 max(680,100*num_rows)], ...
        'BorderType','none','AutoResizeChildren','off');
end
