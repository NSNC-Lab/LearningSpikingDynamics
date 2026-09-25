%close all; clear all;
script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir); addpath(script_dir); end
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\rasters_epoch_135.mat';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = split_and_load_large_sim_object(sim_location);
%Method by which to choose the simulation target
% none -- use all batches in the distrubutions
% SPIKE -- choose by spike distance
% SSE -- choose by SSE
% CV -- choose by CV
% NCCC -- choose by noise corrected prediction correlation
choose_by = 'SPIKE';
%%

%Distributions
% 1. Noise corrected split half correlation
% 2. Firing Rate
% 3. CV

selection = calculate_selction(choose_by, sim_object, data_object);

%%

%Choices
%good_cells
%possibly_usable_cells
%all_cells
cell_choice = "good_cells";
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


%%

fr_data = [];
fr_sim = [];
cv_data = [];
cv_sim = [];
ls_data = [];
ls_sim = [];
re_data = [];
re_sim = [];

d1_data = [];
d2_data = [];

s1_data = [];
s2_data = [];

for k = choice_cells
    %Calculate Data spikes beforehand for efficiency
    tuning = lower(char(string(data_object.all_data(k).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);

    fr_data = [fr_data, calc_fr_data(spikes)];
    fr_sim = [fr_sim,calc_fr_sim(sim_object,k, selection)];
    cv_data = [cv_data, calc_cv_data(spikes)];
    cv_sim = [cv_sim,calc_cv_sim(sim_object,k, selection)];
    ls_data = [ls_data, calc_ls_data(spikes)];
    ls_sim = [ls_sim, calc_ls_sim(sim_object,k, selection)];
    re_data = [re_data, calc_re_data(spikes)];
    re_sim = [re_sim, calc_re_sim(sim_object,k, selection)];
    subset1 = randperm(10,5);
    possible_trials = 1:10;
    subset2 = possible_trials(~ismember(possible_trials,subset1));
    d1_data = [d1_data, spike_rel_calc(spikes([subset1]))];
    d2_data = [d2_data, spike_rel_calc(spikes([subset2]))];
    subset1 = randperm(8,4);
    possible_trials2 = 1:8;
    subset2 = possible_trials2(~ismember(possible_trials2,subset1));
    s1_data = [s1_data, split_rel_calc(spikes([subset1]))];
    s2_data = [s2_data, split_rel_calc(spikes([subset2]))];
end

%%
%Plots

%Single plots

%Plot fr
%single_plot(fr_sim,fr_data, choose_by, 'Firing Rate', 75)
%Plot CV
%single_plot(cv_sim,cv_data, choose_by, 'CV', 3)
%Plot LS
%single_plot(ls_sim,ls_data, choose_by, 'Lifetime Sparseness', 1)
%Plot RE
%single_plot(re_sim,re_data, choose_by, 'Split-Half Reliability', 1)
%Plot data-data RE
%single_plot(d1_data,d2_data, choose_by, 'SPIKE Reliability', 1)

%Figure 3 Plot (Combined, currently choosing by SPIKE-distance)

plot_fig_3(fr_data,fr_sim,cv_data,cv_sim,ls_data,ls_sim,re_data,re_sim,choose_by,nbins)

%dual_plot_reliability(d1_data,d2_data,s1_data,s2_data,choose_by)


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


function re_val = calc_re_data(spikes)

    
    
    %Split half reliability with Spearman Brown correction
    % train1 = [];
    % train2 = [];
    % for k = 1:10
    %     if mod(k,2) == 0
    %         train1 = [train1;spikes{k}];
    %     else
    %         train2 = [train2;spikes{k}];
    %     end
    % end
    % 
    % bin_edges = (0:100:29799)/10000;
    % data_PSTH1 = histcounts(train1,bin_edges);
    % data_PSTH2 = histcounts(train2,bin_edges);
    % 
    % re_val_no_spear = corr(data_PSTH1',data_PSTH2');
    % 
    % re_val = 2*re_val_no_spear/(1 + re_val_no_spear);


    %Using UT of spike distance mat
    spikes = cellfun(@transpose, spikes, 'UniformOutput', false)';
    STS = SpikeTrainSet(spikes,0,3);
    dist_mat = STS.SPIKEdistanceMatrix();
    re_val = 1 - mean(dist_mat(triu(true(size(dist_mat)),1)),'omitnan');

end


function re_val = calc_re_sim(sim_object,k, selection)
    
    %Split Half
    % re_vals = [];
    % for m = 1:12
    %     spikes = {};
    %     for z = 1:10
    %         spikes{z} = find(sim_object.output(m,z,k,:))/10000;
    %     end
    %     train1 = [];
    %     train2 = [];
    %     for qq = 1:10
    %         if mod(qq,2) == 0
    %             train1 = [train1;spikes{qq}];
    %         else
    %             train2 = [train2;spikes{qq}];
    %         end
    %     end
    %     bin_edges = (0:100:29799)/10000;
    %     sim_PSTH1 = histcounts(train1,bin_edges);
    %     sim_PSTH2 = histcounts(train2,bin_edges);
    % 
    %     re_val_no_spear = corr(sim_PSTH1',sim_PSTH2');
    % 
    %     re_vals = [re_vals, 2*re_val_no_spear/(1 + re_val_no_spear)];
    % 
    % end
    % 
    % if nnz(size(selection) > 1) > 1
    %     re_val = re_vals(selection(k,:));
    % else
    %     re_val = re_vals(selection(k));
    % end

    output_size = size(sim_object.output);
    n_batches = output_size(1);



    %Using UT of spike distance mat
    re_vals = [];
    for m = 1:n_batches
        spikes = {};
        for z = 1:10
            spikes{z} = find(sim_object.output(m,z,k,:))/10000;
        end

        spikes = cellfun(@transpose, spikes, 'UniformOutput', false);
        STS = SpikeTrainSet(spikes,0,3);
        dist_mat = STS.SPIKEdistanceMatrix();
        re_vals = [re_vals,1 - mean(dist_mat(triu(true(size(dist_mat)),1)),'omitnan')];

    end

    if nnz(size(selection) > 1) > 1
        re_val = re_vals(selection(k,:));
    else
        re_val = re_vals(selection(k));
    end

end

function ls_val = calc_ls_data(spikes)
    data_times = [];
    for m = 1:10
        data_times = [data_times;spikes{m}];
    end
    bin_edges = (0:100:29799)/10000;
    data_PSTH = histcounts(data_times,bin_edges);
    N = length(data_PSTH);
    ls_val = (1-((((1/N)*sum(data_PSTH))^2)/(((1/N)*sum(data_PSTH.^2)))))/(1-(1/N));
end


function ls_val = calc_ls_sim(sim_object,k, selection)
    
    
    output_size = size(sim_object.output);
    n_batches = output_size(1);

    lss = [];
    for m = 1:n_batches
        spikes = {};
        for z = 1:10
            spikes{z} = find(sim_object.output(m,z,k,:))/10000;
        end
        sim_times = [];
        for d = 1:10
            sim_times = [sim_times;spikes{d}];
        end
        bin_edges = (0:100:29799)/10000;
        sim_PSTH = histcounts(sim_times,bin_edges);
        N = length(sim_PSTH);
        lss = [lss, (1-((((1/N)*sum(sim_PSTH))^2)/(((1/N)*sum(sim_PSTH.^2)))))/(1-(1/N))]; 
    end


    if nnz(size(selection) > 1) > 1
        ls_val = lss(selection(k,:));
    else
        ls_val = lss(selection(k));
    end
    
end

function cv_val = calc_cv_data(spikes)
    isiarr = cell2mat(cellfun(@(x) diff(x(x >= 0 & x <= 2.9801)),spikes, UniformOutput=false));
    cv_val = std(isiarr)/mean(isiarr);
end

function cv_val = calc_cv_sim(sim_object,k, selection)
    

    output_size = size(sim_object.output);
    n_batches = output_size(1);

    cv_val = [];
    for m = 1:n_batches %For all batches
        isi_s = []; 
        for z  = 1:10 % For all trials
            isi_s = [isi_s,squeeze(diff(find(sim_object.output(m,z,k,:)))/10000)'];
        end
        cv_val = [cv_val,std(isi_s)/mean(isi_s)];
    end
    if nnz(size(selection) > 1) > 1
        cv_val = cv_val(selection(k,:));
    else
        cv_val = cv_val(selection(k));
    end
end


function Hz = calc_fr_sim(sim_object,k, selection)

    if nnz(size(selection) > 1) > 1
        Hz = (sum(sum(sim_object.output(selection(k,:),:,k,:),2),4)/10/2.9801)';
    else
        Hz = (sum(sum(sim_object.output(selection(k),:,k,:),2),4)/10/2.9801)';;
    end
    
end

function Hz = calc_fr_data(spikes)
    Hz = sum(cellfun(@(x) nnz(x >= 0 & x <= 2.9801), spikes)/2.9801)/10; %Get in terms of spikes per second
end


function plot_fig_3(fr_data,fr_sim,cv_data,cv_sim,ls_data,ls_sim,re_data,re_sim, choose_by,nbins)

    fr_lower = 0;
    fr_upper = 65;
    re_lower = 0.6;
    re_upper = 1;
    ls_lower = 0;
    ls_upper = 1;
    cv_lower = 0;
    cv_upper = 3;

    figure(Position=[200,200,1800,900]);
    subplot(2,4,1);
    histogram(fr_data,NumBins=nbins, Normalization="probability",FaceColor=[0.5,0.5,0.5]); hold on
    histogram(fr_sim,NumBins=nbins, Normalization="probability",FaceColor=[0.0,0.1,0.3])
    legend({'Data', 'Model'})
    title('Firing Rate probability Distribution')
    xlim([fr_lower, fr_upper])
    subplot(2,4,4);
    histogram(cv_data,NumBins=nbins, Normalization="probability",FaceColor=[0.5,0.5,0.5]); hold on
    histogram(cv_sim,NumBins=nbins, Normalization="probability",FaceColor=[0.0,0.1,0.3])
    legend({'Data', 'Model'})
    title('CV probability Distribution')
    xlim([cv_lower, cv_upper])
    subplot(2,4,3);
    histogram(ls_data,NumBins=nbins, Normalization="probability",FaceColor=[0.5,0.5,0.5]); hold on
    histogram(ls_sim,NumBins=nbins, Normalization="probability",FaceColor=[0.0,0.1,0.3])
    legend({'Data', 'Model'})
    title('Lifetime Sparseness probability Distribution')
    xlim([ls_lower, ls_upper])
    subplot(2,4,2);
    histogram(re_data,NumBins=nbins, Normalization="probability",FaceColor=[0.5,0.5,0.5]); hold on
    histogram(re_sim,NumBins=nbins, Normalization="probability",FaceColor=[0.0,0.1,0.3])
    legend({'Data', 'Model'})
    title('SPIKE reliability probability Distribution')
    xlim([re_lower, re_upper])
    subplot(2,4,5);
    R2 = 1 - sum((fr_data - fr_sim).^2) / sum((fr_data - mean(fr_data)).^2);
    scatter(fr_data, fr_sim,3,'black','filled'); hold on;
    plot(fr_lower:fr_upper,fr_lower:fr_upper,'k--'); hold on
    create_contour(fr_lower,fr_upper,fr_data,fr_sim)
    title(['FR scatter'])
    xlabel('Data')
    ylabel('Model')
    xlim([fr_lower, fr_upper])
    ylim([fr_lower, fr_upper])
    subplot(2,4,8);
    %Protect against NANs (cell 43 is very low firing rate and the best
    %apporximation was non-firing)
    cv_data = cv_data(~isnan(cv_sim));
    cv_sim = cv_sim(~isnan(cv_sim));
    R2 = 1 - sum((cv_data - cv_sim).^2) / sum((cv_data - mean(cv_data)).^2);
    scatter(cv_data, cv_sim,3,'black','filled'); hold on;
    plot(cv_lower:cv_upper,cv_lower:cv_upper,'k--')
    create_contour(cv_lower,cv_upper,cv_data,cv_sim)
    title(['CV scatter'])
    xlabel('Data')
    ylabel('Model')
    xlim([cv_lower, cv_upper])
    ylim([cv_lower, cv_upper])
    subplot(2,4,7);
    ls_data = ls_data(~isnan(ls_sim));
    ls_sim = ls_sim(~isnan(ls_sim));
    R2 = 1 - sum((ls_data - ls_sim).^2) / sum((ls_data - mean(ls_data)).^2);
    scatter(ls_data, ls_sim,3,'black','filled'); hold on;
    plot(ls_lower:ls_upper,ls_lower:ls_upper,'k--')
    create_contour(ls_lower,ls_upper,ls_data,ls_sim)
    title(['LS scatter'])
    xlabel('Data')
    ylabel('Model')
    xlim([ls_lower, ls_upper])
    ylim([ls_lower, ls_upper])
    sgtitle(['Feature Distribution Plots. Chosen by : [', choose_by,']'])
    subplot(2,4,6);
    re_data = re_data(~isnan(re_sim));
    re_sim = re_sim(~isnan(re_sim));
    R2 = 1 - sum((re_data - re_sim).^2) / sum((re_data - mean(re_data)).^2);
    scatter(re_data, re_sim,3,'black','filled'); hold on;
    create_contour(re_lower,re_upper,re_data,re_sim)
    plot(re_lower:0.01:re_upper,re_lower:0.01:re_upper,'k--')
    title(['RE scatter'])
    xlabel('Data')
    ylabel('Moel')
    xlim([re_lower, re_upper])
    ylim([re_lower, re_upper])
    sgtitle(['Feature Distribution Plots. Chosen by : [', choose_by,']'])
end

function create_contour(lower_limit,upper_limit,data_val,sim_val)

    edges = linspace(lower_limit,upper_limit,50);
    counts = histcounts2(data_val,sim_val,edges,edges);
    
    %Greate guassian smoothing kernel
    %g_filt = [1,4,7,4,1;4,16,26,16,4;7,26,41,26,7;4,16,26,16,4;1,4,7,4,1];
    
    x = 0:0.1:10;
    mu = 5;
    sigma = 0.25;

    g = normpdf(x,mu,sigma);

    g_filt = g'*g;


    %2D convolve counts to create a bloom and smooth affect.
    counts = conv2(counts,g_filt,"same");

    centers = edges(1:end-1) + diff(edges)/2;

    hold on
    h = imagesc(centers,centers,counts');
    set(gca,'YDir','normal')

    h.AlphaData = 0.35*(counts' > 0);
    uistack(h,'bottom')
    
    colormap("parula")

end


function single_plot(sim,data,choose_by, feature, scatter_limit)
    figure(Position=[400,400,600,300]);
    subplot(1,2,1);
    histogram(sim,NumBins=50, Normalization="probability"); hold on;
    histogram(data,NumBins=50, Normalization="probability");
    legend({'Sim', 'Data'})
    title([feature, ' probability Distribution'])
    subplot(1,2,2)
    if strcmp(choose_by, 'none')
        scatter(repelem(data, 12), sim); hold on;
    else
        scatter(data, sim); hold on;
    end
    %plot(0:scatter_limit,0:scatter_limit,'k--')
    plot(-0.3:scatter_limit+2,-0.3:scatter_limit+2,'k--')
    title([feature, ' scatter'])
    xlabel('Data')
    ylabel('Sim')
    %xlim([0, scatter_limit])
    %ylim([0, scatter_limit])
    xlim([0, 0.5])
    ylim([0, 0.5])
    sgtitle([feature,' plots. Chosen by : [', choose_by, ']'])
end

function dual_plot_reliability(SPIKE_data1,SPIKE_data2,split_data1,split_data2,choose_by)
    figure(Position=[400,400,600,300]);
    subplot(2,2,1);
    histogram(SPIKE_data1,NumBins=50, Normalization="probability"); hold on;
    histogram(SPIKE_data2,NumBins=50, Normalization="probability");
    legend({'Data1', 'Data2'})
    title('probability Distribution: Data Group 1 [SPIKE]')
    subplot(2,2,2);
    histogram(split_data1,NumBins=50, Normalization="probability"); hold on;
    histogram(split_data2,NumBins=50, Normalization="probability");
    legend({'Data1', 'Data2'})
    title('probability Distribution: Data Group 1 [Split-half]')
    subplot(2,2,3)
    if strcmp(choose_by, 'none')
        scatter(repelem(SPIKE_data1, 12), SPIKE_data2); hold on;
    else
        scatter(SPIKE_data1, SPIKE_data2); hold on;
    end
    plot(0:0.01:0.5,0:0.01:0.5,'k--')
    title('[SPIKE] scatter')
    xlabel('SPIKE_data1')
    ylabel('SPIKE_data2')
    xlim([0, 0.5])
    ylim([0, 0.5])
    subplot(2,2,4)
    if strcmp(choose_by, 'none')
        scatter(repelem(split_data1, 12), split_data2); hold on;
    else
        scatter(split_data1, split_data2); hold on;
    end
    plot(-0.3:0.01:1,-0.3:0.01:1,'k--')
    title('[Split-half] scatter')
    xlabel('split_data1')
    ylabel('split_data2')
    xlim([-0.3, 1])
    ylim([-0.3, 1])
    sgtitle('SPIKE and Split-half data vs data')
end

function re_val = split_rel_calc(spikes)
    train1 = [];
    train2 = [];
    for k = 1:4
        if mod(k,2) == 0
            train1 = [train1;spikes{k}];
        else
            train2 = [train2;spikes{k}];
        end
    end
    bin_edges = (0:100:29799)/10000;
    data_PSTH1 = histcounts(train1,bin_edges);
    data_PSTH2 = histcounts(train2,bin_edges);

    re_val_no_spear = corr(data_PSTH1',data_PSTH2');
    re_val = 2*re_val_no_spear/(1 + re_val_no_spear);

end



function re_val = spike_rel_calc(spikes)
    spikes = cellfun(@transpose, spikes, 'UniformOutput', false)';
    STS = SpikeTrainSet(spikes,0,3);
    dist_mat = STS.SPIKEdistanceMatrix();
    re_val = 1- mean(dist_mat(triu(true(size(dist_mat)),1)),'omitnan');
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
