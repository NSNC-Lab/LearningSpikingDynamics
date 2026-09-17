%Bring stuff in
%close all; clear all;
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_all_cells_Eprop';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = load(sim_location);
use_NaNs = 'true';

color_vec = [];
for k = 1:220
    if strcmp(data_object.all_data(k).layer, 'L2/3'); color_vec = [color_vec, 1]; end
    if strcmp(data_object.all_data(k).layer, 'L4'); color_vec = [color_vec, 2]; end
    if strcmp(data_object.all_data(k).layer, 'L5/6'); color_vec = [color_vec, 3]; end
    if strcmp(data_object.all_data(k).layer, 'NaN'); color_vec = [color_vec, 4]; end
end

% Do selection
Choose_by = 'SPIKE';
selection = calculate_selction(Choose_by, sim_object, data_object);
%%
[Feature_Matrix,Parameter_matrix] = Scatter_analysis(sim_object,selection,color_vec,use_NaNs);

if strcmp(use_NaNs, 'false')
    Feature_Matrix = Feature_Matrix(find(color_vec ~= 4),:);
    Parameter_matrix = Parameter_matrix(find(color_vec ~= 4),:);
    color_vec = color_vec(find(color_vec ~= 4));
end

figure(Position=[100,100,1200,800])
for k = 1:12
    for m = 1:4
        subplot(4,12,4*(k-1) + m)
        scatter(Feature_Matrix(:,m), Parameter_matrix(:,k), 10, color_vec)
    end
end

figure(Position=[100,100,1200,800])
for k = 1:12
    for m = 1:12
        subplot(12,12,12*(k-1) + m)
        scatter(Parameter_matrix(:,m), Parameter_matrix(:,k), 10, color_vec)
    end
end

function [Feature_Matrix,Parameter_matrix] = Scatter_analysis(sim_object,selection,layer_vec,use_NaNs)
        
    %Create a matrix that we can actually use in order to do this
    %analysis. It should be 14 params by 220 cells
    
    holder_params = zeros([220,12]);
    field_names = fields(sim_object.params);
    for k = 1:length(fields(sim_object.params))
        values = sim_object.params.(field_names{k});
        for m = 1:220
            holder_params(m,k) = squeeze(values(selection(m),m,end));
        end
    end
    
    %Z-score
    Z = zscore(holder_params,0,1); 
    Parameter_matrix = Z;

    %Calculate Features
    fr_sim = [];
    cv_sim = [];
    ls_sim = [];
    re_sim = [];
    
    for k = 1:220
        %Calculate Data spikes beforehand for efficiency
        %tuning = lower(char(string(data_object.all_data(k).tuning_type)));
        %if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
        %spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
        fr_sim = [fr_sim,calc_fr_sim(sim_object,k, selection)];
        cv_sim = [cv_sim,calc_cv_sim(sim_object,k, selection)];
        ls_sim = [ls_sim, calc_ls_sim(sim_object,k, selection)];
        re_sim = [re_sim, calc_re_sim(sim_object,k, selection)];
    end

    Feature_Matrix = [fr_sim',cv_sim',ls_sim',re_sim'];

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
        for k = 1:220
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


function Hz = calc_fr_sim(sim_object,k, selection)
    if nnz(size(selection) > 1) > 1
        Hz = (sum(sum(sim_object.output(selection(k,:),:,k,:),2),4)/10/2.9801)';
    else
        Hz = (sum(sum(sim_object.output(selection(k),:,k,:),2),4)/10/2.9801)';;
    end
end

function cv_val = calc_cv_sim(sim_object,k, selection)
    cv_val = [];
    for m = 1:12 %For all batches
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

function ls_val = calc_ls_sim(sim_object,k, selection)
    lss = [];
    for m = 1:12
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


    %Using UT of spike distance mat
    re_vals = [];
    for m = 1:12
        spikes = {};
        for z = 1:10
            spikes{z} = find(sim_object.output(m,z,k,:))/10000;
        end

        spikes = cellfun(@transpose, spikes, 'UniformOutput', false);
        STS = SpikeTrainSet(spikes,0,3);
        dist_mat = STS.SPIKEdistanceMatrix();
        re_vals = [re_vals,mean(dist_mat(triu(true(size(dist_mat)),1)),'omitnan')];

    end

    if nnz(size(selection) > 1) > 1
        re_val = re_vals(selection(k,:));
    else
        re_val = re_vals(selection(k));
    end

end