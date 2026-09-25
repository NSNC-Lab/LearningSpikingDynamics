%This should be a 6 tile plot of the MDS plots colored by FR, CV, and
%Lifetime Sparseness

%Import stuff
close all; clear all;
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\rasters_epoch_135.mat';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = split_and_load_large_sim_object(sim_location);
Distance_measure = 'RMSE';

%Do selection
Choose_by = 'SPIKE';
selection = calculate_selction(Choose_by, sim_object, data_object);

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

elseif strcmp(cell_choice, 'possibly_usable_cells')
    choice_cells = Possibly_usable_cells;

elseif strcmp(cell_choice, 'all_cells')
    choice_cells = 1:220;
end

%Calculate MDS positions from joint Projection
dis_mat = calc_dis(Distance_measure,data_object,sim_object,selection,Good_Cells,Possibly_usable_cells,cell_choice); 
[Y,stress] = mdscale(dis_mat,2, 'criterion','metricsstress');

%Calculate feature distributions
fr_data = [];
fr_sim = [];
cv_data = [];
cv_sim = [];
ls_data = [];
ls_sim = [];
re_data = [];
re_sim = [];
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
    ls_sim = [ls_sim, calc_ls_sim(sim_object,k, selection);];
    re_data = [re_data, calc_re_data(spikes)];
    re_sim = [re_sim, calc_re_sim(sim_object,k, selection)];
end

%%
xlim_val = -2;
xlim_val2 = 6;

close all
%Plot
figure(Position=[100,100,1500,750]);
t = tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
ax(1) = nexttile;
scatter(Y(1:length(Y)/2,1),Y(1:length(Y)/2,2),20,fr_data,'filled');
title('Firing rate')
ylabel('Data','FontWeight','bold')
xlim([xlim_val,xlim_val2])
ylim([-4,4])
colorbar;
colormap('parula')
ax(2) = nexttile;
scatter(Y(1:length(Y)/2,1),Y(1:length(Y)/2,2),20,re_data,'filled');
title('Reliability')
xlim([xlim_val,xlim_val2])
ylim([-4,4])
colorbar;
ax(3) = nexttile;
scatter(Y(1:length(Y)/2,1),Y(1:length(Y)/2,2),20,ls_data,'filled');
title('Lifetime sparseness')
xlim([xlim_val,xlim_val2])
ylim([-4,4])
colorbar;
ax(4) = nexttile;
scatter(Y(1:length(Y)/2,1),Y(1:length(Y)/2,2),20,cv_data,'filled');
title('CV')
xlim([xlim_val,xlim_val2])
ylim([-4,4])
colorbar;

ax(5) = nexttile;
scatter(Y(length(Y)/2+1:length(Y),1),Y(length(Y)/2+1:length(Y),2),20,fr_sim,'filled');
ylabel('Model','FontWeight','bold')
xlim([xlim_val,xlim_val2])
ylim([-4,4])
colorbar;
ax(6) = nexttile;
scatter(Y(length(Y)/2+1:length(Y),1),Y(length(Y)/2+1:length(Y),2),20,re_sim,'filled');
xlim([xlim_val,xlim_val2])
ylim([-4,4])
colorbar;
ax(7) = nexttile;
scatter(Y(length(Y)/2+1:length(Y),1),Y(length(Y)/2+1:length(Y),2),20,ls_sim,'filled');
xlim([xlim_val,xlim_val2])
ylim([-4,4])
colorbar;
ax(8) = nexttile;
scatter(Y(length(Y)/2+1:length(Y),1),Y(length(Y)/2+1:length(Y),2),20,cv_sim,'filled');
xlim([xlim_val,xlim_val2])
ylim([-4,4])
colorbar;
sgtitle(t,'MDS projections colored by features')





% %Small little correlation table
% safeCorr = @(x,y) corr(x(:),y(:),'Rows','complete');
% 
% R = [
%     safeCorr(Y(1:220,1),   fr_data), ...
%     safeCorr(Y(1:220,1), re_data), ...
%     safeCorr(Y(1:220,1),   ls_data), ...
%     safeCorr(abs(Y(1:220,2)), cv_data);
% 
%     safeCorr(Y(221:440,1), fr_sim), ...
%     safeCorr(Y(221:440,1), re_sim), ...
%     safeCorr(Y(221:440,1), ls_sim), ...
%     safeCorr(abs(Y(221:440,2)), cv_sim)
% ];
% 
% T = array2table(R, ...
%     'VariableNames',{'FiringRate','Reliability','LifetimeSparseness','CV'}, ...
%     'RowNames',{'Data','Model'});
% fig = uifigure('Position',[500 500 650 70]);
% 
% uit = uitable(fig, ...
%     'Data',T, ...
%     'Position',[0 0 650 70]);


function dissimilarity_matrix = calc_dis(Distance_measure,data_object,sim_object,selection,Good_Cells,Possibly_usuable_cells,cell_choice)
    

    if strcmp(cell_choice, 'good_cells')
        choices = Good_Cells;
    elseif strcmp(cell_choice, 'possibly_usable_cells')
        choices = Possibly_usuable_cells;
    elseif strcmp(cell_choice, 'all_cells')
        choices = 1:220;
    else
        disp('pick valid choice')
    end

    if strcmp(Distance_measure,'RMSE')

        sim_PSTHs = [];
        data_PSTHs = [];

        for k = choices
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
            
            for z = 1:10
                spikes{z+10} = find(sim_object.output(selection(k),z,k,:))/10000;
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
            data_PSTHs = [data_PSTHs;histcounts(data_times,bin_edges)];
            sim_PSTHs = [sim_PSTHs;histcounts(sim_times,bin_edges)];
        end

        %Merge for dual broadcast
        all_PSTH = [data_PSTHs; sim_PSTHs];
        

        size_val = size(all_PSTH);
        dissimilarity_matrix = zeros([size_val(1),size_val(1)]);
        
        for k = 1:size_val(1)
            for z  = 1:size_val(1)
                dissimilarity_matrix(k,z) = sqrt(mean(((all_PSTH(k,:)-all_PSTH(z,:)).^2)));
            end
        end
    
    end

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