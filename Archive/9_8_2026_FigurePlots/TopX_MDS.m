%This should be a 6 tile plot of the MDS plots colored by FR, CV, and
%Lifetime Sparseness

%Import stuff
close all; clear all;
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\200_double_data_set.mat';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_location2 = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_cluster_info_atten_modMartin_OliverCriterion.mat';
data_object = load(data_location);
data_object2 = load(data_location2);
sim_object = split_and_load_large_sim_object(sim_location);
Distance_measure = 'RMSE';

%%

cell_choice = "good_cells";
Good_Cells = [7,19,30,33,52,53,61,68,69,77,96,102,106,108,124,126,127,128,129,130,132,133,149,186,200,210,218];
Good_Cells2 = [225,226,227,231,235,237,240,242,244,251,254,257,262,270,272,273,274,275,279,290,297,305,313,317,318,319,322,323,324,325,326,327,337,350,354, 361,362,363,364,380,386,388,400,403,407,410,411,413,422,429];
Good_Cells = [Good_Cells,Good_Cells2];

Possibly_usable_cells = [220,217,215,208,203,202,191,189,175,165,150,143,142,141,139,136,134,119,113,110,103,101,97,95,90,89,84,79,78,76,75,74,71,70,63,62,56,50,49,44,38,32,28,25,22,14,6];
Possibly_usable_cells = [Possibly_usable_cells,Good_Cells];

if strcmp(cell_choice, 'good_cells')
    choice_cells = Good_Cells;

elseif strcmp(cell_choice, 'possibly_usable_cells')
    choice_cells = Possibly_usable_cells;

elseif strcmp(cell_choice, 'all_cells')
    choice_cells = 1:cell_size;
end


%%
top_perc = 5; %Percentage closeness to the best fit.

%Do selection
Choose_by = 'SPIKE';
[selection,cell_identity] = calculate_selction_topX(Choose_by, sim_object, data_object,data_object2, top_perc);
%%

%Calculate MDS positions from joint Projection
[reduced_cell_identity,dis_mat] = calc_dis(Distance_measure,data_object,data_object2,sim_object,selection,cell_identity,choice_cells); 
[Y,stress] = mdscale(dis_mat,2, 'criterion','metricsstress');

% %Calculate feature distributions
% fr_data = [];
% fr_sim = [];
% cv_data = [];
% cv_sim = [];
% ls_data = [];
% ls_sim = [];
% re_data = [];
% re_sim = [];
% for k = 1:length(selection)
%     %Calculate Data spikes beforehand for efficiency
%     tuning = lower(char(string(data_object.all_data(cell_identity(k)).tuning_type)));
%     if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
%     spikes = data_object.all_data(cell_identity(k)).ctrl_tar1_timestamps(:,focus);
% 
%     fr_data = [fr_data, calc_fr_data(spikes)];
%     fr_sim = [fr_sim,calc_fr_sim(sim_object,k, selection,cell_identity)];
%     cv_data = [cv_data, calc_cv_data(spikes)];
%     cv_sim = [cv_sim,calc_cv_sim(sim_object,k, selection,cell_identity)];
%     ls_data = [ls_data, calc_ls_data(spikes)];
%     ls_sim = [ls_sim, calc_ls_sim(sim_object,k, selection,cell_identity);];
%     re_data = [re_data, calc_re_data(spikes)];
%     re_sim = [re_sim, calc_re_sim(sim_object,k, selection,cell_identity)];
% end

%%

%1. Go through all of the groups of batches
std_by_cell = [];
max_by_cell = [];

for k = choice_cells 
    %selections = selection(find(k == cell_identity));
    cell_Y = Y(find(k == reduced_cell_identity)+length(choice_cells),:); % Perhaps + length of choice cells since you appended that data onto the front?
    
    centroid = mean(cell_Y,1);
    distances = [];

    cells_size = size(cell_Y);


    for m = 1:cells_size(1)
        distances = [distances, sqrt(sum((cell_Y(m,:) - centroid).^2))];
    end

    std_by_cell = [std_by_cell,std(distances)];
    max_by_cell = [max_by_cell,max(distances)];
end

[val,~] = min(max_by_cell(max_by_cell>1));
idx3 = find(max_by_cell == val);

idx3 = find(choice_cells == 7);

[val,~] = max(max_by_cell(max_by_cell<2.2));
idx2 = find(max_by_cell == val);

idx2 = find(choice_cells == 33);

[val,idx] = max(std_by_cell);
selected_scatter = find(choice_cells(idx2) == reduced_cell_identity);
selected_scatter2 = find(choice_cells(idx3) == reduced_cell_identity);

close all
%Plot
figure(Position=[100,100,900,900]);
t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
ax(1) = nexttile;
scatter(Y(length(choice_cells):end,1),Y(length(choice_cells):end,2),15,'black','filled'); hold on
scatter(Y(length(choice_cells)+selected_scatter,1),Y(length(choice_cells)+selected_scatter,2),30,'green','filled'); hold on
scatter(Y(length(choice_cells)+selected_scatter2,1),Y(length(choice_cells)+selected_scatter2,2),30,'red','filled'); hold on
scatter(Y(idx2,1),Y(idx2,2),30,'cyan','filled'); hold on
scatter(Y(idx3,1),Y(idx3,2),30,'cyan','filled'); hold on
legend({'All Cells',['High spread cell: ',num2str(choice_cells(idx2))],['Low spread cell: ',num2str(choice_cells(idx3))],'Data'})
xlim([-2,6])
ylim([-4,4])
ax(2) = nexttile;
histogram(max_by_cell,'FaceColor',[0,0,0],'Normalization','probability')
ylabel('Probability')
xlabel('MDS distance')
title('Relative cluster spread distribution')


Data_Rasters = zeros([431,10,29801]);
for k = choice_cells
    if k > 220
            tuning = lower(char(string(data_object2.all_data(k-220).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object2.all_data(k-220).passive_tar1_timestamps(:,focus);
    
        else
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
    
        end
    for m = 1:10
        spike_trial = spikes{m};
        valid_spikes = round(spike_trial((spike_trial>=0.0001) & (spike_trial<2.9801))*10000);
        Data_Rasters(k,m,valid_spikes) = 1;
    end
end


col1 = tiledlayout(t,4,1,'TileSpacing','none','Padding','none');
col1.Layout.Tile = 3;

ax(3) = nexttile(col1);
Raster_matrix = squeeze(Data_Rasters(choice_cells(idx2),:,:));
raster_plot_func(Raster_matrix)

ylabel('Data')
title(['Cell: ',num2str(choice_cells(idx2))])

ax(4) = nexttile(col1);

%Find closest
dists = sum(([Y(length(choice_cells)+selected_scatter,1),Y(length(choice_cells)+selected_scatter,2)] - [Y(idx2,1),Y(idx2,2)]).^2,2);
[val,idx4] = min(dists);
used_idxs = find(choice_cells(idx2) == cell_identity);
closest_cell = selection(used_idxs(idx4));


Raster_matrix = squeeze(sim_object.output(closest_cell,:,choice_cells(idx2),:));
raster_plot_func(Raster_matrix)
ylabel('Closest within 5%')

ax(5) = nexttile(col1);

%Find Furthest within 5%
dists = sum(([Y(length(choice_cells)+selected_scatter,1),Y(length(choice_cells)+selected_scatter,2)] - [Y(idx2,1),Y(idx2,2)]).^2,2);
[val,idx5] = max(dists);
used_idxs = find(choice_cells(idx2) == cell_identity);
furthest_cell = selection(used_idxs(idx5));

Raster_matrix = squeeze(sim_object.output(furthest_cell,:,choice_cells(idx2),:));
raster_plot_func(Raster_matrix)
ylabel('Furthest within 5%')

col2 = tiledlayout(t,4,1,'TileSpacing','none','Padding','none');
col2.Layout.Tile = 4;

ax(6) = nexttile(col2);
Raster_matrix = squeeze(Data_Rasters(choice_cells(idx3),:,:));
raster_plot_func(Raster_matrix)

ylabel('Data')
title(['Cell: ',num2str(choice_cells(idx3))])

ax(7) = nexttile(col2);

%Find closest
dists = sum(([Y(length(choice_cells)+selected_scatter2,1),Y(length(choice_cells)+selected_scatter2,2)] - [Y(idx3,1),Y(idx3,2)]).^2,2);
[val,idx6] = min(dists);
used_idxs = find(choice_cells(idx3) == cell_identity);
closest_cell = selection(used_idxs(idx6));


Raster_matrix = squeeze(sim_object.output(closest_cell,:,choice_cells(idx3),:));
raster_plot_func(Raster_matrix)
ylabel('Closest within 5%')

ax(8) = nexttile(col2);

%Find Furthest within 5%
dists = sum(([Y(length(choice_cells)+selected_scatter2,1),Y(length(choice_cells)+selected_scatter2,2)] - [Y(idx3,1),Y(idx3,2)]).^2,2);
[val,idx7] = max(dists);
used_idxs = find(choice_cells(idx3) == cell_identity);
furthest_cell = selection(used_idxs(idx7));

Raster_matrix = squeeze(sim_object.output(furthest_cell,:,choice_cells(idx3),:));
raster_plot_func(Raster_matrix)
ylabel('Furthest within 5%')


sim_PSTHs = [];
data_PSTHs = [];
bin_edges = (0:100:29799)/10000;

for k = choice_cells
    if k > 220
        tuning = lower(char(string(data_object2.all_data(k-220).tuning_type)));
        if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
        spikes = data_object2.all_data(k-220).passive_tar1_timestamps(:,focus);

    else
        tuning = lower(char(string(data_object.all_data(k).tuning_type)));
        if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
        spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);

    end
    data_times = [];
    for m = 1:10
        data_times = [data_times;spikes{m}];
    end
    data_PSTHs = [data_PSTHs;histcounts(data_times,bin_edges)];
end

for k = 1:length(selection)
    
    spikes = {};
   
    
    for z = 1:10
            spikes{z} = find(sim_object.output(selection(k),z,cell_identity(k),:))/10000;
    end
    
    
    sim_times = [];
    for d = 1:10
        sim_times = [sim_times;spikes{d}];
    end
    
    
    sim_PSTHs = [sim_PSTHs;histcounts(sim_times,bin_edges)];
end



ax(9) = nexttile(col1);
used_idxs = find(choice_cells(idx2) == cell_identity);

smoothing = 5;

plot(movmean(sim_PSTHs(used_idxs(idx4),:),smoothing),'Color',[1,0.5,0]); hold on
plot(movmean(sim_PSTHs(used_idxs(idx5),:),smoothing),'b'); hold on
plot(movmean(data_PSTHs((idx2),:),smoothing),'r')
legend({'Closest','Furthest','Data'},Location="westoutside" )

ax(10) = nexttile(col2);
used_idxs = find(choice_cells(idx3) == cell_identity);

plot(movmean(sim_PSTHs(used_idxs(idx6),:),smoothing),'Color',[1,0.5,0]); hold on
plot(movmean(sim_PSTHs(used_idxs(idx7),:),smoothing),'b'); hold on
plot(movmean(data_PSTHs((idx3),:),smoothing),'r')


% ax(2) = nexttile;
% scatter(Y(1:length(selection),1),Y(1:length(selection),2),20,re_data,'filled');
% title('Reliability')
% xlim([-2,6])
% ylim([-4,4])
% ax(3) = nexttile;
% scatter(Y(1:length(selection),1),Y(1:length(selection),2),20,ls_data,'filled');
% title('Lifetime sparseness')
% xlim([-2,6])
% ylim([-4,4])
% ax(4) = nexttile;
% scatter(Y(1:length(selection),1),Y(1:length(selection),2),20,cv_data,'filled');
% title('CV')
% xlim([-2,6])
% ylim([-4,4])
% 
% ax(5) = nexttile;
% scatter(Y(length(selection)+1:length(selection)*2,1),Y(length(selection)+1:length(selection)*2,2),20,fr_sim,'filled');
% ylabel('Model','FontWeight','bold')
% xlim([-2,6])
% ylim([-4,4])
% ax(6) = nexttile;
% scatter(Y(length(selection)+1:length(selection)*2,1),Y(length(selection)+1:length(selection)*2,2),20,re_sim,'filled');
% xlim([-2,6])
% ylim([-4,4])
% ax(7) = nexttile;
% scatter(Y(length(selection)+1:length(selection)*2,1),Y(length(selection)+1:length(selection)*2,2),20,ls_sim,'filled');
% xlim([-2,6])
% ylim([-4,4])
% ax(8) = nexttile;
% scatter(Y(length(selection)+1:length(selection)*2,1),Y(length(selection)+1:length(selection)*2,2),20,cv_sim,'filled');
% xlim([-2,6])
% ylim([-4,4])
% sgtitle(t,'MDS projections colored by features')

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

function [reduced_cell_identity, dissimilarity_matrix] = calc_dis(Distance_measure,data_object,data_object2,sim_object,selection,cell_identity,choice_cells)
    

    if strcmp(Distance_measure,'RMSE')
        reduced_cell_identity = [];
        sim_PSTHs = [];
        data_PSTHs = [];
        bin_edges = (0:100:29799)/10000;

        for k = choice_cells
            if k > 220
                tuning = lower(char(string(data_object2.all_data(k-220).tuning_type)));
                if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
                spikes = data_object2.all_data(k-220).passive_tar1_timestamps(:,focus);
        
            else
                tuning = lower(char(string(data_object.all_data(k).tuning_type)));
                if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
                spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
        
            end
            data_times = [];
            for m = 1:10
                data_times = [data_times;spikes{m}];
            end
            data_PSTHs = [data_PSTHs;histcounts(data_times,bin_edges)];
        end

        for k = 1:length(selection)

            if ismember(cell_identity(k),choice_cells)
            
                spikes = {};
               
                
                for z = 1:10
                        spikes{z} = find(sim_object.output(selection(k),z,cell_identity(k),:))/10000;
                end
                
                
                sim_times = [];
                for d = 1:10
                    sim_times = [sim_times;spikes{d}];
                end
                
                
                sim_PSTHs = [sim_PSTHs;histcounts(sim_times,bin_edges)];
                reduced_cell_identity = [reduced_cell_identity, cell_identity(k)];
            end
        end

        %Merge for dual broadcast
        all_PSTH = [data_PSTHs; sim_PSTHs];
        
        dissimilarity_matrix = zeros([length(all_PSTH),length(all_PSTH)]);

        for k = 1:length(all_PSTH)
            for z  = 1:length(all_PSTH)
                dissimilarity_matrix(k,z) = sqrt(mean(((all_PSTH(k,:)-all_PSTH(z,:)).^2)));
            end
        end
    
    end

end


function [selection,cell_identity] = calculate_selction_topX(choose_by, sim_object, data_object, data_object2, top_perc)
    
    output_size = size(sim_object.output);    
    n_batches = output_size(1);
    cell_identity = [];

    selection = [];
    if strcmp(choose_by, 'SPIKE')
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

            %So this gives us our best
            [val,idx] = min(distances);

            %Calculate 5% margin cutoff of best
            cutoff = val + (top_perc/100)*val;
            choices = find(distances < cutoff);

            selection = [selection, choices];
            for mmm = 1:length(choices)
                cell_identity = [cell_identity,k];
            end
        end

        for k = 1:211
            distances = [];
            for m = 1:n_batches
                tuning = lower(char(string(data_object2.all_data(k).tuning_type)));
                if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
                spikes = data_object2.all_data(k).passive_tar1_timestamps(:,focus);
                for z = 1:10
                    spikes{z+10} = find(sim_object.output(m,z,k+220,:))/10000;
                end
                spikes = cellfun(@transpose, spikes, 'UniformOutput', false)';
                STS = SpikeTrainSet(spikes,0,3);
                dist_mat = STS.SPIKEdistanceMatrix();
                distances = [distances,mean(mean(dist_mat(1:10,11:20)))];
            end

            %So this gives us our best
            [val,idx] = min(distances);

            %Calculate 5% margin cutoff of best
            cutoff = val + (top_perc/100)*val;
            choices = find(distances < cutoff);

            selection = [selection, choices];
            for mmm = 1:length(choices)
                cell_identity = [cell_identity,k+220];
            end
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
    re_val = mean(dist_mat(triu(true(size(dist_mat)),1)),'omitnan');

end


function re_val = calc_re_sim(sim_object,k, selection,cell_identity)
    
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
            spikes{z} = find(sim_object.output(m,z,cell_identity(k),:))/10000;
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


function ls_val = calc_ls_sim(sim_object,k, selection,cell_identity)
    
    
    lss = [];
    for m = 1:12
        spikes = {};
        for z = 1:10
            spikes{z} = find(sim_object.output(m,z,cell_identity(k),:))/10000;
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

function cv_val = calc_cv_sim(sim_object,k, selection,cell_identity)
    cv_val = [];
    for m = 1:12 %For all batches
        isi_s = []; 
        for z  = 1:10 % For all trials
            isi_s = [isi_s,squeeze(diff(find(sim_object.output(m,z,cell_identity(k),:)))/10000)'];
        end
        cv_val = [cv_val,std(isi_s)/mean(isi_s)];
    end
    if nnz(size(selection) > 1) > 1
        cv_val = cv_val(selection(k,:));
    else
        cv_val = cv_val(selection(k));
    end
end


function Hz = calc_fr_sim(sim_object,k, selection,cell_identity)
    if nnz(size(selection) > 1) > 1
        Hz = (sum(sum(sim_object.output(selection(k,:),:,cell_identity(k),:),2),4)/10/2.9801)';
    else
        Hz = (sum(sum(sim_object.output(selection(k),:,cell_identity(k),:),2),4)/10/2.9801)';;
    end
    
end

function Hz = calc_fr_data(spikes)
    Hz = sum(cellfun(@(x) nnz(x >= 0 & x <= 2.9801), spikes)/2.9801)/10; %Get in terms of spikes per second
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
