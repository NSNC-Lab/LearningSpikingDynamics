%Start with a deul projection of the distance matrix for the 100ms PSTH
%RMSE bewteen the data and the latest EPROP run

%Note: Could also try SPIKE distance as dissimilarity metric

%Import stuff
close all; clear all;
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_all_cells_Eprop';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = load(sim_location);
Distance_measure = 'RMSE';

%Do selection
Choose_by = 'SPIKE';
selection = calculate_selction(Choose_by, sim_object, data_object);

%%

%Calculate the dissimilarity matrix
dis_mat = calc_dis(Distance_measure,data_object,sim_object,selection); 
%%
%Do MDS projection
[Y,stress] = mdscale(dis_mat,2, 'criterion','metricsstress');
%Calculate distances (euclidean)
distances = sqrt((Y(1:220,1)-Y(221:440,1)).^2 + (Y(1:220,2)-Y(221:440,2)).^2);
median_val = median(distances);

%Compute Data Rasters
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

%Calcualte PSTHs for plotting
sim_PSTHs = [];
data_PSTHs = [];

for k = 1:220
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
    bin_edges = (0:200:29799)/10000;
    data_PSTHs = [data_PSTHs;movmean(histcounts(data_times,bin_edges),3)];
    sim_PSTHs = [sim_PSTHs;movmean(histcounts(sim_times,bin_edges),3)];
end



%%
close all;
%Plot
%Label a cell
Labeled_cell = 7;
Labeled_cell2 = 19;
Labeled_cell3 = 61;

figure(Position = [400,100,1000,800]);
t = tiledlayout(5,3,'TileSpacing','compact','Padding','compact');
ax(1) = nexttile([2,1]);
%subplot(10,3,[1,4,7,10])
%Data -> 1-220
%Sim -> 221-440
scatter(Y(1:220,1),Y(1:220,2),20,'black','filled'); hold on
scatter(Y(Labeled_cell,1),Y(Labeled_cell,2),20,'red','filled'); hold on
scatter(Y(Labeled_cell2,1),Y(Labeled_cell2,2),20,'red','filled'); hold on
scatter(Y(Labeled_cell3,1),Y(Labeled_cell3,2),20,'red','filled'); hold on
text(Y(Labeled_cell,1),Y(Labeled_cell,2),['Cell ',num2str(Labeled_cell),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','r')
text(Y(Labeled_cell2,1),Y(Labeled_cell2,2),[' \leftarrow ','Cell ',num2str(Labeled_cell2)],'HorizontalAlignment','left','FontWeight','bold','Color','r')
text(Y(Labeled_cell3,1),Y(Labeled_cell3,2),[' \leftarrow ','Cell ',num2str(Labeled_cell3)],'HorizontalAlignment','left','FontWeight','bold','Color','r')
ylim([-4,4])
median_scalebar(median_val)
xlabel('MDS axis 1')
ylabel('MDS axis 2')
title('Data distribution from Joint Projection')
%subplot(10,3,[2,5,8,11])
ax(2) = nexttile([2,1]);
scatter(Y(221:440,1),Y(221:440,2),20,'black','filled'); hold on
scatter(Y(Labeled_cell+220,1),Y(Labeled_cell+220,2),20,'red','filled'); hold on
scatter(Y(Labeled_cell2+220,1),Y(Labeled_cell2+220,2),20,'red','filled'); hold on
scatter(Y(Labeled_cell3+220,1),Y(Labeled_cell3+220,2),20,'red','filled'); hold on
text(Y(Labeled_cell+220,1),Y(Labeled_cell+220,2),['Cell ',num2str(Labeled_cell),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','r')
text(Y(Labeled_cell2+220,1),Y(Labeled_cell2+220,2),[' \leftarrow ','Cell ',num2str(Labeled_cell2)],'HorizontalAlignment','left','FontWeight','bold','Color','r')
text(Y(Labeled_cell3+220,1),Y(Labeled_cell3+220,2),[' \leftarrow ','Cell ',num2str(Labeled_cell3)],'HorizontalAlignment','left','FontWeight','bold','Color','r')
ylim([-4,4])
median_scalebar(median_val)
xlabel('MDS axis 1')
ylabel('MDS axis 2')
title('Model distribution from Joint Projection')

%Distances in MDS space histogram
%subplot(10,3,[3,6,9,12])
ax(3) = nexttile([2,1]);
histogram(distances,20,'FaceColor',[0,0,0],'FaceAlpha',0.95); hold on
h = plot([median_val,median_val],[0,60],'r--','LineWidth',2);
title(['MDS space distance distribution. Median : ',sprintf(' %.2f',median_val),''])
xlabel('MDS Space Euclidean Distance')

legend(h,'Median')

sgtitle(['MDS shapes -- Joint projection using [',Distance_measure , '] -- Best batch chosen by [',Choose_by,']'])


model_or_data = 'Data';
%subplot(10,3,[13,16])
ax(4) = nexttile;
Raster_matrix = squeeze(Data_Rasters(Labeled_cell,:,:));
cell_num = num2str(Labeled_cell);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
%subplot(10,3,[14,17])
ax(5) = nexttile;
Raster_matrix = squeeze(Data_Rasters(Labeled_cell2,:,:));
cell_num = num2str(Labeled_cell2);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
%subplot(10,3,[15,18])
ax(6) = nexttile;
Raster_matrix = squeeze(Data_Rasters(Labeled_cell3,:,:));
cell_num = num2str(Labeled_cell3);
raster_plot_func(Raster_matrix,cell_num,model_or_data)

model_or_data = 'Model';
%subplot(10,3,[19,22])
ax(7) = nexttile;
Raster_matrix = squeeze(sim_object.output(selection(Labeled_cell),:,Labeled_cell,:));
cell_num = num2str(Labeled_cell);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
%subplot(10,3,[20,23])
ax(8) = nexttile;
Raster_matrix = squeeze(sim_object.output(selection(Labeled_cell2),:,Labeled_cell2,:));
cell_num = num2str(Labeled_cell2);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
%subplot(10,3,[21,24])
ax(9) = nexttile;
Raster_matrix = squeeze(sim_object.output(selection(Labeled_cell3),:,Labeled_cell3,:));
cell_num = num2str(Labeled_cell3);
raster_plot_func(Raster_matrix,cell_num,model_or_data)

%subplot(10,3,[25,28])
ax(10) = nexttile;
plot(sim_PSTHs(Labeled_cell,:),'r','LineWidth',1); hold on
plot(data_PSTHs(Labeled_cell,:),'b','LineWidth',1);
%subplot(10,3,[26,29])
ax(11) = nexttile;
plot(sim_PSTHs(Labeled_cell2,:),'r','LineWidth',1); hold on
plot(data_PSTHs(Labeled_cell2,:),'b','LineWidth',1);
%subplot(10,3,[27,30])
ax(12) = nexttile;
plot(sim_PSTHs(Labeled_cell3,:),'r','LineWidth',1); hold on
plot(data_PSTHs(Labeled_cell3,:),'b','LineWidth',1);
legend({'model', 'data'},'Location','north')

function raster_plot_func(Raster_matrix,cell_num,model_or_data)
    
    ax = gca;
    ax.Position(2) = ax.Position(2) - 0.05;
    ax.Position(4) = ax.Position(4) - 0.05;
    [trial,sample] = find(Raster_matrix);
    time = sample/10000;

    line([time time]', ...
         [trial-0.5 trial+0.5]', ...
         'Color','k','LineWidth',0.6)

    xlim([0 2.9801])
    ylim([0.5 size(Raster_matrix,1)+0.5])
    %set(gca,'YDir','reverse','YTick',1:size(Raster_matrix,1))
    xlabel('Time (s)')
    %ylabel('Trial')
    title(['Cell ',cell_num,' — ',model_or_data])
    yticklabels('')
    box off
end

function dissimilarity_matrix = calc_dis(Distance_measure,data_object,sim_object,selection)
    

    if strcmp(Distance_measure,'RMSE')

        sim_PSTHs = [];
        data_PSTHs = [];

        for k = 1:220
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
        
        dissimilarity_matrix = zeros([440,440]);

        for k = 1:440
            for z  = 1:440
                dissimilarity_matrix(k,z) = sqrt(mean(((all_PSTH(k,:)-all_PSTH(z,:)).^2)));
            end
        end
    
    end

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

function median_scalebar(L)
    xl = xlim;
    yl = ylim;
    x0 = xl(1) + 0.05*range(xl);
    y0 = yl(1) + 0.05*range(yl);

    hold on
    plot([x0 x0+L],[y0 y0],'k-','LineWidth',3)
    text(x0+L/2,y0,sprintf(' %.2f',L), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom')
end