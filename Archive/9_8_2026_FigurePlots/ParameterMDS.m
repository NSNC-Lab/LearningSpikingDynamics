%Bring stuff in
close all; clear all;
InitializecSPIKE;
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_all_cells_Eprop';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = load(sim_location);
Distance_measure = 'Z-Score-Euclidean';


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
diff_mat = calc_diff_mat(Distance_measure,sim_object,selection);

color_vec2 = [];
for k = 1:220
    values = sim_object.params.strf_alpha;
    color_vec2 = [color_vec2, squeeze(values(selection(k),k,end))];
end
color_vec3 = [];
for k = 1:220
    values = sim_object.params.on_ron_gsyn;
    color_vec3 = [color_vec3, squeeze(values(selection(k),k,end))];
end
color_vec4 = [];
for k = 1:220
    values = sim_object.params.off_ron_gsyn;
    color_vec4 = [color_vec4, squeeze(values(selection(k),k,end))];
end
%Total inhibitory
% color_vec5 = [];
% for k = 1:220
%     values = sim_object.params.sonoff_ron_gsyn.*(sim_object.params.on_sonoff_gsyn+sim_object.params.sonoff_ron_gsyn);
%     color_vec5 = [color_vec5, squeeze(values(selection(k),k,end))];
% end

%on sonoff
color_vec5 = [];
for k = 1:220
    values = sim_object.params.on_sonoff_gsyn;
    color_vec5 = [color_vec5, squeeze(values(selection(k),k,end))];
end

%off sonoff
color_vec6 = [];
for k = 1:220
    values = sim_object.params.off_sonoff_gsyn;
    color_vec6 = [color_vec6, squeeze(values(selection(k),k,end))];
end

%sonoff ron
color_vec7 = [];
for k = 1:220
    values = sim_object.params.sonoff_ron_gsyn;
    color_vec7 = [color_vec7, squeeze(values(selection(k),k,end))];
end

% Do MDS and plot
[Y,stress] = mdscale(diff_mat, 2, 'criterion','metricsstress');
%%
%Parameter MDS figure
close all;
figure(Position=[400,400,1000,650]);
subplot(2,3,1)
scatter(Y(:,1),Y(:,2),20,'black','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('MDS space shape')
%Color by layer
subplot(2,3,2)
scatter(Y(:,1),Y(:,2),20,color_vec,'filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('Colored by layer')
%Color by Strf Alpha
subplot(2,3,3)
scatter(Y(:,1),Y(:,2),20,color_vec2,'filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('Colored by STRF alpha')
subplot(2,3,4)
scatter(Y(:,1),Y(:,2),20,color_vec3,'filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('Colored by on -> ron gsyn')
subplot(2,3,5)
scatter(Y(:,1),Y(:,2),20,color_vec4,'filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('Colored by off -> ron gsyn')
subplot(2,3,6)
scatter(Y(:,1),Y(:,2),20,color_vec5,'filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('Colored by "Inhibitory Strength"')

sgtitle(['Parameter MDS using: ', Distance_measure])



%Figure 4

Labeled_cell = 210;
Labeled_cell2 = 186;
Labeled_cell3 = 92;
Labeled_cell4 = 102;
Labeled_cell5 = 87;

figure(Position=[100,100,1600,700]);
outer = tiledlayout(2,1, 'TileSpacing','compact','Padding','compact');

top = tiledlayout(outer,1,5,'TileSpacing','compact','Padding','compact');

top.Layout.Tile = 1;

ax(1) = nexttile(top);
scatter(Y(:,1),Y(:,2),20,color_vec3,'filled'); hold on
scatter(Y(Labeled_cell,1),Y(Labeled_cell,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{on \rightarrow C}')
v = text(Y(Labeled_cell,1),Y(Labeled_cell,2),['Cell ',num2str(Labeled_cell),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');
v.Position(2) = v.Position(2) + 0.25;
ax(2) = nexttile(top);
scatter(Y(:,1),Y(:,2),20,color_vec4,'filled'); hold on
scatter(Y(Labeled_cell2,1),Y(Labeled_cell2,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{off \rightarrow C}')
v = text(Y(Labeled_cell2,1),Y(Labeled_cell2,2),['Cell ',num2str(Labeled_cell2),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');
ax(3) = nexttile(top);
scatter(Y(:,1),Y(:,2),20,color_vec5,'filled'); hold on
scatter(Y(Labeled_cell3,1),Y(Labeled_cell3,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{on \rightarrow PV}')
v = text(Y(Labeled_cell3,1),Y(Labeled_cell3,2),['Cell ',num2str(Labeled_cell3),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');

%sgtitle(['Parameter MDS using: ', Distance_measure])
ax(4) = nexttile(top);
scatter(Y(:,1),Y(:,2),20,color_vec6,'filled'); hold on
scatter(Y(Labeled_cell4,1),Y(Labeled_cell4,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{off \rightarrow PV}')
v = text(Y(Labeled_cell4,1),Y(Labeled_cell4,2),['Cell ',num2str(Labeled_cell4),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');

%sgtitle(['Parameter MDS using: ', Distance_measure])
ax(5) = nexttile(top);
scatter(Y(:,1),Y(:,2),20,color_vec7,'filled'); hold on
scatter(Y(Labeled_cell5,1),Y(Labeled_cell5,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{PV \rightarrow C}')
v = text(Y(Labeled_cell5,1),Y(Labeled_cell5,2),['Cell ',num2str(Labeled_cell5),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');
colormap('parula')
%sgtitle(['Parameter MDS using: ', Distance_measure])

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

model_or_data = 'Data';

bottom = tiledlayout(outer,1,5,'TileSpacing','compact','Padding','compact');
bottom.Layout.Tile = 2;

col1 = tiledlayout(bottom,4,1,'TileSpacing','none','Padding','compact');
col1.Layout.Tile = 1;
col2 = tiledlayout(bottom,4,1,'TileSpacing','none','Padding','compact');
col2.Layout.Tile = 2;
col3 = tiledlayout(bottom,4,1,'TileSpacing','none','Padding','compact');
col3.Layout.Tile = 3;
col4 = tiledlayout(bottom,4,1,'TileSpacing','none','Padding','compact');
col4.Layout.Tile = 4;
col5 = tiledlayout(bottom,4,1,'TileSpacing','none','Padding','compact');
col5.Layout.Tile = 5;

ax(6) = nexttile(col1);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell,:,:));
cell_num = num2str(Labeled_cell);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
ylabel('Data')
title(['Cell: ', num2str(Labeled_cell)])
%subplot(10,3,[14,17])
ax(7) = nexttile(col2);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell2,:,:));
cell_num = num2str(Labeled_cell2);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
title(['Cell: ', num2str(Labeled_cell2)])
%subplot(10,3,[15,18])
ax(8) = nexttile(col3);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell3,:,:));
cell_num = num2str(Labeled_cell3);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
title(['Cell: ', num2str(Labeled_cell3)])
ax(9) = nexttile(col4);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell4,:,:));
cell_num = num2str(Labeled_cell4);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
title(['Cell: ', num2str(Labeled_cell4)])
ax(10) = nexttile(col5);
Raster_matrix = squeeze(Data_Rasters(Labeled_cell5,:,:));
cell_num = num2str(Labeled_cell5);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
title(['Cell: ', num2str(Labeled_cell5)])



model_or_data = 'Model';
%subplot(10,3,[19,22])
ax(11) = nexttile(col1);
Raster_matrix = squeeze(sim_object.output(selection(Labeled_cell),:,Labeled_cell,:));
cell_num = num2str(Labeled_cell);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
ylabel('Model')
%subplot(10,3,[20,23])
ax(12) = nexttile(col2);
Raster_matrix = squeeze(sim_object.output(selection(Labeled_cell2),:,Labeled_cell2,:));
cell_num = num2str(Labeled_cell2);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
%subplot(10,3,[21,24])
ax(13) = nexttile(col3);
Raster_matrix = squeeze(sim_object.output(selection(Labeled_cell3),:,Labeled_cell3,:));
cell_num = num2str(Labeled_cell3);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
ax(14) = nexttile(col4);
Raster_matrix = squeeze(sim_object.output(selection(Labeled_cell4),:,Labeled_cell4,:));
cell_num = num2str(Labeled_cell4);
raster_plot_func(Raster_matrix,cell_num,model_or_data)
ax(15) = nexttile(col5);
Raster_matrix = squeeze(sim_object.output(selection(Labeled_cell5),:,Labeled_cell5,:));
cell_num = num2str(Labeled_cell5);
raster_plot_func(Raster_matrix,cell_num,model_or_data)

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

%subplot(10,3,[25,28])
[y, Fs] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target1.wav');
y = y((0.25*Fs)-1:end);
time = (0:length(y)-1) / Fs;
time2 = linspace(0,max(y),length(sim_PSTHs(Labeled_cell,:)));

ax(16) = nexttile(col1);
plot(sim_PSTHs(Labeled_cell,:),'r','LineWidth',1); hold on
plot(data_PSTHs(Labeled_cell,:),'b','LineWidth',1);
xticks([])
yticklabels('')
legend({'model', 'data'},'Location','westoutside')
%subplot(10,3,[26,29])
ax(17) = nexttile(col2);
plot(sim_PSTHs(Labeled_cell2,:),'r','LineWidth',1); hold on
plot(data_PSTHs(Labeled_cell2,:),'b','LineWidth',1);
yticklabels('')
xticks([])
%subplot(10,3,[27,30])
ax(18) = nexttile(col3);
plot(sim_PSTHs(Labeled_cell3,:),'r','LineWidth',1); hold on
plot(data_PSTHs(Labeled_cell3,:),'b','LineWidth',1);
yticklabels('')
xticks([])
ax(19) = nexttile(col4);
plot(sim_PSTHs(Labeled_cell4,:),'r','LineWidth',1); hold on
plot(data_PSTHs(Labeled_cell4,:),'b','LineWidth',1);
yticklabels('')
xticks([])
ax(20) = nexttile(col5);
plot(sim_PSTHs(Labeled_cell5,:),'r','LineWidth',1); hold on
plot(data_PSTHs(Labeled_cell5,:),'b','LineWidth',1);
yticklabels('')
xticks([])


ax(21) = nexttile(col1);
plot(time,y,'k')
xlim([0, time(end)])
xlabel('Time (s)')
yticklabels('')
yticks([])
xticks(0:0.5:time(end))
ax(22) = nexttile(col2);
plot(time,y,'k')
xlim([0, time(end)])
xlabel('Time (s)')
yticklabels('')
yticks([])
ax(23) = nexttile(col3);
plot(time,y,'k')
xlim([0, time(end)])
xlabel('Time (s)')
yticklabels('')
yticks([])
ax(24) = nexttile(col4);
plot(time,y,'k')
xlim([0, time(end)])
xlabel('Time (s)')
yticklabels('')
yticks([])
ax(25) = nexttile(col5);
plot(time,y,'k')
xlim([0, time(end)])
xlabel('Time (s)')
yticklabels('')
yticks([])

eventTime = 1.25;  % seconds

columnAxes = [ax(6), ax(11), ax(16), ax(21)];

for a = columnAxes
    xline(a, eventTime, '--k', 'LineWidth', 1.5);
end

%%
color_vecs = [];
f_vals = fields(sim_object.params);
for m = 1:12
    sub_vec = [];
    for k = 1:220
        values = sim_object.params.(f_vals{m});
        sub_vec = [sub_vec, squeeze(values(selection(k),k,end))];
    end
    color_vecs = [color_vecs; sub_vec];
end



%All parameters Figure
figure(Position=[100,100,1300,900]);
t = tiledlayout(3,4,'TileSpacing','compact','Padding','compact');
for k = 1:12
    ax(k) = nexttile;
    scatter(Y(:,1),Y(:,2),20,color_vecs(k,:),'filled'); hold on
    xlabel('MDS-Axis 1')
    ylabel('MDS-Axis 2')
    title(strrep(f_vals{k},'_',' '))
    colorbar;
end
colormap('copper')
sgtitle(['Parameter MDS using: ', Distance_measure])

function diff_mat = calc_diff_mat(Distance_measure,sim_object,selection)

    if strcmp(Distance_measure, 'Z-Score-Euclidean')
        
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

        %Take the euclidean distance between all pairwise comparisons
        diff_mat = zeros([220,220]);
        for z = 1:220
            for x = 1:220
                %This is just basically the difference between two points
                diff_mat(z,x) = norm(Z(z,:)-Z(x,:));
            end
        end
    
    elseif strcmp(Distance_measure, 'Z-Score-Cosine')
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
        
        % diff_mat = zeros([220,220]);
        % for z = 1:220
        %     for x = 1:220
        %         %Compute Cosine Distance
        %         diff_mat(z,x) = pdist([Z(z,:);Z(x,:)],'cosine');
        %     end
        % end
        
        %When using the method above sometimes along the diagnol due to
        %numerical stability you get some really small numbers and that is
        %likely throwing things for a loop. Squareform forces the diagnol
        %to zero (which is what it should be)
        diff_mat = squareform(pdist(Z,'cosine'));


    
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


function raster_plot_func(Raster_matrix,cell_num,model_or_data)
    
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
