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
Distance_measure = 'Z-Score-Euclidean';



%%

% Do selection
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
    nbins = 10;

elseif strcmp(cell_choice, 'possibly_usable_cells')
    choice_cells = Possibly_usable_cells;
    nbins = 20;

elseif strcmp(cell_choice, 'all_cells')
    choice_cells = 1:220;
    nbins = 50;
end

color_vec = [];
for k = choice_cells
    if strcmp(data_object.all_data(k).layer, 'L2/3'); color_vec = [color_vec, 1]; end
    if strcmp(data_object.all_data(k).layer, 'L4'); color_vec = [color_vec, 2]; end
    if strcmp(data_object.all_data(k).layer, 'L5/6'); color_vec = [color_vec, 3]; end
    if strcmp(data_object.all_data(k).layer, 'NaN'); color_vec = [color_vec, 4]; end
end

diff_mat = calc_diff_mat(Distance_measure,sim_object,selection,Good_Cells,Possibly_usable_cells,cell_choice);

color_vec2 = [];
for k = choice_cells
    values = sim_object.params.strf_alpha;
    color_vec2 = [color_vec2, squeeze(values(selection(k),k,end))];
end
color_vec3 = [];
for k = choice_cells
    values = sim_object.params.on_ron_gsyn;
    color_vec3 = [color_vec3, squeeze(values(selection(k),k,end))];
end
color_vec4 = [];
for k = choice_cells
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
for k = choice_cells
    values = sim_object.params.on_sonoff_gsyn;
    color_vec5 = [color_vec5, squeeze(values(selection(k),k,end))];
end

%off sonoff
color_vec6 = [];
for k = choice_cells
    values = sim_object.params.off_sonoff_gsyn;
    color_vec6 = [color_vec6, squeeze(values(selection(k),k,end))];
end

%sonoff ron
color_vec7 = [];
for k = choice_cells
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

%Labeled_cell = 60;
%Labeled_cell2 = 33;
%Labeled_cell3 = 92;
%Labeled_cell4 = 102;
%Labeled_cell5 = 87;


Labeled_cell = 52;
Labeled_cell2 = 128;
Labeled_cell3 = 130;
Labeled_cell4 = 19;
Labeled_cell5 = 218;


ic1 = find(choice_cells == Labeled_cell);
ic2 = find(choice_cells == Labeled_cell2);
ic3 = find(choice_cells == Labeled_cell3);
ic4 = find(choice_cells == Labeled_cell4);
ic5 = find(choice_cells == Labeled_cell5);

figure(Position=[100,100,1600,700]);
outer = tiledlayout(2,1, 'TileSpacing','compact','Padding','compact');

top = tiledlayout(outer,1,5,'TileSpacing','compact','Padding','compact');

top.Layout.Tile = 1;

ax(1) = nexttile(top);
create_contour_pspace(Y(:,1),Y(:,2),color_vec3)
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{on \rightarrow C}')
scatter(Y(:,1),Y(:,2),20,color_vec3,'filled'); hold on
scatter(Y(ic1,1),Y(ic1,2),20,'red','filled'); hold on
v = text(Y(ic1,1),Y(ic1,2),['Cell ',num2str(Labeled_cell),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');
v.Position(2) = v.Position(2) + 0.25;
xlim([min(Y(:,1)), max(Y(:,1))])
ylim([min(Y(:,2)), max(Y(:,2))])
ax(2) = nexttile(top);
create_contour_pspace(Y(:,1),Y(:,2),color_vec4)
scatter(Y(:,1),Y(:,2),20,color_vec4,'filled'); hold on
scatter(Y(ic2,1),Y(ic2,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{off \rightarrow C}')
v = text(Y(ic2,1),Y(ic2,2),['Cell ',num2str(Labeled_cell2),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');
xlim([min(Y(:,1)), max(Y(:,1))])
ylim([min(Y(:,2)), max(Y(:,2))])
ax(3) = nexttile(top);
create_contour_pspace(Y(:,1),Y(:,2),color_vec5)
scatter(Y(:,1),Y(:,2),20,color_vec5,'filled'); hold on
scatter(Y(ic3,1),Y(ic3,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{on \rightarrow PV}')
v = text(Y(ic3,1),Y(ic3,2),['Cell ',num2str(Labeled_cell3),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');
xlim([min(Y(:,1)), max(Y(:,1))])
ylim([min(Y(:,2)), max(Y(:,2))])
%sgtitle(['Parameter MDS using: ', Distance_measure])
ax(4) = nexttile(top);
create_contour_pspace(Y(:,1),Y(:,2),color_vec6)
scatter(Y(:,1),Y(:,2),20,color_vec6,'filled'); hold on
scatter(Y(ic4,1),Y(ic4,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{off \rightarrow PV}')
v = text(Y(ic4,1),Y(ic4,2),['Cell ',num2str(Labeled_cell4),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');
xlim([min(Y(:,1)), max(Y(:,1))])
ylim([min(Y(:,2)), max(Y(:,2))])
%sgtitle(['Parameter MDS using: ', Distance_measure])
ax(5) = nexttile(top);
create_contour_pspace(Y(:,1),Y(:,2),color_vec7)
scatter(Y(:,1),Y(:,2),20,color_vec7,'filled'); hold on
scatter(Y(ic5,1),Y(ic5,2),20,'red','filled')
xlabel('MDS-Axis 1')
ylabel('MDS-Axis 2')
title('g_{PV \rightarrow C}')
v = text(Y(ic5,1),Y(ic5,2),['Cell ',num2str(Labeled_cell5),' \rightarrow'],'HorizontalAlignment','right','FontWeight','bold','Color','k');
colormap('parula')
%sgtitle(['Parameter MDS using: ', Distance_measure])
xlim([min(Y(:,1)), max(Y(:,1))])
ylim([min(Y(:,2)), max(Y(:,2))])

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
psth_bin_edges = (0:200:29800)/10000;  % 20 ms bins, in seconds
psth_time = psth_bin_edges(1:end-1);   % plot each bin at its left edge

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
    data_PSTHs = [data_PSTHs;movmean(histcounts(data_times,psth_bin_edges),3)];
    sim_PSTHs = [sim_PSTHs;movmean(histcounts(sim_times,psth_bin_edges),3)];
end

%subplot(10,3,[25,28])
[y, Fs] = audioread('C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Targets\200k_target1.wav');
y = y((0.25*Fs)-1:end);
time = (0:length(y)-1) / Fs;

ax(16) = nexttile(col1);
plot(psth_time,sim_PSTHs(Labeled_cell,:),'r','LineWidth',1); hold on
plot(psth_time,data_PSTHs(Labeled_cell,:),'b','LineWidth',1);
xlim([0 2.9801])
xticks([])
yticklabels('')
legend({'model', 'data'},'Location','westoutside','AutoUpdate','off')
%subplot(10,3,[26,29])
ax(17) = nexttile(col2);
plot(psth_time,sim_PSTHs(Labeled_cell2,:),'r','LineWidth',1); hold on
plot(psth_time,data_PSTHs(Labeled_cell2,:),'b','LineWidth',1);
xlim([0 2.9801])
yticklabels('')
xticks([])
%subplot(10,3,[27,30])
ax(18) = nexttile(col3);
plot(psth_time,sim_PSTHs(Labeled_cell3,:),'r','LineWidth',1); hold on
plot(psth_time,data_PSTHs(Labeled_cell3,:),'b','LineWidth',1);
xlim([0 2.9801])
yticklabels('')
xticks([])
ax(19) = nexttile(col4);
plot(psth_time,sim_PSTHs(Labeled_cell4,:),'r','LineWidth',1); hold on
plot(psth_time,data_PSTHs(Labeled_cell4,:),'b','LineWidth',1);
xlim([0 2.9801])
yticklabels('')
xticks([])
ax(20) = nexttile(col5);
plot(psth_time,sim_PSTHs(Labeled_cell5,:),'r','LineWidth',1); hold on
plot(psth_time,data_PSTHs(Labeled_cell5,:),'b','LineWidth',1);
xlim([0 2.9801])
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

for a = ax([6,11,16,21])
    xline(a, 0.4, 'LineWidth', 1,'Color',[0.5,0,0], ...
        'HandleVisibility','off');
    xline(a, 0.5, 'LineWidth', 1,'Color',[0.5,0,0], ...
        'HandleVisibility','off');
end

for a = ax([9,14,19,24])
    xline(a, 1, 'LineWidth', 1,'Color',[0.5,0,0], ...
        'HandleVisibility','off');
    xline(a, 1.2, 'LineWidth', 1,'Color',[0.5,0,0], ...
        'HandleVisibility','off');
end



%%
color_vecs = [];
f_vals = fields(sim_object.params);
for m = 1:12
    sub_vec = [];
    for k = choice_cells
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

function diff_mat = calc_diff_mat(Distance_measure,sim_object,selection,Good_Cells,Possibly_usable_cells,cell_choice)

    if strcmp(cell_choice, 'good_cells')
        choices = Good_Cells;
    elseif strcmp(cell_choice, 'possibly_usable_cells')
        choices = Possibly_usable_cells;
    elseif strcmp(cell_choice, 'all_cells')
        choices = 1:220;
    else
        disp('pick valid choice')
    end

    if strcmp(Distance_measure, 'Z-Score-Euclidean')
        
        %Create a matrix that we can actually use in order to do this
        %analysis. It should be 14 params by 220 cells
        
        holder_params = zeros([length(choices),12]);
        field_names = fields(sim_object.params);
        for k = 1:length(fields(sim_object.params))
            values = sim_object.params.(field_names{k});
            for m = 1:length(choices)
                holder_params(m,k) = squeeze(values(selection(m),m,end));
            end
        end
        
        %Z-score
        Z = zscore(holder_params,0,1);

        %Take the euclidean distance between all pairwise comparisons
        diff_mat = zeros([length(choices),length(choices)]);
        for z = 1:length(choices)
            for x = 1:length(choices)
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
    
    output_size = size(sim_object.output);
    n_batches = output_size(1);

    selection = [];
    if strcmp(choose_by, 'none')
        selection = repmat(1:12,220,1);
    elseif strcmp(choose_by, 'SSE')
        for k = 1:220
            if k == 210
                disp('')
            end
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
            NCCCs = [];
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

function create_contour_pspace(x,y,weight)

    edges = linspace(min([x;y]),max([x;y]),50);
    %counts = histcounts2(data_val,sim_val,edges,edges);
    grid = zeros([50,50]);

    for k = 1:length(weight)
        grid(round((x(k)-min(x)+1)*(50/(max(x)-min(x)+1))),round((y(k)-min(y)+1)*(50/(max(y)-min(y)+1)))) = weight(k);
    end

    %Greate guassian smoothing kernel
    %g_filt = [1,4,7,4,1;4,16,26,16,4;7,26,41,26,7;4,16,26,16,4;1,4,7,4,1];
    
    x = 0:0.1:10;
    mu = 5;
    sigma = 0.25;

    g = normpdf(x,mu,sigma);

    g_filt = g'*g;


    %2D convolve counts to create a bloom and smooth affect.
    counts = conv2(grid,g_filt,"same");

    centers = edges(1:end-1) + diff(edges)/2;

    hold on
    h = imagesc(centers,centers,counts');
    set(gca,'YDir','normal')

    h.AlphaData = 0.35*(counts' > 0);
    uistack(h,'bottom')
    
    colormap("parula")

end



