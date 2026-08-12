close all
clear all

load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\20_epoch_wide_eprop_cell_7_V2.mat") %Latest Forward

%% Construct the experimental PSTHs
data_object = load("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat");
sample_rate_hz = 10000;
n_batches = size(output,1);
n_trials = size(output,2);
n_cells = 220;
simlen = size(output,4);
PSTH_gran = 100; 
n_bins = (simlen-mod(simlen,PSTH_gran))/PSTH_gran;
Rasters_data = false(n_cells,n_trials,simlen);
PSTHs_data = zeros(n_cells,n_bins);
used_timesteps = n_bins*PSTH_gran;
bin_edges_seconds = (0:PSTH_gran:used_timesteps)/sample_rate_hz;

for k = 1:n_cells
    tuning = lower(char(string(data_object.all_data(k).tuning_type)));
    if contains(tuning,'contra')
        focus = 1;
    elseif contains(tuning,'45')
        focus = 2;
    elseif contains(tuning,'center')
        focus = 3;
    elseif contains(tuning,'ipsi')
        focus = 4;
    else
        error('Unrecognized tuning type for cell %d: %s',k,tuning);
    end

    SpikeTimes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
    picture = false(n_trials,simlen);
    for m = 1:n_trials
        % Match Data/data_handler.py by binning the original timestamps.
        % Rounding to the 0.1-ms raster first can move spikes across 10-ms bins.
        stim_mask = (SpikeTimes{m} >= 0) & ...
            (SpikeTimes{m} <= simlen/sample_rate_hz);
        cut_times = SpikeTimes{m}(stim_mask);
        PSTHs_data(k,:) = PSTHs_data(k,:) + ...
            histcounts(cut_times,bin_edges_seconds);

        trial_indices = floor(cut_times*sample_rate_hz);
        trial_indices = trial_indices( ...
            trial_indices >= 0 & trial_indices < simlen);
        picture(m,trial_indices+1) = true;
    end
    Rasters_data(k,:,:) = picture;
end

cell = 7;
n_cells = 1;
sim_PSTHs = squeeze(sum(sum(reshape( ...
    permute(output(:,:,:,1:used_timesteps),[3,1,2,4]), ...
    [n_cells,n_batches,n_trials,PSTH_gran,n_bins]),4),3));

%%



correlations = [];
for k = 1:n_batches
    %correlations = [correlations, max(xcorr(PSTHs_data(cell,:), sim_PSTHs(k,:),'normalized'))];
    %corr_val = corr(transpose([PSTHs_data(cell,:);sim_PSTHs(k,:)]));
    %correlations = [correlations, corr_val(1,2)];

    result = linCCC(PSTHs_data(cell,:),sim_PSTHs(k,:));
    ccc_val = result.ccc;
    correlations = [correlations, ccc_val];
end

[val,idx] = max(correlations);

figure;
subplot(2,1,1)
spy(squeeze(Rasters_data(cell,:,:)))
title('data')
subplot(2,1,2)
spy(squeeze(output(idx,:,:,:)))
title('sim')

[val idx] = sort(correlations,'descend');

n = 100; %How many plots
f = uifigure('Position',[100 100 800 800]);
g = uigridlayout(f,[n+1 1],'Scrollable','on');  %Creates a scrollable figure
g.RowHeight = repmat({100},1,n+1);

for k = 0:n
    ax = uiaxes(g);
    if k==0, A=squeeze(Rasters_data(cell,:,:)); title_text = 'data'; else, A=squeeze(output(idx(k),:,:,:)); title_text = sprintf('Batch %d | CCC = %.3f', idx(k), correlations(idx(k))); end
    [r,c] = find(A);
    plot(ax,c,r,'k.');
    ax.YDir = 'reverse';
    title(ax,title_text,'Interpreter','none');
end


%% Look at PCA projection space colored by correlation

params_names = fields(params);
cell = 7;
batch_size = size(output,1);
matrix_p = [];
for k = 1:12
    pram = params.(params_names{k});
    pram_single = squeeze(pram(:,:,size(params.(params_names{k}),3)));
    matrix_p = [matrix_p,pram_single];
end

Z = zscore(matrix_p,0,1);
[coeff, score, latent,~,explained] = pca(Z);

figure;
%scatter3(score(:,1),score(:,2),score(:,3),30,correlations,'filled')
scatter(score(:,1),score(:,2),30,correlations,'filled')
title('first two principle components')

figure;
heatmap(coeff)
colormap('parula')

%% Look at PLS

y = correlations(:);
[XL,YL,XS,YS,beta,pctvar,mse] = plsregress(Z,y,12,'CV',10);

figure;
scatter(XS(:,1),XS(:,2),30,y,'filled');
colorbar;
xlabel('PLS component 1');
ylabel('PLS component 2');
title('PLS')

%% MDS based on loss
PSTHs_all = [PSTHs_data(7,:);sim_PSTHs];
comp_matrix_all = zeros(1201, 1201);

for m = 1:1201
    for k = 1:1201
        %RMSE
        r = sqrt(mean((PSTHs_all(m,:) - PSTHs_all(k,:)).^2));
        comp_matrix_all(m,k) = r;
        %SSE
        %r = sum((PSTHs_all(m,:) - PSTHs_all(k,:)).^2);
        %comp_matrix_all(m,k) = r;
    end
end

[Y,stress] = mdscale(comp_matrix_all,2, 'criterion','metricsstress');
%%

mds_corr = [1,correlations]';
%mds_sse = [0;sse_losses_all];

figure;
scatter(Y(:,1),Y(:,2),10,mds_corr,'filled')

[xq,yq] = meshgrid(linspace(min(Y(:,1)),max(Y(:,1)),1000),linspace(min(Y(:,2)),max(Y(:,2)),1000));
F = scatteredInterpolant(Y(isfinite(mds_corr),1),Y(isfinite(mds_corr),2),mds_corr(isfinite(mds_corr)),'natural','none');
figure(Position=  [0,0,500,500]); contourf(xq,yq,F(xq,yq),30,'LineColor','none'); hold on; %scatter(Y(:,1),Y(:,2),12,mds_corr,'filled'); 
colorbar; axis equal tight; title('Interpolated MDS correlation'); axis square


%%
% Loss bar chart

%Calculate sse losses
r_vals = sum((PSTHs_data(7,:) - sim_PSTHs(:,:)).^2,2);

figure;
boxplot(r_vals)
ylim([0,12000])

%% Parameter plots

params_names = fields(params);
figure;
cell = 1; %This is cell 7 but we only ran cell 7 so it is the first one.
batch_size = size(output,1);
for k = 1:12
    pram = params_names{k};
    subplot(4,3,k)
    %for m = 1:batch_size
    %    plot(squeeze(params.(pram)(m,cell,:)),'Color',[0.5,0.5,0.5,0.5]); hold on
    %end
    plot(squeeze(params.(pram)(idx(1),cell,:)),'Color',[0.7,0.2,0.5,0.5],'LineWidth',2); hold on
    plot(squeeze(mean(params.(pram)(:,cell,:),1)),'LineWidth',3); hold on
    title(pram)
end
%%
%Looking at parameter change per epoch
figure;
for k = 1:12
    pram = params_names{k};
    subplot(4,3,k)
    %for m = 1:batch_size
    %    plot(diff(squeeze(params.(pram)(m,cell,:))).^2,'Color',[0.5,0.5,0.5,0.5]); hold on
    %end
    plot(diff(squeeze(params.(pram)(idx(1),cell,:))).^2,'Color',[0.7,0.2,0.5,0.5],'LineWidth',2); hold on
    plot(mean(diff(transpose(squeeze(params.(pram)(:,cell,:)))).^2,2),'LineWidth',3); hold on
    title(pram)
end




