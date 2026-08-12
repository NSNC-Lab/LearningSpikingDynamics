%% Performance-directed analysis of the wide forward-sensitivity run
% This is intentionally separate from correlation_investigation_wide_run.m.
% It compares unsupervised PCA with supervised PLS, tests whether additional
% dimensions improve held-out prediction, tests a nonlinear tree ensemble,
% and describes which parameter directions contract among the top solutions.

close all
clearvars

rng(20260811,'twister')

analysis_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(analysis_dir));
if ~isfile(fullfile(project_root,'20_epoch_wide_forwards_sensitivity_cell_7.mat'))
    % This fallback lets the script be tested before it is copied to Archive.
    project_root = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics';
end

run_file = fullfile(project_root,'20_epoch_wide_forwards_sensitivity_cell_7.mat');
data_file = fullfile(project_root,'Data','Data', ...
    'all_units_info_with_polished_criteria_modified_perf.mat');
results_dir = fullfile(analysis_dir,'wide_run_performance_separation_results');

target_cell = 7;
sample_rate_hz = 10000;
psth_gran = 100;                    % 10-ms bins at 0.1-ms resolution
top_fraction = 0.10;
n_folds = 10;
n_trees = 200;
n_bootstrap = 250;
n_constraint_permutations = 2000;

if ~exist(results_dir,'dir')
    mkdir(results_dir)
end

%% Reconstruct the target and simulated PSTHs

run_object = load(run_file,'output','params');
data_object = load(data_file,'all_data');

n_batches = size(run_object.output,1);
n_trials = size(run_object.output,2);
simlen = size(run_object.output,4);
used_timesteps = simlen - mod(simlen,psth_gran);
n_bins = used_timesteps/psth_gran;
duration_seconds = used_timesteps/sample_rate_hz;

tuning = lower(char(string(data_object.all_data(target_cell).tuning_type)));
if contains(tuning,'contra')
    focus = 1;
elseif contains(tuning,'45')
    focus = 2;
elseif contains(tuning,'center')
    focus = 3;
elseif contains(tuning,'ipsi')
    focus = 4;
else
    error('Unrecognized tuning type for cell %d: %s',target_cell,tuning)
end

bin_edges_seconds = (0:psth_gran:used_timesteps)/sample_rate_hz;
target_trial_psth = zeros(n_trials,n_bins);
spike_times = data_object.all_data(target_cell).ctrl_tar1_timestamps(:,focus);
for trial = 1:n_trials
    times = spike_times{trial};
    times = times(times >= 0 & times <= simlen/sample_rate_hz);
    target_trial_psth(trial,:) = histcounts(times,bin_edges_seconds);
end
target_psth = sum(target_trial_psth,1);

reshaped_output = reshape( ...
    run_object.output(:,:,:,1:used_timesteps), ...
    [n_batches,n_trials,1,psth_gran,n_bins]);
sim_trial_psth = reshape(sum(reshaped_output,4), ...
    [n_batches,n_trials,n_bins]);
sim_psth = reshape(sum(sim_trial_psth,2),[n_batches,n_bins]);
clear reshaped_output

ccc = rowwise_ccc(sim_psth,target_psth);
pearson_r = rowwise_correlation(sim_psth,target_psth);
psth_sse = sum((double(sim_psth) - target_psth).^2,2);
target_fr = sum(target_psth)/(n_trials*duration_seconds);
sim_fr = sum(sim_psth,2)/(n_trials*duration_seconds);
relative_fr_error = abs(sim_fr-target_fr)/max(target_fr,eps);

%% Match the final output to the parameter vector that generated it

parameter_names = fieldnames(run_object.params);
n_parameters = numel(parameter_names);
X = zeros(n_batches,n_parameters);
for parameter_index = 1:n_parameters
    parameter_history = run_object.params.(parameter_names{parameter_index});
    if size(parameter_history,1) ~= n_batches || size(parameter_history,2) ~= 1
        error(['Expected one-cell parameter histories with size batch x 1 x epoch. ' ...
            '%s has size %s.'],parameter_names{parameter_index}, ...
            mat2str(size(parameter_history)))
    end
    X(:,parameter_index) = reshape( ...
        parameter_history(:,1,end),n_batches,1);
end

valid = all(isfinite(X),2) & isfinite(ccc) & isfinite(relative_fr_error);
X = X(valid,:);
ccc = ccc(valid);
pearson_r = pearson_r(valid);
psth_sse = psth_sse(valid);
sim_fr = sim_fr(valid);
sim_trial_psth = sim_trial_psth(valid,:,:);
n_observations = size(X,1);

[Z,parameter_mean,parameter_sd] = zscore(X,0,1);
parameter_sd(parameter_sd == 0) = 1;
[pca_coeff,pca_score,pca_latent,~,pca_explained] = pca(Z);
pca_cumulative = cumsum(pca_explained);

n_pls_display = min(3,n_parameters);
[~,~,pls_score,~,~,pls_pctvar] = plsregress(Z,ccc,n_pls_display);

%% Held-out test: do additional dimensions actually predict CCC?

cv_regression = cvpartition(n_observations,'KFold',n_folds);
pcr_prediction = nan(n_observations,n_parameters);
pls_prediction = nan(n_observations,n_parameters);
tree_prediction = nan(n_observations,1);

tree_template = templateTree( ...
    'MinLeafSize',4, ...
    'NumVariablesToSample',n_parameters);

for fold = 1:n_folds
    train_mask = training(cv_regression,fold);
    test_mask = test(cv_regression,fold);

    train_mean = mean(X(train_mask,:),1);
    train_sd = std(X(train_mask,:),0,1);
    train_sd(train_sd == 0) = 1;
    X_train = (X(train_mask,:) - train_mean)./train_sd;
    X_test = (X(test_mask,:) - train_mean)./train_sd;
    y_train = ccc(train_mask);

    [fold_coeff,fold_score,~,~,~,fold_pca_mean] = pca(X_train);
    fold_test_score = (X_test-fold_pca_mean)*fold_coeff;

    for component_count = 1:n_parameters
        pcr_beta = [ones(sum(train_mask),1), ...
            fold_score(:,1:component_count)]\y_train;
        pcr_prediction(test_mask,component_count) = ...
            [ones(sum(test_mask),1), ...
            fold_test_score(:,1:component_count)]*pcr_beta;

        [~,~,~,~,pls_beta] = plsregress( ...
            X_train,y_train,component_count);
        pls_prediction(test_mask,component_count) = ...
            [ones(sum(test_mask),1),X_test]*pls_beta;
    end

    tree_model = fitrensemble( ...
        X(train_mask,:),y_train, ...
        'Method','Bag', ...
        'NumLearningCycles',n_trees, ...
        'Learners',tree_template);
    tree_prediction(test_mask) = predict(tree_model,X(test_mask,:));
end

pcr_r2 = zeros(n_parameters,1);
pls_r2 = zeros(n_parameters,1);
pcr_spearman = zeros(n_parameters,1);
pls_spearman = zeros(n_parameters,1);
for component_count = 1:n_parameters
    pcr_r2(component_count) = heldout_r2(ccc, ...
        pcr_prediction(:,component_count));
    pls_r2(component_count) = heldout_r2(ccc, ...
        pls_prediction(:,component_count));
    pcr_spearman(component_count) = corr(ccc, ...
        pcr_prediction(:,component_count),'Type','Spearman');
    pls_spearman(component_count) = corr(ccc, ...
        pls_prediction(:,component_count),'Type','Spearman');
end
tree_r2 = heldout_r2(ccc,tree_prediction);
tree_spearman = corr(ccc,tree_prediction,'Type','Spearman');
linear_r2 = pcr_r2(end);
linear_spearman = pcr_spearman(end);

%% Can the top 10% be separated from the rest on held-out batches?

ccc_cutoff = prctile(ccc,100*(1-top_fraction));
high_performer = ccc >= ccc_cutoff;
cv_classification = cvpartition(high_performer,'KFold',n_folds);
logistic_score = nan(n_observations,1);
tree_classification_score = nan(n_observations,1);

for fold = 1:n_folds
    train_mask = training(cv_classification,fold);
    test_mask = test(cv_classification,fold);

    train_mean = mean(X(train_mask,:),1);
    train_sd = std(X(train_mask,:),0,1);
    train_sd(train_sd == 0) = 1;
    X_train = (X(train_mask,:) - train_mean)./train_sd;
    X_test = (X(test_mask,:) - train_mean)./train_sd;

    logistic_model = fitclinear( ...
        X_train,high_performer(train_mask), ...
        'Learner','logistic', ...
        'Regularization','ridge', ...
        'Lambda',1e-3, ...
        'Solver','lbfgs', ...
        'ClassNames',[false true]);
    [~,fold_score] = predict(logistic_model,X_test);
    positive_column = find(logistic_model.ClassNames == true,1);
    logistic_score(test_mask) = fold_score(:,positive_column);

    classification_model = fitcensemble( ...
        X(train_mask,:),high_performer(train_mask), ...
        'Method','Bag', ...
        'NumLearningCycles',n_trees, ...
        'Learners',tree_template, ...
        'Prior','uniform', ...
        'ClassNames',[false true]);
    [~,fold_score] = predict(classification_model,X(test_mask,:));
    positive_column = find(classification_model.ClassNames == true,1);
    tree_classification_score(test_mask) = fold_score(:,positive_column);
end

[~,~,~,logistic_auc] = perfcurve( ...
    high_performer,logistic_score,true);
[~,~,~,tree_auc] = perfcurve( ...
    high_performer,tree_classification_score,true);

%% Interpret the high-performing region

parameter_spearman = zeros(n_parameters,1);
for parameter_index = 1:n_parameters
    parameter_spearman(parameter_index) = corr( ...
        X(:,parameter_index),ccc,'Type','Spearman');
end

top_mean_shift = mean(Z(high_performer,:),1)';
top_std_ratio = (std(Z(high_performer,:),0,1)./std(Z,0,1))';

full_covariance = cov(Z);
top_covariance = cov(Z(high_performer,:));
covariance_ridge = 1e-6*trace(full_covariance)/n_parameters;
[constraint_vectors,constraint_values] = eig( ...
    top_covariance + covariance_ridge*eye(n_parameters), ...
    full_covariance + covariance_ridge*eye(n_parameters));
[constraint_values,order] = sort(real(diag(constraint_values)),'ascend');
constraint_vectors = real(constraint_vectors(:,order));
constraint_vectors = constraint_vectors./vecnorm(constraint_vectors,2,1);

full_tree_model = fitrensemble( ...
    X,ccc, ...
    'Method','Bag', ...
    'NumLearningCycles',2*n_trees, ...
    'Learners',tree_template);
tree_importance = predictorImportance(full_tree_model)';
tree_importance = tree_importance/sum(tree_importance);

% Identify the strongest pairwise compensations within the top solutions.
top_parameter_correlation = corr(Z(high_performer,:));
upper_triangle = triu(true(n_parameters),1);
[pair_row,pair_column] = find(upper_triangle);
pair_correlation = top_parameter_correlation(upper_triangle);
[~,pair_order] = sort(abs(pair_correlation),'descend');
pair_row = pair_row(pair_order);
pair_column = pair_column(pair_order);
pair_correlation = pair_correlation(pair_order);

% Test whether the top set is more compact than equally sized random subsets.
observed_covariance = top_covariance + covariance_ridge*eye(n_parameters);
observed_trace = trace(observed_covariance);
observed_logdet = stable_logdet(observed_covariance);
null_trace = zeros(n_constraint_permutations,1);
null_logdet = zeros(n_constraint_permutations,1);
n_top = sum(high_performer);
for permutation = 1:n_constraint_permutations
    random_subset = randperm(n_observations,n_top);
    random_covariance = cov(Z(random_subset,:)) + ...
        covariance_ridge*eye(n_parameters);
    null_trace(permutation) = trace(random_covariance);
    null_logdet(permutation) = stable_logdet(random_covariance);
end
trace_p_value = (1+sum(null_trace <= observed_trace))/ ...
    (n_constraint_permutations+1);
logdet_p_value = (1+sum(null_logdet <= observed_logdet))/ ...
    (n_constraint_permutations+1);

% Estimate how much ten-trial sampling noise changes the ranking. Experimental
% and simulated trials are independently resampled because they are not paired.
bootstrap_ccc = zeros(n_observations,n_bootstrap);
for bootstrap_index = 1:n_bootstrap
    data_draw = randi(n_trials,[n_trials,1]);
    simulation_draw = randi(n_trials,[n_trials,1]);
    bootstrap_target = sum(target_trial_psth(data_draw,:),1);
    bootstrap_simulation = reshape( ...
        sum(sim_trial_psth(:,simulation_draw,:),2), ...
        [n_observations,n_bins]);
    bootstrap_ccc(:,bootstrap_index) = rowwise_ccc( ...
        bootstrap_simulation,bootstrap_target);
end
bootstrap_rank_spearman = corr(ccc,bootstrap_ccc,'Type','Spearman')';
bootstrap_cutoff = prctile(bootstrap_ccc,100*(1-top_fraction),1);
bootstrap_high = bootstrap_ccc >= bootstrap_cutoff;
bootstrap_top_retention = sum(bootstrap_high & high_performer,1)'/n_top;
[~,bootstrap_best_batch] = max(bootstrap_ccc,[],1);
[~,observed_best_batch] = max(ccc);
observed_best_probability = mean(bootstrap_best_batch == observed_best_batch);
best_batch_probability = accumarray(bootstrap_best_batch',1, ...
    [n_observations,1])/n_bootstrap;
bootstrap_ccc_mean = mean(bootstrap_ccc,2);
bootstrap_ccc_sd = std(bootstrap_ccc,0,2);

%% Save quantitative tables

dimension_table = table( ...
    (1:n_parameters)',pca_explained,pca_cumulative, ...
    pcr_r2,pls_r2,pcr_spearman,pls_spearman, ...
    'VariableNames',{'Components','PCAExplainedPercent', ...
    'PCACumulativePercent','PCR_Heldout_R2','PLS_Heldout_R2', ...
    'PCR_Heldout_Spearman','PLS_Heldout_Spearman'});
writetable(dimension_table,fullfile(results_dir,'dimension_cross_validation.csv'))

parameter_table = table( ...
    string(parameter_names),parameter_spearman,top_mean_shift, ...
    top_std_ratio,tree_importance, ...
    'VariableNames',{'Parameter','SpearmanWithCCC', ...
    'TopDecileMeanShiftZ','TopDecileStdRatio','TreeImportance'});
writetable(parameter_table,fullfile(results_dir,'parameter_summary.csv'))

constraint_table = array2table( ...
    constraint_vectors(:,1:min(3,n_parameters)), ...
    'VariableNames',compose('Constraint%d',1:min(3,n_parameters)), ...
    'RowNames',parameter_names);
writetable(constraint_table, ...
    fullfile(results_dir,'tightest_constraint_loadings.csv'), ...
    'WriteRowNames',true)

pairwise_table = table( ...
    string(parameter_names(pair_row)), ...
    string(parameter_names(pair_column)), ...
    pair_correlation, ...
    'VariableNames',{'Parameter1','Parameter2','TopDecileCorrelation'});
writetable(pairwise_table,fullfile(results_dir, ...
    'top_decile_pairwise_correlations.csv'))

bootstrap_table = table( ...
    (1:n_observations)',ccc,bootstrap_ccc_mean,bootstrap_ccc_sd, ...
    best_batch_probability, ...
    'VariableNames',{'Batch','ObservedCCC','BootstrapMeanCCC', ...
    'BootstrapSD','ProbabilityOfBeingBest'});
writetable(bootstrap_table,fullfile(results_dir, ...
    'bootstrap_batch_stability.csv'))

%% Figure 1: PCA versus performance-directed PLS

performance_map = custom_performance_colormap(256);
figure_1 = figure('Color','w','Position',[100 100 1250 900]);
layout = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

axis_pca = nexttile;
scatter(pca_score(:,1),pca_score(:,2),25,ccc,'filled', ...
    'MarkerFaceAlpha',0.65)
hold on
scatter(pca_score(high_performer,1),pca_score(high_performer,2), ...
    38,ccc(high_performer),'filled','MarkerEdgeColor','k')
xlabel(sprintf('PC1 (%.1f%%)',pca_explained(1)))
ylabel(sprintf('PC2 (%.1f%%)',pca_explained(2)))
title('Unsupervised PCA')
grid on
clim(axis_pca,[min(ccc),max(ccc)])
pca_bar = colorbar(axis_pca);
pca_bar.Label.String = 'CCC';

axis_pls = nexttile;
scatter(pls_score(:,1),pls_score(:,2),25,ccc,'filled', ...
    'MarkerFaceAlpha',0.65)
hold on
scatter(pls_score(high_performer,1),pls_score(high_performer,2), ...
    38,ccc(high_performer),'filled','MarkerEdgeColor','k')
xlabel('PLS component 1')
ylabel('PLS component 2')
title('Performance-directed PLS')
grid on
clim(axis_pls,[min(ccc),max(ccc)])
pls_bar = colorbar(axis_pls);
pls_bar.Label.String = 'CCC';

nexttile
plot(1:n_parameters,pca_cumulative,'o-','LineWidth',2)
yline(80,'--','80%')
xlabel('Number of PCs')
ylabel('Cumulative parameter variance (%)')
ylim([0 102])
grid on

axis_rate = nexttile;
scatter(pearson_r,ccc,25,relative_fr_error,'filled', ...
    'MarkerFaceAlpha',0.65)
xlabel('Pearson correlation')
ylabel('CCC')
title('Rate mismatch changes the score')
grid on
rate_bar = colorbar;
rate_bar.Label.String = 'Relative firing-rate error';

colormap(figure_1,performance_map)
title(layout,sprintf('Cell %d wide run: parameter geometry and CCC',target_cell))
exportgraphics(figure_1,fullfile(results_dir,'01_pca_vs_pls.pdf'), ...
    'ContentType','vector')

%% Figure 2: held-out evidence for dimension and nonlinearity

figure_2 = figure('Color','w','Position',[120 120 1250 850]);
layout = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile
plot(1:n_parameters,pcr_r2,'o-','LineWidth',2)
hold on
plot(1:n_parameters,pls_r2,'s-','LineWidth',2)
yline(tree_r2,'--','Tree ensemble','LineWidth',1.5)
yline(0,':')
xlabel('Number of components')
ylabel('Held-out R^2')
legend('PCR','PLS','Location','best')
grid on

nexttile
plot(1:n_parameters,pcr_spearman,'o-','LineWidth',2)
hold on
plot(1:n_parameters,pls_spearman,'s-','LineWidth',2)
yline(tree_spearman,'--','Tree ensemble','LineWidth',1.5)
xlabel('Number of components')
ylabel('Held-out Spearman correlation')
legend('PCR','PLS','Location','best')
grid on

nexttile
bar(categorical({'Linear, all 12','Tree ensemble'}), ...
    [linear_r2,tree_r2])
yline(0,':')
ylabel('Held-out R^2 for CCC')
title('Continuous performance prediction')
grid on

nexttile
bar(categorical({'Logistic','Tree ensemble'}), ...
    [logistic_auc,tree_auc])
yline(0.5,':','Chance')
ylim([0.45 1])
ylabel('Held-out ROC AUC')
title(sprintf('Top %.0f%% classification',100*top_fraction))
grid on

title(layout,'Cross-validated separation of high-performing solutions')
exportgraphics(figure_2,fullfile(results_dir, ...
    '02_cross_validated_separation.pdf'),'ContentType','vector')

%% Figure 3: which individual parameters are constrained?

display_names = strrep(parameter_names,'_','\_');
figure_3 = figure('Color','w','Position',[140 100 1200 1000]);
layout = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile
barh(parameter_spearman)
yticks(1:n_parameters)
yticklabels(display_names)
xline(0,':')
xlabel('Spearman correlation with CCC')
title('Marginal association')
grid on

nexttile
barh(top_mean_shift)
yticks(1:n_parameters)
yticklabels(display_names)
xline(0,':')
xlabel('Mean shift in full-set SD units')
title('Top-decile location')
grid on

nexttile
barh(top_std_ratio)
yticks(1:n_parameters)
yticklabels(display_names)
xline(1,'--')
xlim([0,max(1.15,1.05*max(top_std_ratio))])
xlabel('Top-decile SD / full-set SD')
title('Top-decile contraction')
grid on

title(layout,sprintf('Top solutions: CCC >= %.3f',ccc_cutoff))
exportgraphics(figure_3,fullfile(results_dir, ...
    '03_parameter_constraints.pdf'),'ContentType','vector')

%% Figure 4: multivariate constraints and nonlinear importance

figure_4 = figure('Color','w','Position',[160 100 1250 850]);
layout = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile([2 1])
imagesc(constraint_vectors(:,1:min(3,n_parameters)))
colorbar
clim([-max(abs(constraint_vectors(:,1:3)),[],'all'), ...
    max(abs(constraint_vectors(:,1:3)),[],'all')])
yticks(1:n_parameters)
yticklabels(display_names)
xticks(1:min(3,n_parameters))
xticklabels(compose('Constraint %d',1:min(3,n_parameters)))
title('Loadings of tightest multivariate directions')

nexttile
semilogy(1:n_parameters,constraint_values,'o-','LineWidth',2)
yline(1,'--','Full-cloud variance')
xlabel('Constraint direction (tightest to broadest)')
ylabel('Top/full variance ratio')
grid on

nexttile
barh(tree_importance)
yticks(1:n_parameters)
yticklabels(display_names)
xlabel('Normalized tree importance')
title('Nonlinear performance importance')
grid on

title(layout,'Structure beyond two-dimensional PCA')
exportgraphics(figure_4,fullfile(results_dir, ...
    '04_multivariate_constraints.pdf'),'ContentType','vector')

%% Figure 5: marginal performance curves (descriptive, not causal)

figure_5 = figure('Color','w','Position',[80 60 1450 1050]);
layout = tiledlayout(4,3,'TileSpacing','compact','Padding','compact');
n_quantile_bins = 10;
for parameter_index = 1:n_parameters
    nexttile
    [bin_center,bin_mean,bin_sem] = quantile_summary( ...
        X(:,parameter_index),ccc,n_quantile_bins);
    errorbar(bin_center,bin_mean,bin_sem,'o-','LineWidth',1.5, ...
        'MarkerFaceColor',[0.1 0.55 0.2])
    xlabel(display_names{parameter_index})
    ylabel('Mean CCC')
    grid on
end
title(layout,'Marginal CCC versus each parameter (quantile bins)')
exportgraphics(figure_5,fullfile(results_dir, ...
    '05_marginal_performance_curves.pdf'),'ContentType','vector')

%% Figure 6: strongest compensatory relationships in the top decile

figure_6 = figure('Color','w','Position',[100 80 1300 900]);
layout = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile
n_pairs_to_show = min(10,numel(pair_correlation));
barh(pair_correlation(n_pairs_to_show:-1:1))
yticks(1:n_pairs_to_show)
pair_labels = strings(n_pairs_to_show,1);
for pair_index = 1:n_pairs_to_show
    source_index = n_pairs_to_show-pair_index+1;
    pair_labels(pair_index) = string(parameter_names{pair_row(source_index)}) + ...
        " vs " + string(parameter_names{pair_column(source_index)});
end
yticklabels(strrep(pair_labels,'_','\_'))
xline(0,':')
xlabel('Correlation within top decile')
title('Strongest pairwise relationships')
grid on

for pair_rank = 1:min(3,numel(pair_correlation))
    nexttile
    first_parameter = pair_row(pair_rank);
    second_parameter = pair_column(pair_rank);
    scatter(Z(:,first_parameter),Z(:,second_parameter),12, ...
        [0.75 0.75 0.75],'filled','MarkerFaceAlpha',0.25)
    hold on
    scatter(Z(high_performer,first_parameter), ...
        Z(high_performer,second_parameter),28,[0.05 0.60 0.20], ...
        'filled','MarkerEdgeColor','k','MarkerFaceAlpha',0.75)
    fit_coefficients = polyfit( ...
        Z(high_performer,first_parameter), ...
        Z(high_performer,second_parameter),1);
    fit_x = xlim;
    plot(fit_x,polyval(fit_coefficients,fit_x),'k--','LineWidth',1.5)
    xlabel(display_names{first_parameter})
    ylabel(display_names{second_parameter})
    title(sprintf('Top-decile r = %.3f',pair_correlation(pair_rank)))
    grid on
end

title(layout,'Compensatory parameter structure among high performers')
exportgraphics(figure_6,fullfile(results_dir, ...
    '06_top_decile_parameter_compensation.pdf'),'ContentType','vector')

%% Figure 7: ten-trial bootstrap stability

figure_7 = figure('Color','w','Position',[120 90 1250 850]);
layout = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile
histogram(bootstrap_rank_spearman,20)
xlabel('Spearman(original ranking, bootstrap ranking)')
ylabel('Bootstrap replicates')
title(sprintf('Median = %.3f',median(bootstrap_rank_spearman)))
grid on

nexttile
histogram(100*bootstrap_top_retention,20)
xlabel('Original top decile retained (%)')
ylabel('Bootstrap replicates')
title(sprintf('Median = %.1f%%',100*median(bootstrap_top_retention)))
grid on

nexttile
scatter(ccc,bootstrap_ccc_mean,25,bootstrap_ccc_sd,'filled', ...
    'MarkerFaceAlpha',0.7)
hold on
limits = [min([ccc;bootstrap_ccc_mean]),max([ccc;bootstrap_ccc_mean])];
plot(limits,limits,'k--')
xlim(limits)
ylim(limits)
xlabel('Observed CCC')
ylabel('Mean bootstrap CCC')
title('Performance estimate stability')
stability_bar = colorbar;
stability_bar.Label.String = 'Bootstrap CCC SD';
grid on

nexttile
[sorted_best_probability,best_probability_order] = sort( ...
    best_batch_probability,'descend');
n_best_to_show = min(10,n_observations);
bar(sorted_best_probability(1:n_best_to_show))
xticks(1:n_best_to_show)
xticklabels(string(best_probability_order(1:n_best_to_show)))
xlabel('Batch')
ylabel('Probability of being bootstrap best')
title(sprintf('Observed best batch %d remains best %.1f%%', ...
    observed_best_batch,100*observed_best_probability))
grid on

title(layout,'Uncertainty from ten stochastic trials')
exportgraphics(figure_7,fullfile(results_dir, ...
    '07_trial_bootstrap_stability.pdf'),'ContentType','vector')

%% Save reusable results and a concise text report

save(fullfile(results_dir,'wide_run_performance_separation_results.mat'), ...
    'X','Z','parameter_names','ccc','pearson_r','psth_sse','sim_fr', ...
    'target_fr','relative_fr_error','high_performer','ccc_cutoff', ...
    'pca_coeff','pca_score','pca_latent','pca_explained', ...
    'pls_score','pls_pctvar','pcr_prediction','pls_prediction', ...
    'tree_prediction','pcr_r2','pls_r2','tree_r2', ...
    'pcr_spearman','pls_spearman','tree_spearman', ...
    'logistic_auc','tree_auc','parameter_spearman','top_mean_shift', ...
    'top_std_ratio','constraint_vectors','constraint_values', ...
    'tree_importance','parameter_mean','parameter_sd', ...
    'pair_row','pair_column','pair_correlation', ...
    'observed_trace','observed_logdet','null_trace','null_logdet', ...
    'trace_p_value','logdet_p_value','bootstrap_ccc_mean', ...
    'bootstrap_ccc_sd','bootstrap_rank_spearman', ...
    'bootstrap_top_retention','best_batch_probability', ...
    'observed_best_batch','observed_best_probability')

[best_pcr_r2,best_pcr_components] = max(pcr_r2);
[best_pls_r2,best_pls_components] = max(pls_r2);
[sorted_contraction,contraction_order] = sort(top_std_ratio,'ascend');
[sorted_importance,importance_order] = sort(tree_importance,'descend');

report_file = fullfile(results_dir,'summary.txt');
report_id = fopen(report_file,'w');
cleanup = onCleanup(@() fclose(report_id));
fprintf(report_id,'Wide-run performance-separation analysis\n');
fprintf(report_id,'Run: %s\n',run_file);
fprintf(report_id,'Cell: %d | batches: %d | trials: %d\n\n', ...
    target_cell,n_observations,n_trials);
fprintf(report_id,'CCC median: %.4f | maximum: %.4f\n',median(ccc),max(ccc));
fprintf(report_id,'Top-decile cutoff: %.4f\n',ccc_cutoff);
fprintf(report_id,'Target firing rate: %.3f Hz\n\n',target_fr);
fprintf(report_id,'PC1+PC2 parameter variance: %.2f%%\n',pca_cumulative(2));
fprintf(report_id,'Best PCR: %d components, held-out R^2 = %.4f\n', ...
    best_pcr_components,best_pcr_r2);
fprintf(report_id,'Best PLS: %d components, held-out R^2 = %.4f\n', ...
    best_pls_components,best_pls_r2);
fprintf(report_id,'Tree ensemble: held-out R^2 = %.4f, Spearman = %.4f\n', ...
    tree_r2,tree_spearman);
fprintf(report_id,'Top-decile AUC: logistic = %.4f, tree = %.4f\n\n', ...
    logistic_auc,tree_auc);
fprintf(report_id,['Top-set compactness permutation tests: trace p = %.6f, ' ...
    'log-determinant p = %.6f\n'],trace_p_value,logdet_p_value);
fprintf(report_id,['Trial bootstrap: median rank Spearman = %.4f, ' ...
    'median top-decile retention = %.1f%%\n'], ...
    median(bootstrap_rank_spearman),100*median(bootstrap_top_retention));
fprintf(report_id,'Observed best batch %d remains best in %.1f%% of bootstraps\n\n', ...
    observed_best_batch,100*observed_best_probability);
fprintf(report_id,'Most contracted individual parameters:\n');
for rank = 1:min(5,n_parameters)
    index = contraction_order(rank);
    fprintf(report_id,'  %d. %s: SD ratio %.4f\n',rank, ...
        parameter_names{index},sorted_contraction(rank));
end
fprintf(report_id,'\nStrongest top-decile parameter relationships:\n');
for rank = 1:min(5,numel(pair_correlation))
    fprintf(report_id,'  %d. %s vs %s: r = %.4f\n',rank, ...
        parameter_names{pair_row(rank)},parameter_names{pair_column(rank)}, ...
        pair_correlation(rank));
end
fprintf(report_id,'\nMost important nonlinear predictors:\n');
for rank = 1:min(5,n_parameters)
    index = importance_order(rank);
    fprintf(report_id,'  %d. %s: importance %.4f\n',rank, ...
        parameter_names{index},sorted_importance(rank));
end

fprintf('\nAnalysis complete. Results saved to:\n%s\n',results_dir)

%% Local functions

function ccc = rowwise_ccc(simulated,target)
    simulated = double(simulated);
    target = double(target(:)');
    simulated_mean = mean(simulated,2);
    target_mean = mean(target);
    simulated_centered = simulated-simulated_mean;
    target_centered = target-target_mean;
    covariance = mean(simulated_centered.*target_centered,2);
    denominator = var(simulated,1,2) + var(target,1,2) + ...
        (simulated_mean-target_mean).^2;
    ccc = 2*covariance./max(denominator,eps);
end

function r = rowwise_correlation(simulated,target)
    simulated = double(simulated);
    target = double(target(:)');
    simulated_centered = simulated-mean(simulated,2);
    target_centered = target-mean(target);
    numerator = sum(simulated_centered.*target_centered,2);
    denominator = sqrt(sum(simulated_centered.^2,2) .* ...
        sum(target_centered.^2));
    r = numerator./max(denominator,eps);
end

function r2 = heldout_r2(observed,predicted)
    r2 = 1-sum((observed-predicted).^2)/sum((observed-mean(observed)).^2);
end

function map = custom_performance_colormap(n_colors)
    anchors = [0.20 0.15 0.45; ...
               0.20 0.55 0.85; ...
               0.95 0.80 0.15; ...
               0.05 0.65 0.20];
    anchor_locations = linspace(0,1,size(anchors,1));
    map = interp1(anchor_locations,anchors,linspace(0,1,n_colors),'linear');
end

function value = stable_logdet(matrix)
    eigenvalues = eig((matrix+matrix')/2);
    value = sum(log(max(real(eigenvalues),eps)));
end

function [centers,means,sems] = quantile_summary(x,y,n_bins)
    edges = prctile(x,linspace(0,100,n_bins+1));
    edges = unique(edges);
    if numel(edges) < 3
        centers = mean(x);
        means = mean(y);
        sems = std(y)/sqrt(numel(y));
        return
    end
    edges(1) = -inf;
    edges(end) = inf;
    groups = discretize(x,edges);
    n_actual_bins = numel(edges)-1;
    centers = nan(n_actual_bins,1);
    means = nan(n_actual_bins,1);
    sems = nan(n_actual_bins,1);
    for bin = 1:n_actual_bins
        mask = groups == bin;
        centers(bin) = median(x(mask));
        means(bin) = mean(y(mask));
        sems(bin) = std(y(mask))/sqrt(sum(mask));
    end
end
