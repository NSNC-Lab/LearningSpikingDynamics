%% Parameter stability analysis for the wide forward-sensitivity run
% Parameter slice e is the state that generated epoch e. Therefore,
% diff(params,1,epoch) measures the optimizer update between saved epochs.
% The post-update state after the final epoch is not present in params.

close all; clearvars; clc;

analysis_dir = fileparts(mfilename('fullpath'));
if isempty(analysis_dir), analysis_dir = pwd; end
project_root = fileparts(fileparts(analysis_dir));
run_file = fullfile(project_root,'20_epoch_wide_forwards_sensitivity_cell_7.mat');
results_dir = fullfile(analysis_dir,'wide_run_parameter_stability_results');
if ~exist(results_dir,'dir'), mkdir(results_dir); end

S = load(run_file,'params','sse_losses_all','cv_losses_all');
preferred_names = {'strf_gain','strf_alpha','output_ad','abs_ref','rel_ref_a', ...
    'rel_ref_b','rel_ref_c','on_ron_gsyn','off_ron_gsyn','sonoff_ron_gsyn', ...
    'on_sonoff_gsyn','off_sonoff_gsyn'};
available = fieldnames(S.params)';
parameter_names = [preferred_names(ismember(preferred_names,available)), ...
    setdiff(available,preferred_names,'stable')];

first_parameter = S.params.(parameter_names{1});
if ndims(first_parameter) >= 3
    n_batches = size(first_parameter,1); n_cells = size(first_parameter,2);
    n_epochs = size(first_parameter,3);
else
    n_batches = size(first_parameter,1); n_cells = 1; n_epochs = size(first_parameter,2);
end
cell_index = 1;
n_parameters = numel(parameter_names);
parameter_history = nan(n_batches,n_epochs,n_parameters);
for p = 1:n_parameters
    values = S.params.(parameter_names{p});
    if ndims(values) >= 3
        assert(size(values,1)==n_batches && size(values,2)>=cell_index && size(values,3)==n_epochs, ...
            'Unexpected shape for params.%s.',parameter_names{p});
        parameter_history(:,:,p) = reshape(values(:,cell_index,:),n_batches,n_epochs);
    else
        assert(n_cells==1 && isequal(size(values),[n_batches,n_epochs]), ...
            'Unexpected shape for params.%s.',parameter_names{p});
        parameter_history(:,:,p) = values;
    end
end
assert(n_epochs >= 7,'At least seven saved epochs are needed for this analysis.');

% Q95-Q05 scaling makes changes comparable despite units ranging from seconds
% to conductances. The scale is computed from the whole sampled parameter cloud.
flat_parameters = reshape(parameter_history,[],n_parameters);
robust_scale = prctile(flat_parameters,95,1)-prctile(flat_parameters,5,1);
fallback_scale = std(flat_parameters,0,1,'omitnan');
bad_scale = ~isfinite(robust_scale) | robust_scale<=eps;
robust_scale(bad_scale) = fallback_scale(bad_scale);
robust_scale(~isfinite(robust_scale) | robust_scale<=eps) = 1;

normalized_steps = diff(parameter_history,1,2)./reshape(robust_scale,1,1,[]);
movement = sqrt(mean(normalized_steps.^2,3));
n_transitions = size(movement,2);
window = min(5,floor(n_transitions/2));
trend_window = min(8,n_transitions);
early_steps = 1:window;
late_steps = n_transitions-window+1:n_transitions;
late_epochs = n_epochs-window+1:n_epochs;

early_rms = sqrt(mean(movement(:,early_steps).^2,2));
late_rms = sqrt(mean(movement(:,late_steps).^2,2));
late_to_early = late_rms./max(early_rms,eps);

% A continuous high-movement tail is expected, so these are descriptive flags.
high_cutoff = prctile(late_rms,90);
high_movement = late_rms >= high_cutoff;
late_median = median(late_rms,'omitnan');
late_mad = median(abs(late_rms-late_median),'omitnan');
robust_cutoff = late_median + 3*1.4826*late_mad;
robust_outlier = late_rms > robust_cutoff;

late_delta = normalized_steps(:,late_steps,:);
net_displacement = reshape(sum(late_delta,2),n_batches,n_parameters);
path_length = reshape(sum(abs(late_delta),2),n_batches,n_parameters);
path_efficiency = sum(abs(net_displacement),2)./max(sum(path_length,2),eps);
step_products = late_delta(:,2:end,:).*late_delta(:,1:end-1,:);
sign_flip_fraction = mean(reshape(step_products<0,n_batches,[]),2);

late_slope = nan(n_batches,1);
x_trend = (1:trend_window)';
for b = 1:n_batches
    fit_values = log10(movement(b,end-trend_window+1:end)'+eps);
    coefficients = polyfit(x_trend,fit_values,1);
    late_slope(b) = coefficients(1);
end

late_parameter_rms = reshape(sqrt(mean(late_delta.^2,2)),n_batches,n_parameters);
parameter_energy = late_parameter_rms.^2;
energy_fraction = parameter_energy./max(sum(parameter_energy,2),eps);
[dominant_share,dominant_index] = max(energy_fraction,[],2);
dominant_parameter = string(parameter_names(dominant_index))';

log_parameter_movement = log10(late_parameter_rms+eps);
movement_center = median(log_parameter_movement,1,'omitnan');
movement_mad = median(abs(log_parameter_movement-movement_center),1,'omitnan');
movement_mad = max(1.4826*movement_mad,eps);
parameter_robust_z = (log_parameter_movement-movement_center)./movement_mad;
parameter_extreme = any(parameter_robust_z>3,2);

% Separate boundary contact from movement: sticking/chattering at a clamp is
% different from an unconstrained drift direction.
[lower_bounds,upper_bounds] = parameter_bounds(parameter_names);
boundary_hit = false(n_batches,n_parameters);
boundary_rate = zeros(n_batches,n_parameters);
for p = 1:n_parameters
    tail_values = parameter_history(:,late_epochs,p);
    tolerance = 1e-6*max([1,abs(lower_bounds(p)),abs(upper_bounds(p))],[],'omitnan');
    hit = false(size(tail_values));
    if isfinite(lower_bounds(p)), hit = hit | abs(tail_values-lower_bounds(p))<=tolerance; end
    if isfinite(upper_bounds(p)), hit = hit | abs(tail_values-upper_bounds(p))<=tolerance; end
    boundary_hit(:,p) = any(hit,2);
    boundary_rate(:,p) = mean(hit,2);
end
any_boundary_hit = any(boundary_hit,2);
number_boundaries_hit = sum(boundary_hit,2);

sse = reshape_loss(S.sse_losses_all,n_batches,n_epochs,cell_index,'sse_losses_all');
cv = reshape_loss(S.cv_losses_all,n_batches,n_epochs,cell_index,'cv_losses_all');
final_sse = sse(:,end); final_cv = cv(:,end);
late_sse_change = sse(:,end)-sse(:,max(1,end-window));
late_cv_change = cv(:,end)-cv(:,max(1,end-window));
late_sse_volatility = std(diff(sse(:,max(1,end-window):end),1,2),0,2,'omitnan');
late_cv_volatility = std(diff(cv(:,max(1,end-window):end),1,2),0,2,'omitnan');

persistent_mask = late_to_early>=1;
stability_class = repmat("contracting",n_batches,1);
stability_class(persistent_mask & path_efficiency>=0.5) = "persistent drift";
stability_class(persistent_mask & path_efficiency<0.5) = "persistent oscillation";

[rho_sse,p_sse] = corr(late_rms,final_sse,'Type','Spearman','Rows','complete');
[rho_cv,p_cv] = corr(late_rms,final_cv,'Type','Spearman','Rows','complete');
[rho_sse_vol,p_sse_vol] = corr(late_rms,late_sse_volatility,'Type','Spearman','Rows','complete');
[rho_cv_vol,p_cv_vol] = corr(late_rms,late_cv_volatility,'Type','Spearman','Rows','complete');

%% Figure 1: Does the population converge, and what does averaging hide?
transition = 1:n_transitions;
movement_percentiles = prctile(movement,[10 25 50 75 90 99],1);
mean_parameter = squeeze(mean(parameter_history,1,'omitnan'));
movement_of_mean = sqrt(mean((diff(mean_parameter,1,1)./robust_scale).^2,2));

f1 = figure('Position',[50 50 1250 820]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile; hold on;
fill([transition fliplr(transition)],[movement_percentiles(1,:) fliplr(movement_percentiles(5,:))], ...
    [0.75 0.85 1],'EdgeColor','none','FaceAlpha',0.45);
fill([transition fliplr(transition)],[movement_percentiles(2,:) fliplr(movement_percentiles(4,:))], ...
    [0.35 0.60 0.90],'EdgeColor','none','FaceAlpha',0.45);
semilogy(transition,movement_percentiles(3,:),'b-','LineWidth',2);
semilogy(transition,movement_percentiles(6,:),'Color',[0.75 0.15 0.15],'LineWidth',1.5);
xlabel('Saved-epoch transition'); ylabel('Normalized joint update norm');
title('Population contraction with a persistent tail'); grid on;
legend('10th-90th percentile','25th-75th percentile','median','99th percentile','Location','best');

nexttile; hold on;
semilogy(transition,mean(movement,1,'omitnan'),'k-','LineWidth',2);
semilogy(transition,movement_of_mean,'r--','LineWidth',2);
xlabel('Saved-epoch transition'); ylabel('Normalized movement'); grid on;
title('Difference first vs. average first');
legend('Mean of batchwise movement','Movement of across-batch mean','Location','best');

[~,batch_order] = sort(late_rms,'ascend');
nexttile; imagesc(transition,1:n_batches,log10(movement(batch_order,:)+eps)); axis xy;
xlabel('Saved-epoch transition'); ylabel('Batches sorted by late movement');
title('Batch update heatmap'); colorbar;

nexttile; hold on;
median_sse = median(sse,1,'omitnan'); median_cv = median(cv,1,'omitnan');
yyaxis left; semilogy(1:n_epochs,median_sse,'b-o','LineWidth',1.5); ylabel('Median SSE');
yyaxis right; plot(1:n_epochs,median_cv,'r-o','LineWidth',1.5); ylabel('Median CV loss');
xlabel('Saved epoch'); title('Population loss histories'); grid on;
exportgraphics(f1,fullfile(results_dir,'01_population_convergence.pdf'),'ContentType','vector');

%% Figure 2: Separate contraction, continued drift, and oscillation
class_order = ["contracting","persistent drift","persistent oscillation"];
class_colors = [0.25 0.55 0.85; 0.90 0.45 0.12; 0.65 0.20 0.65];
f2 = figure('Position',[80 80 1250 820]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile; hold on;
for c = 1:numel(class_order)
    use = stability_class==class_order(c);
    loglog(early_rms(use),late_rms(use),'o','MarkerSize',4,'MarkerFaceColor',class_colors(c,:), ...
        'MarkerEdgeColor','none');
end
limits = [max(eps,min([early_rms;late_rms])) max([early_rms;late_rms])];
plot(limits,limits,'k--','LineWidth',1); axis square; grid on;
xlabel('Early RMS movement'); ylabel('Late RMS movement'); title('Contraction by batch');
legend([class_order,"late = early"],'Location','best');

nexttile; hold on;
for c = 1:numel(class_order)
    use = stability_class==class_order(c);
    scatter(late_to_early(use),path_efficiency(use),16,class_colors(c,:),'filled');
end
xline(1,'k--'); yline(0.5,'k--'); grid on;
xlabel('Late / early RMS movement'); ylabel('Late path efficiency');
title('Drift (high efficiency) vs. oscillation');

nexttile; scatter(late_rms,late_sse_volatility,18,double(high_movement),'filled');
set(gca,'XScale','log','YScale','log'); grid on; colorbar;
xlabel('Late RMS movement'); ylabel('Late SSE-update volatility');
title(sprintf('Movement vs SSE volatility: \rho_s = %.3f (p=%.2g)',rho_sse_vol,p_sse_vol));

nexttile; scatter(late_rms,final_sse,18,double(high_movement),'filled');
set(gca,'XScale','log','YScale','log'); grid on; colorbar;
xlabel('Late RMS movement'); ylabel('Final SSE');
title(sprintf('Movement vs final SSE: \rho_s = %.3f (p=%.2g)',rho_sse,p_sse));
exportgraphics(f2,fullfile(results_dir,'02_batch_stability_classes.pdf'),'ContentType','vector');

%% Figure 3: Which parameters and boundaries drive the high-movement tail?
rest = ~high_movement;
late_rms_all = median(late_parameter_rms,1,'omitnan');
late_rms_high = median(late_parameter_rms(high_movement,:),1,'omitnan');
late_rms_rest = median(late_parameter_rms(rest,:),1,'omitnan');
energy_high = mean(energy_fraction(high_movement,:),1,'omitnan');
boundary_all = mean(boundary_rate,1,'omitnan');
boundary_high = mean(boundary_rate(high_movement,:),1,'omitnan');
boundary_rest = mean(boundary_rate(rest,:),1,'omitnan');
dominant_count_high = accumarray(dominant_index(high_movement),1,[n_parameters 1]);
late_correlation_high = corr(reshape(late_delta(high_movement,:,:),[],n_parameters),'Rows','pairwise');

f3 = figure('Position',[110 70 1350 900]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile; bar([late_rms_rest(:),late_rms_high(:)]); set(gca,'XTick',1:n_parameters, ...
    'XTickLabel',parameter_names,'XTickLabelRotation',45); ylabel('Median normalized late RMS');
title('Parameter movement: remaining 90% vs top 10%'); legend('remaining 90%','top 10%'); grid on;

nexttile; bar(energy_high); set(gca,'XTick',1:n_parameters,'XTickLabel',parameter_names, ...
    'XTickLabelRotation',45); ylabel('Mean fraction of squared movement');
title('Drivers within the top 10% movement tail'); grid on;

nexttile; bar([boundary_rest(:),boundary_high(:)]); set(gca,'XTick',1:n_parameters, ...
    'XTickLabel',parameter_names,'XTickLabelRotation',45); ylabel('Fraction of late states at a clamp');
title('Boundary contact'); legend('remaining 90%','top 10%'); grid on;

nexttile; imagesc(late_correlation_high,[-1 1]); axis square; colorbar; colormap(gca,bluewhitered(256));
set(gca,'XTick',1:n_parameters,'YTick',1:n_parameters,'XTickLabel',parameter_names, ...
    'YTickLabel',parameter_names,'XTickLabelRotation',45); title('Correlated late updates in top 10%');
exportgraphics(f3,fullfile(results_dir,'03_parameter_drivers.pdf'),'ContentType','vector');

%% Figure 4: Detailed trajectories of the strongest movers
[~,ranked_batches] = sort(late_rms,'descend');
n_show = min(8,n_batches);
colors = lines(n_parameters);
f4 = figure('Position',[140 40 1400 1000]);
tiledlayout(4,2,'Padding','compact','TileSpacing','compact');
for rank_index = 1:n_show
    b = ranked_batches(rank_index);
    displacement = reshape((parameter_history(b,:,:)-parameter_history(b,1,:))./ ...
        reshape(robust_scale,1,1,[]),n_epochs,n_parameters);
    nexttile; plot(1:n_epochs,displacement,'LineWidth',1.1); yline(0,'k:'); grid on;
    xlabel('Saved epoch'); ylabel('Change / robust scale');
    title(sprintf('Batch %d | %s | late/early %.2f | SSE %.0f', ...
        b,stability_class(b),late_to_early(b),final_sse(b)),'Interpreter','none');
end
legend(parameter_names,'Interpreter','none','Location','eastoutside');
exportgraphics(f4,fullfile(results_dir,'04_strongest_batch_trajectories.pdf'),'ContentType','vector');

%% Save batch- and parameter-level results
batch = (1:n_batches)';
batch_table = table(batch,early_rms,late_rms,late_to_early,late_slope,path_efficiency, ...
    sign_flip_fraction,high_movement,robust_outlier,parameter_extreme,stability_class, ...
    dominant_parameter,dominant_share,any_boundary_hit,number_boundaries_hit, ...
    final_sse,late_sse_change,late_sse_volatility,final_cv,late_cv_change,late_cv_volatility);
batch_table = sortrows(batch_table,'late_rms','descend');
writetable(batch_table,fullfile(results_dir,'batch_stability_metrics.csv'));

parameter_table = table(string(parameter_names)',robust_scale',late_rms_all',late_rms_rest', ...
    late_rms_high',energy_high',dominant_count_high,boundary_all',boundary_rest',boundary_high', ...
    sum(parameter_robust_z>3,1)',lower_bounds',upper_bounds','VariableNames', ...
    {'parameter','robust_scale','late_rms_all','late_rms_remaining90','late_rms_top10', ...
    'energy_fraction_top10','dominant_count_top10','boundary_rate_all','boundary_rate_remaining90', ...
    'boundary_rate_top10','extreme_batch_count','lower_bound','upper_bound'});
writetable(parameter_table,fullfile(results_dir,'parameter_stability_summary.csv'));

save(fullfile(results_dir,'parameter_stability_analysis.mat'),'parameter_names','robust_scale', ...
    'movement','early_rms','late_rms','late_to_early','late_slope','path_efficiency', ...
    'sign_flip_fraction','high_movement','robust_outlier','parameter_extreme','stability_class', ...
    'late_parameter_rms','energy_fraction','boundary_hit','boundary_rate','batch_table','parameter_table');

summary_file = fullfile(results_dir,'summary.txt');
fid = fopen(summary_file,'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'Wide-run parameter stability analysis\n');
fprintf(fid,'=====================================\n');
fprintf(fid,'Input: %s\n',run_file);
fprintf(fid,'Batches: %d | saved epochs: %d | parameters: %d\n\n',n_batches,n_epochs,n_parameters);
fprintf(fid,'Median joint movement: %.5g at transition 1, %.5g at transition %d (%.1f%% change).\n', ...
    median(movement(:,1)),median(movement(:,end)),n_transitions, ...
    100*(median(movement(:,end))/median(movement(:,1))-1));
fprintf(fid,'Contracting batches (late/early < 1): %d/%d (%.1f%%).\n',sum(~persistent_mask),n_batches,100*mean(~persistent_mask));
fprintf(fid,'Persistent drift: %d/%d (%.1f%%).\n',sum(stability_class=="persistent drift"),n_batches,100*mean(stability_class=="persistent drift"));
fprintf(fid,'Persistent oscillation: %d/%d (%.1f%%).\n',sum(stability_class=="persistent oscillation"),n_batches,100*mean(stability_class=="persistent oscillation"));
fprintf(fid,'Top-10%% high-movement cutoff: %.5g (%d batches).\n',high_cutoff,sum(high_movement));
fprintf(fid,'Raw-scale robust cutoff (median + 3 scaled MAD): %.5g (%d batches).\n',robust_cutoff,sum(robust_outlier));
fprintf(fid,'Any parameter at a boundary in the late window: %.1f%% overall, %.1f%% in top 10%%, %.1f%% in remaining 90%%.\n\n', ...
    100*mean(any_boundary_hit),100*mean(any_boundary_hit(high_movement)),100*mean(any_boundary_hit(rest)));
fprintf(fid,'Spearman(late movement, final SSE) = %.4f, p = %.3g.\n',rho_sse,p_sse);
fprintf(fid,'Spearman(late movement, final CV) = %.4f, p = %.3g.\n',rho_cv,p_cv);
fprintf(fid,'Spearman(late movement, late SSE volatility) = %.4f, p = %.3g.\n',rho_sse_vol,p_sse_vol);
fprintf(fid,'Spearman(late movement, late CV volatility) = %.4f, p = %.3g.\n\n',rho_cv_vol,p_cv_vol);
fprintf(fid,'Top batches by late movement:\n');
for k = 1:min(15,n_batches)
    b = ranked_batches(k);
    fprintf(fid,'%2d. batch %4d | late %.5g | ratio %.3f | efficiency %.3f | %-22s | driver %s (%.1f%%) | SSE %.1f | CV %.4f\n', ...
        k,b,late_rms(b),late_to_early(b),path_efficiency(b),stability_class(b), ...
        dominant_parameter(b),100*dominant_share(b),final_sse(b),final_cv(b));
end
fprintf(fid,'\nInterpretation cautions:\n');
fprintf(fid,'- These are epoch-to-epoch parameter updates, not trial-to-trial changes.\n');
fprintf(fid,'- Top-10%% and median+3-MAD flags are descriptive diagnostics, not proof of a separate unstable population.\n');
fprintf(fid,'- Continued movement can be useful descent, stochastic-gradient noise, clamp chattering, or movement along a sloppy/non-identifiable direction.\n');
fprintf(fid,'- Twenty saved epochs cannot establish asymptotic convergence or divergence.\n');
fprintf(fid,'- The final post-epoch optimizer update is not present in the saved parameter history.\n');
clear cleanup;

fprintf('Saved stability analysis to:\n%s\n',results_dir);
fprintf('Contracting: %d/%d | persistent drift: %d | persistent oscillation: %d | robust late outliers: %d\n', ...
    sum(~persistent_mask),n_batches,sum(stability_class=="persistent drift"), ...
    sum(stability_class=="persistent oscillation"),sum(robust_outlier));
fprintf('Late movement correlations: final SSE %.3f, final CV %.3f, SSE volatility %.3f.\n', ...
    rho_sse,rho_cv,rho_sse_vol);

%% Local helpers
function loss = reshape_loss(values,n_batches,n_epochs,cell_index,name)
if ndims(values)>=3
    assert(size(values,1)==n_batches && size(values,2)>=cell_index && size(values,3)==n_epochs, ...
        'Unexpected shape for %s.',name);
    loss = reshape(values(:,cell_index,:),n_batches,n_epochs);
else
    assert(isequal(size(values),[n_batches,n_epochs]),'Unexpected shape for %s.',name);
    loss = values;
end
end

function [lower,upper] = parameter_bounds(names)
lower_map = struct('strf_gain',0.001,'strf_alpha',0.1,'output_ad',0,'abs_ref',0, ...
    'rel_ref_a',0.001,'rel_ref_b',0,'rel_ref_c',0.2,'on_ron_gsyn',0.001, ...
    'off_ron_gsyn',0.001,'sonoff_ron_gsyn',0.001,'on_sonoff_gsyn',0.001,'off_sonoff_gsyn',0.001);
upper_map = struct('strf_alpha',250,'rel_ref_c',1);
lower = nan(1,numel(names)); upper = nan(1,numel(names));
for k = 1:numel(names)
    if isfield(lower_map,names{k}), lower(k)=lower_map.(names{k}); end
    if isfield(upper_map,names{k}), upper(k)=upper_map.(names{k}); end
end
end

function map = bluewhitered(n)
if nargin<1, n=256; end
x = linspace(0,1,n)';
map = [min(1,2*x), 1-2*abs(x-0.5), min(1,2*(1-x))];
end
