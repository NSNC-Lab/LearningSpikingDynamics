function [nccc, TTRC, mean_pred_cor, pred_cor, pair_cor] = ...
    calculate_nccc_from_histograms(ris, sim_PSTH)
%CALCULATE_NCCC_FROM_HISTOGRAMS Robust noise-corrected correlation.
% Correlation is undefined for constant vectors. A non-positive TTRC also
% makes the reliability correction undefined, so those cases return NaN.

n_trials = size(ris, 1);
pair_cor = nan(n_trials);
trial_has_variance = max(ris, [], 2) ~= min(ris, [], 2);

for first = 1:n_trials-1
    if ~trial_has_variance(first); continue; end
    for second = first+1:n_trials
        if ~trial_has_variance(second); continue; end
        value = corr(ris(first,:).', ris(second,:).', 'Rows', 'complete');
        if isfinite(value)
            pair_cor(first, second) = value;
        end
    end
end

upper_triangle = pair_cor(triu(true(n_trials), 1));
TTRC = mean(upper_triangle, 'omitnan');

pred_cor = nan(n_trials, 1);
sim_has_variance = max(sim_PSTH) ~= min(sim_PSTH);
if sim_has_variance
    for trial = 1:n_trials
        if ~trial_has_variance(trial); continue; end
        value = corr(ris(trial,:).', sim_PSTH(:), 'Rows', 'complete');
        if isfinite(value)
            pred_cor(trial) = value;
        end
    end
end
mean_pred_cor = mean(pred_cor, 'omitnan');

if isfinite(TTRC) && TTRC > 0 && isfinite(mean_pred_cor)
    nccc = mean_pred_cor / sqrt(TTRC);
else
    nccc = NaN;
end
end
