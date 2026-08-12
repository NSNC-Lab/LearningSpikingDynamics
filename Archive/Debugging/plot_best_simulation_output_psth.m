clear; close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));

outputPath = fullfile(repoRoot, 'simulation_output.mat');
configPath = fullfile(repoRoot, 'simulation_config.yaml');
saveDir = fullfile(repoRoot, 'Archive', 'Debugging', 'simulation_output_best_psth');
saveFigures = true;

cfg = readConfig(configPath);
cfg.repoRoot = repoRoot;
loadedOutput = load(outputPath);
output = loadedOutput.output;

if ndims(output) ~= 4
    error('Expected output to have shape batch x trial x cell x time. Got %s.', mat2str(size(output)));
end

[nBatch, nTrial, nCell, simLen] = size(output);
cfg.simLen = firstNonempty(cfg.simLen, simLen);
cfg.cellTargets = normalizeCellTargets(cfg.cellTargets, nCell);

if cfg.simLen ~= simLen
    warning('Config sim_len (%d) does not match output time length (%d). Using output length.', cfg.simLen, simLen);
    cfg.simLen = simLen;
end

if saveFigures && ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

[dataRaster, dataPsth, binCenters] = buildDataRasterAndPsth(cfg, nCell, nTrial);
[simPsth, lossPerBatchCell] = buildSimPsthAndLoss(output, dataPsth, cfg);

[bestLossPerCell, bestBatchPerCell] = min(lossPerBatchCell, [], 1);
[globalBestLoss, globalLinearIndex] = min(lossPerBatchCell(:));
[globalBestBatch, globalBestCellIndex] = ind2sub(size(lossPerBatchCell), globalLinearIndex);

fprintf('Global best: cell target %d, output batch %d, PSTH MSE %.4f\n', ...
    cfg.cellTargets(globalBestCellIndex), globalBestBatch, globalBestLoss);

for cellIndex = 1:nCell
    batchIndex = bestBatchPerCell(cellIndex);
    cellTarget = cfg.cellTargets(cellIndex);

    simRaster = reshape(output(batchIndex, :, cellIndex, :), [nTrial, simLen]);
    thisDataRaster = reshape(dataRaster(cellIndex, :, :), [nTrial, simLen]);
    thisSimPsth = reshape(simPsth(batchIndex, cellIndex, :), [1, numel(binCenters)]);
    thisDataPsth = reshape(dataPsth(cellIndex, :), [1, numel(binCenters)]);

    fig = figure('Color', 'w', 'Name', sprintf('Cell %d best batch %d', cellTarget, batchIndex));
    tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plotRaster(simRaster, cfg.dtMs);
    title(sprintf('Simulation raster | cell %d | batch %d | PSTH MSE %.4f', ...
        cellTarget, batchIndex, bestLossPerCell(cellIndex)));

    nexttile;
    plotRaster(thisDataRaster, cfg.dtMs);
    title(sprintf('Data raster | cell %d', cellTarget));

    nexttile;
    plot(binCenters, thisDataPsth, 'k-', 'LineWidth', 1.5); hold on;
    plot(binCenters, thisSimPsth, 'r-', 'LineWidth', 1.2);
    xlim([0, simLen * cfg.dtMs / 1000]);
    xlabel('Time (s)');
    ylabel(sprintf('Spikes / %g ms bin', cfg.psthGranularity * cfg.dtMs));
    legend({'Data', 'Simulation'}, 'Location', 'best');
    title('PSTH overlay');
    grid on;

    if saveFigures
        saveas(fig, fullfile(saveDir, sprintf('cell_%03d_best_batch_%03d.png', cellTarget, batchIndex)));
    end
end

lossTable = table(cfg.cellTargets(:), bestBatchPerCell(:), bestLossPerCell(:), ...
    'VariableNames', {'CellTarget', 'BestBatch', 'BestPsthMse'});
disp(lossTable);

if saveFigures
    writetable(lossTable, fullfile(saveDir, 'best_output_psth_losses.csv'));
    fprintf('Saved plots and losses to %s\n', saveDir);
end

function cfg = readConfig(configPath)
    text = fileread(configPath);
    cfg.dtMs = readScalar(text, 'dt', 0.1);
    cfg.simLen = readScalar(text, 'sim_len', []);
    cfg.psthGranularity = readScalar(text, 'PSTH_granularity', 100);
    cfg.dataTarget = readString(text, 'data_target', 'peak');
    cfg.dataPath = readString(text, 'data', 'Data/Data/all_units_info_with_polished_criteria_modified_perf.mat');
    cfg.cellTargets = readNumberList(text, 'cell_targets', []);
end

function value = readScalar(text, key, defaultValue)
    pattern = ['(?m)^\s*' regexptranslate('escape', key) '\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'];
    token = regexp(text, pattern, 'tokens', 'once');
    if isempty(token)
        value = defaultValue;
    else
        value = str2double(token{1});
    end
end

function value = readString(text, key, defaultValue)
    pattern = ['(?m)^\s*' regexptranslate('escape', key) '\s*:\s*"?([^"\r\n#]+)"?'];
    token = regexp(text, pattern, 'tokens', 'once');
    if isempty(token)
        value = defaultValue;
    else
        value = strtrim(token{1});
    end
end

function values = readNumberList(text, key, defaultValue)
    pattern = ['(?m)^\s*' regexptranslate('escape', key) '\s*:\s*\[([^\]]*)\]'];
    token = regexp(text, pattern, 'tokens', 'once');
    if isempty(token)
        values = defaultValue;
        return;
    end
    values = sscanf(strrep(token{1}, ',', ' '), '%f').';
end

function value = firstNonempty(value, fallback)
    if isempty(value)
        value = fallback;
    end
end

function cellTargets = normalizeCellTargets(cellTargets, nCell)
    if isempty(cellTargets)
        cellTargets = 1:nCell;
    end
    if numel(cellTargets) ~= nCell
        warning('Config has %d cell target(s), but output has %d cell(s). Using 1:%d for labels.', ...
            numel(cellTargets), nCell, nCell);
        cellTargets = 1:nCell;
    end
end

function [dataRaster, dataPsth, binCenters] = buildDataRasterAndPsth(cfg, nCell, nTrial)
    dataPath = cfg.dataPath;
    if ~isfile(dataPath)
        dataPath = fullfile(cfg.repoRoot, strrep(cfg.dataPath, '/', filesep));
    end

    loadedData = load(dataPath, 'all_data');
    allData = loadedData.all_data;

    nBin = floor(cfg.simLen / cfg.psthGranularity);
    binEdges = (0:nBin) * cfg.psthGranularity * cfg.dtMs / 1000;
    binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

    dataRaster = zeros(nCell, nTrial, cfg.simLen);
    dataPsth = zeros(nCell, nBin);
    dtSeconds = cfg.dtMs / 1000;

    for cellIndex = 1:nCell
        cellTarget = cfg.cellTargets(cellIndex);
        unit = allData(cellTarget);

        if strcmpi(cfg.dataTarget, 'peak')
            angleIndex = peakAngleIndex(unit);
        else
            error('This script currently supports data_target: peak. Found "%s".', cfg.dataTarget);
        end

        timestamps = unit.ctrl_tar1_timestamps;
        for trialIndex = 1:nTrial
            trialSpikes = getTrialSpikes(timestamps, trialIndex, angleIndex);
            trialSpikes = double(trialSpikes(:)).';
            trialSpikes = trialSpikes(trialSpikes >= 0 & trialSpikes <= cfg.simLen * dtSeconds);

            if isempty(trialSpikes)
                continue;
            end

            spikeIndices = floor(trialSpikes / dtSeconds) + 1;
            spikeIndices = spikeIndices(spikeIndices >= 1 & spikeIndices <= cfg.simLen);
            dataRaster(cellIndex, trialIndex, spikeIndices) = 1;
            dataPsth(cellIndex, :) = dataPsth(cellIndex, :) + histcounts(trialSpikes, binEdges);
        end
    end
end

function angleIndex = peakAngleIndex(unit)
    tuningType = string(unit.tuning_type);
    if contains(tuningType, 'contra', 'IgnoreCase', true)
        angleIndex = 1;
    elseif contains(tuningType, '45', 'IgnoreCase', true)
        angleIndex = 2;
    elseif contains(tuningType, 'center', 'IgnoreCase', true)
        angleIndex = 3;
    else
        angleIndex = 4;
    end
end

function trialSpikes = getTrialSpikes(timestamps, trialIndex, angleIndex)
    if iscell(timestamps)
        trialSpikes = timestamps{trialIndex, angleIndex};
    else
        trialSpikes = timestamps(trialIndex, angleIndex);
    end
end

function [simPsth, lossPerBatchCell] = buildSimPsthAndLoss(output, dataPsth, cfg)
    [nBatch, ~, nCell, ~] = size(output);
    nBin = floor(cfg.simLen / cfg.psthGranularity);
    simPsth = zeros(nBatch, nCell, nBin);

    for binIndex = 1:nBin
        timeIndex = (binIndex - 1) * cfg.psthGranularity + 1 : binIndex * cfg.psthGranularity;
        counts = squeeze(sum(sum(output(:, :, :, timeIndex), 2), 4));
        simPsth(:, :, binIndex) = reshape(counts, [nBatch, nCell]);
    end

    lossPerBatchCell = zeros(nBatch, nCell);
    for cellIndex = 1:nCell
        dataTrace = reshape(dataPsth(cellIndex, :), [1, 1, nBin]);
        lossPerBatchCell(:, cellIndex) = mean((simPsth(:, cellIndex, :) - dataTrace) .^ 2, 3);
    end
end

function plotRaster(raster, dtMs)
    [trialIndex, timeIndex] = find(raster > 0);
    scatter((timeIndex - 1) * dtMs / 1000, trialIndex, 8, 'filled', ...
        'MarkerFaceColor', [0.05 0.45 0.85], 'MarkerFaceAlpha', 0.8);
    ylim([0.5, size(raster, 1) + 0.5]);
    xlim([0, size(raster, 2) * dtMs / 1000]);
    ylabel('Trial');
    xlabel('Time (s)');
    set(gca, 'YDir', 'reverse', 'Box', 'off');
end
