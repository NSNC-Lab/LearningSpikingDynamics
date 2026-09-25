function build_cell_browser_data(workbook_path, data_path, wav_path, output_path)
%BUILD_CELL_BROWSER_DATA Export classifications and spike plots for the browser.

if nargin < 4
    output_path = fullfile(fileparts(mfilename('fullpath')), 'cell_data.js');
end

raw = readcell(workbook_path);
loaded = load(data_path, 'all_data');
all_data = loaded.all_data;

response_labels = {'Onset', 'Offset', 'Both', 'Neither'};
quality_labels = {'Non-stationary', 'Noise', 'Not really stereotyped', ...
    'Low stereotypy', 'Moderate stereotypy', 'High stereotypy', 'Artifacting'};

duration = 2.9801;
bin_edges = 0:0.02:2.98;
bin_times = bin_edges(1:end-1) + diff(bin_edges) / 2;
cells = repmat(struct(), 1, 220);

for k = 1:220
    row = k + 1;
    response = 'Unclassified';
    for column = 2:5
        if is_marked(raw, row, column)
            response = response_labels{column - 1};
            break
        end
    end

    quality = {};
    for column = 6:12
        if is_marked(raw, row, column)
            quality{end + 1} = quality_labels{column - 5}; %#ok<AGROW>
        end
    end
    if isempty(quality)
        quality = {'Unclassified'};
    end

    tuning = char(string(all_data(k).tuning_type));
    tuning_lower = lower(tuning);
    if contains(tuning_lower, 'contra')
        focus = 1;
    elseif contains(tuning_lower, '45')
        focus = 2;
    elseif contains(tuning_lower, 'center')
        focus = 3;
    elseif contains(tuning_lower, 'ipsi')
        focus = 4;
    else
        error('Unknown tuning type for cell %d: %s', k, tuning);
    end

    source_spikes = all_data(k).ctrl_tar1_timestamps(:, focus);
    trials = cell(1, 10);
    pooled = [];
    for trial = 1:10
        values = double(source_spikes{trial}(:));
        values = values(values > 0 & values < duration);
        trials{trial} = round(values(:).', 5);
        pooled = [pooled; values]; %#ok<AGROW>
    end

    counts = movmean(histcounts(pooled, bin_edges), 3);
    cells(k).id = k;
    cells(k).response = response;
    cells(k).quality = quality;
    layer = all_data(k).layer;
    if ismissing_value(layer) || strcmpi(char(string(layer)), 'nan')
        cells(k).layer = '';
    else
        cells(k).layer = char(string(layer));
    end
    cells(k).tuning = tuning;
    cells(k).note = cell_text(raw, row, 13);
    cells(k).trials = trials;
    cells(k).psth = round(counts, 4);
end

[waveform, sample_rate] = audioread(wav_path);
if size(waveform, 2) > 1
    waveform = mean(waveform, 2);
end
start_sample = max(1, round(0.25 * sample_rate) - 1);
waveform = waveform(start_sample:end);
waveform = waveform(1:min(numel(waveform), round(duration * sample_rate)));
sample_indices = unique(round(linspace(1, numel(waveform), 420)));
waveform = waveform(sample_indices);
waveform = waveform / max(abs(waveform));
waveform_times = (sample_indices - 1) / sample_rate;

payload.duration = duration;
payload.binTimes = round(bin_times, 4);
payload.waveformTimes = round(waveform_times, 5);
payload.waveform = round(waveform(:).', 5);
payload.eventWindows = [0.38, 0.50; 0.96, 1.20];
payload.cells = cells;

json = jsonencode(payload);
file_id = fopen(output_path, 'w');
if file_id < 0
    error('Could not open output file: %s', output_path);
end
cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id, 'window.CELL_BROWSER_DATA = %s;\n', json);
fprintf('Wrote %d cells to %s\n', numel(cells), output_path);
end


function tf = is_marked(raw, row, column)
value = raw{row, column};
tf = (ischar(value) || isstring(value)) && strcmpi(strtrim(string(value)), 'x');
end


function value = cell_text(raw, row, column)
entry = raw{row, column};
if ismissing_value(entry)
    value = '';
else
    value = char(string(entry));
end
end


function tf = ismissing_value(value)
tf = isempty(value);
if ~tf
    try
        tf = all(ismissing(value), 'all');
    catch
        tf = false;
    end
end
if ~tf && isstring(value)
    tf = all(strlength(value) == 0, 'all');
end
end
