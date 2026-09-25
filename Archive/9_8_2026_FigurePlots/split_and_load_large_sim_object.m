function [sim_object, split_filename] = split_and_load_large_sim_object(source_filename, varargin)
%SPLIT_AND_LOAD_LARGE_SIM_OBJECT Split an oversized MAT file and load it.
%
% sim_object = split_and_load_large_sim_object(source_filename)
% sim_object = split_and_load_large_sim_object(source_filename, ...
%     'PythonExecutable', '/path/to/python', ...
%     'Destination', '/path/to/result_split.mat', ...
%     'MaxPartBytes', 1e9)

parser = inputParser;
parser.addRequired('source_filename', @(x) ischar(x) || isstring(x));
parser.addParameter('PythonExecutable', '', ...
    @(x) ischar(x) || isstring(x));
parser.addParameter('Destination', '', ...
    @(x) ischar(x) || isstring(x));
parser.addParameter('MaxPartBytes', 1e9, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
parser.parse(source_filename, varargin{:});

source_filename = char(parser.Results.source_filename);
if ~isfile(source_filename)
    error('split_and_load_large_sim_object:MissingSource', ...
        'Source MAT file was not found: %s', source_filename);
end

if isempty(parser.Results.Destination)
    [source_dir, source_name] = fileparts(source_filename);
    split_filename = fullfile(source_dir, [source_name '_split.mat']);
else
    split_filename = char(parser.Results.Destination);
end

utility_dir = fileparts(mfilename('fullpath'));
splitter = fullfile(utility_dir, 'split_large_mat_output.py');
if ~isfile(splitter)
    error('split_and_load_large_sim_object:MissingSplitter', ...
        'Python splitter was not found: %s', splitter);
end

python_executable = char(parser.Results.PythonExecutable);
if isempty(python_executable)
    python_executable = matlab_python_executable();
end

if ~isfile(split_filename)
    command = sprintf('%s %s %s --max-part-bytes %.0f', ...
        shell_quote(python_executable), ...
        shell_quote(splitter), ...
        shell_quote(source_filename), ...
        parser.Results.MaxPartBytes);

    fprintf('Splitting large MAT file with Python...\n');
    [status, command_output] = system(command, '-echo');
    if status ~= 0
        error('split_and_load_large_sim_object:PythonFailed', ...
            'MAT-file splitting failed (status %d):\n%s', ...
            status, command_output);
    end
else
    fprintf('Using existing split file: %s\n', split_filename);
end

sim_object = load_large_sim_object(split_filename);
end


function executable = matlab_python_executable()
executable = 'python';
try
    environment = pyenv;
    configured = char(environment.Executable);
    if ~isempty(configured) && isfile(configured)
        executable = configured;
    end
catch
    % Fall back to the Python executable available on PATH.
end
end


function quoted = shell_quote(value)
value = char(value);
if ispc
    quoted = ['"' strrep(value, '"', '""') '"'];
else
    quoted = ['''' strrep(value, '''', '''"''"''') ''''];
end
end
