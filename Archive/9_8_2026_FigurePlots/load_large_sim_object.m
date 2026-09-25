function sim_object = load_large_sim_object(filename)
%LOAD_LARGE_SIM_OBJECT Load a normal result or reconstruct split output parts.

info = whos('-file', filename);
names = {info.name};

if any(strcmp(names, 'output'))
    sim_object = load(filename);
    return
end

part_mask = startsWith(names, 'output_part_');
if ~any(part_mask)
    error('load_large_sim_object:MissingOutput', ...
        'No output or output_part_* variables were found in %s.', filename);
end

part_info = info(part_mask);
[~, order] = sort({part_info.name});
part_info = part_info(order);

metadata_names = names(~part_mask);
sim_object = load(filename, metadata_names{:});

first_size = part_info(1).size;
output_size = first_size;
output_size(1) = sum(arrayfun(@(item) item.size(1), part_info));
sim_object.output = zeros(output_size, part_info(1).class);

next_batch = 1;
subs = repmat({':'}, 1, numel(output_size));
for k = 1:numel(part_info)
    part_name = part_info(k).name;
    part_data = load(filename, part_name);
    part = part_data.(part_name);
    final_batch = next_batch + size(part, 1) - 1;
    subs{1} = next_batch:final_batch;
    sim_object.output(subs{:}) = part;
    next_batch = final_batch + 1;
    clear part part_data
end

if isfield(sim_object, 'output_original_shape')
    expected_size = double(sim_object.output_original_shape(:).');
    if ~isequal(size(sim_object.output), expected_size)
        error('load_large_sim_object:ShapeMismatch', ...
            'Reconstructed output has size %s; expected %s.', ...
            mat2str(size(sim_object.output)), mat2str(expected_size));
    end
end
end
