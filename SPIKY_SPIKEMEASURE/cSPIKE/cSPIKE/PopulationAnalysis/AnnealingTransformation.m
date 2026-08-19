function [ trialpopulation ] = AnnealingTransformation( currentPopulation)

Neurons = size(currentPopulation,1);

% Forcing to try to add a neuron if there is only one in the
% current population
forceAdd = 0;
if(sum(currentPopulation) == 1)
    forceAdd = 1;
end
% Forcing to try to remove a neuron if the set is the whole
% population.
forceRemove = 0;
if(sum(currentPopulation) == Neurons)
    forceRemove = 1;
end

% 50-50 chance of adding or removing a spiketrain if not forced
if (forceAdd || rand(1) > 0.5) && ~forceRemove
    % Add
    
    % Chose a neuron randomly from the ones that are not in
    % population
    index = floor(rand(1)*Neurons)+1;
    while currentPopulation(index) ~= 0;
        index = floor(rand(1)*Neurons)+1;
    end
    
    trialpopulation = currentPopulation;
    trialpopulation(index) = 1;
else
    
    %Remove
    
    % Chose a neuron randomly from the ones that are in the
    % population
    index = floor(rand(1)*Neurons)+1;
    while currentPopulation(index) ~= 1;
        index = floor(rand(1)*Neurons)+1;
    end
    trialpopulation = currentPopulation;
    trialpopulation(index) = 0;
end


end

