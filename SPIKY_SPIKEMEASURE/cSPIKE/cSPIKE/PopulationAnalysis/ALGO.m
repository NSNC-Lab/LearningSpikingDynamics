function [bestPerformingPopulation, bestMatrix, bestPerformance,populations,SUBS] = ALGO( CellMatrix, t1,t2 )
% The function takes in as argument a cell matrix of the size
% neurons x stimulus x repeats. Each row represents one neuron and each
% column one stimulus. Within each stimulus there are a number of trials.
% data must be of equal lengths within the dimensions. This means that
% there must be data for each neuron for each stimulus for each repeat. t1
% and t2 are the start and end times of the recordings. For now the
% recordings need to be all set to same time frame and be equally long.

% Getting dimensions of the array
[Neurons,~,~] = size(CellMatrix);

% At the beginning chose all neurons
population = ones(Neurons,1);

% Calculate the discrimination performance of full population. The full
% population has the best values so far
[bestPerformance,bestMatrix] = discriminationPerformance2( CellMatrix, population,t1,t2 );
populations{1} = population;
SUBS = zeros(Neurons,Neurons);

for index = 2:size(population,1)
    
    % initializing performance so that any result will be better
    HighestSubset = -1;
    
    % Going through all neurons and if the neuron belongs to population try
    % removing it
    for i = 1:Neurons
        if(population(i) == 1)
            possibleNextStep = population;
            possibleNextStep(i) = 0;
            [Perform,NextMatrix] = discriminationPerformance2( CellMatrix, possibleNextStep,t1,t2 );
            SUBS(index-1,i) = Perform;
            
            % If the performance of the subpopulation was better than any other
            % subpopulation before we use that one
            if(Perform > HighestSubset)
                HighestSubset = Perform;
                HighestSubsetPOP = possibleNextStep;
            end
            
            % If it was the best so far we take that as our best performing
            % population
            if Perform > bestPerformance
                bestPerformance = Perform;
                bestPerformingPopulation = possibleNextStep;
                bestMatrix = NextMatrix;
            end
        end
    end
    
    population = HighestSubsetPOP;
    populations{index} = HighestSubsetPOP;
    
    
end
end

