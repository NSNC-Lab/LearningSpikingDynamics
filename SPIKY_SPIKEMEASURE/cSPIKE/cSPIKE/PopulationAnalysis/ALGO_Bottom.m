function [bestPerformingPopulation, bestMatrix, bestPerformance,populations,SUBS] = ALGO_Bottom( CellMatrix, t1,t2 )
% The function takes in as argument a cell matrix of the size 
% neurons x stimulus x repeats. Each row represents one neuron and each
% column one stimulus. Within each stimulus there are a number of trials.
% data must be of equal lengths within the dimensions. This means that
% there must be data for each neuron for each stimulus for each repeat. t1
% and t2 are the start and end times of the recordings. For now the
% recordings need to be all set to same time frame and be equally long.

% Getting dimensions of the array
[Neurons,~,~] = size(CellMatrix);
SUBS = zeros(Neurons,Neurons);
% At the beginning chose all neurons
population = zeros(Neurons,1);
bestPerformance = -1;
for index = 1:size(population,1)

    % initializing performance so that any result will be better
    HighestSubset = -1;
    
    % Going through all neurons and if the neuron belongs to population try
    % removing it
    for i = 1:Neurons
        if(population(i) == 0)
            possibleNextStep = population;
            possibleNextStep(i) = 1;
            [Perform,NextMatrix] = discriminationPerformance2( CellMatrix, possibleNextStep,t1,t2 );
            SUBS(index,i) = Perform;
            
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
