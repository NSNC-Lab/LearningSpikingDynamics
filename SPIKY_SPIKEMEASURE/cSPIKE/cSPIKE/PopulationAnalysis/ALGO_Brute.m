function [bestPerformingPopulation, Matrix, bestPerformance] = ALGO_Brute( CellMatrix, t1,t2 )
% The function takes in as argument a cell matrix of the size 
% neurons x stimulus x repeats. Each row represents one neuron and each
% column one stimulus. Within each stimulus there are a number of trials.
% data must be of equal lengths within the dimensions. This means that
% there must be data for each neuron for each stimulus for each repeat. t1
% and t2 are the start and end times of the recordings. For now the
% recordings need to be all set to same time frame and be equally long.

% Getting dimensions of the array
[Neurons,~,~] = size(CellMatrix);

numberOfCombinations = 2^Neurons;

Table = zeros(Neurons,numberOfCombinations);
for i = 1:Neurons
   vector = ones(1,2^(i));
   vector(1:int32(size(vector,2)/2)) = 0;
   Table(int32(i),:) = repmat(int32(vector),[1,int32(numberOfCombinations/(2^(i)))]); 
end
bestPerformance = -1;

for POP = 1:numberOfCombinations
    
    population = Table(:,POP)';
    
    [Pnext,Matrixnext] = discriminationPerformance2( CellMatrix, population,t1,t2 );
 
    if Pnext > bestPerformance
        bestPerformance = Pnext;
        bestPerformingPopulation = population';
        Matrix = Matrixnext;
    end 
end

end

