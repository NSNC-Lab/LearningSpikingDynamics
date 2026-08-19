function [performance,Matrix] = discriminationPerformance2( CellMatrix, population,t1,t2 )
% The function takes in as argument a cell matrix of the size
% neurons x stimulus x repeats. Each row represents one neuron and each
% column one stimulus. Within each stimulus there are a number of trials.
% data must be of equal lengths within the dimensions. This means that
% there must be data for each neuron for each stimulus for each repeat. The
% other argument indicates which neurons are in the subpopulation. t1
% and t2 are the start and end times of the recordings. For now the
% recordings need to be all set to same time frame and be equally long.


[Neurons,Stimuli,Repeats] = size(CellMatrix);
CombinedSpiketrains = cell(1,Stimuli*Repeats);
% An array for summing up all spiketrains
Addindex = 1;
for Stimulus = 1:Stimuli
    for Repeat = 1:Repeats
        concatenated = [];
        for Neuron = 1:Neurons

            % If the neuron belongs to the subpopulation we add its spiketrain to
            % the total.
            
            if(population(Neuron) == 1)
                concatenated = cat(2,concatenated,CellMatrix{Neuron,Stimulus,Repeat});
            end
            
        end
        CombinedSpiketrains{Addindex} = unique(concatenated);
        Addindex = Addindex+1;
    end
end

% After forming the combined spiketrains we compute the distance matrix of
% all the pairs.
STS = SpikeTrainSet(CombinedSpiketrains,t1,t2);
Matrix = STS.SPIKEdistanceMatrix();

% Of all the pairs we count the intra and inter distances and calculate the
% discrimination performance based on them.

Intra = 0;
IntraCount = 0;
Inter = 0;
InterCount = 0;

for Sy = 1:Stimuli
    for Ry = 1:Repeats
        for Sx = 1:Stimuli
            for Rx = 1:Repeats     
                % Calculating the index so that we only calculate each
                % pair once
                Xindex = (Sx-1)*Repeats + Rx;
                Yindex = (Sy-1)*Repeats + Ry;
                %Only calculating upper half of the matrix. Not taking the
                %equal index since we do not want to compare with itself.
                if Xindex > Yindex
                    % If both are sets for the same stimulus it is intra
                    % distance. If different it is inter.
                    if Sy == Sx                        
                        Intra = Intra + Matrix(Xindex, Yindex);
                        IntraCount = IntraCount+1;
                    else                        
                        Inter = Inter + Matrix(Xindex, Yindex);
                        InterCount = InterCount+1; 
                    end
                end
                
            end
        end
    end
end

performance = Inter/InterCount - Intra/IntraCount;

end

