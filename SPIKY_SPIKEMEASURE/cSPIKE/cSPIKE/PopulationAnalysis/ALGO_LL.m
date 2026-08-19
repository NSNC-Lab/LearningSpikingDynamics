function [ POP,Matrix,DiscriminationPerformances ] = ALGO_LL( CellMatrix,t1,t2 ,spPOP,llPOP)
neurons = size(CellMatrix,1);
stimuli = size(CellMatrix,2);
repeats = size(CellMatrix,3);
DiscriminationPerformances = zeros(neurons,1);
for neuron = 1:neurons
    index = 1;
    for stimulus = 1:stimuli
        for repeat = 1:repeats
            STs(index) = CellMatrix(neuron,stimulus,repeat);
            index = index+1;
        end
    end
    STS = SpikeTrainSet(STs,t1,t2);
    Matrix = STS.SPIKEdistanceMatrix;
    Intra = 0;
    IntraCount = 0;
    Inter = 0;
    InterCount = 0;
    
    for Sy = 1:stimuli
        for Ry = 1:repeats
            for Sx = 1:stimuli
                for Rx = 1:repeats
                    % Calculating the index so that we only calculate each
                    % pair once
                    Xindex = (Sx-1)*repeats + Rx;
                    Yindex = (Sy-1)*repeats + Ry;
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
    
    DiscriminationPerformances(neuron) = Inter/InterCount - Intra/IntraCount;
end
%% Assign POP later when we decide how to assign it.
POP = 0;



end

