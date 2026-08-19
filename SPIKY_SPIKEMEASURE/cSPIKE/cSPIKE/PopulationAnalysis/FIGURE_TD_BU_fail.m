
close all;
InitializecSPIKE;

for Figure = 1:2
    clearvars -except Figure;
    if Figure == 1
        noiseLL = 0;
        noiseSP = 0;
        noiseNC = 0;
    else
        noiseLL = 0.5;
        noiseSP = 0;
        noiseNC = 0;
    end
    
    NROrepeats = 5;
    NROStimuli = 4;%
    NRO_LL_Neurons = 3;
    NRO_SP_Neurons = 4;
    NRO_NC_Neurons = 3;
    rate = 20;% Hz
    nroNeurons = 1;%
    responseRATEs = ones(NROStimuli,1)*rate;
    timeCoding = 1;
    baseRate = rate;
    
    jitter = 5;%[ms]
    time = 1;%[s]
    refractoryPeriod = 2;%[ms]
    
    %% creating neurons
    
    for i = 1:NRO_LL_Neurons
        LL{i} = NeuronResponseUnit(1,responseRATEs,baseRate,timeCoding, noiseLL, jitter, time, refractoryPeriod);
    end
    
    responseRATEs = ones(NROStimuli,1)*rate*NRO_SP_Neurons;
    RU1 = NeuronResponseUnit(NRO_SP_Neurons,responseRATEs,baseRate,timeCoding, noiseSP, jitter, time, refractoryPeriod);
    
    responseRATEs = ones(NROStimuli,1)*-1;
    for i = 1:NRO_NC_Neurons
        NC{i} = NeuronResponseUnit(1,responseRATEs,baseRate,timeCoding, noiseNC, jitter, time, refractoryPeriod);
    end
    %% Responses
    % SP coding responses
    for s = 1: NROStimuli
        for r = 1:NROrepeats
            Responses = RU1.Stimulus(s);
            for n = 1:NRO_SP_Neurons
                CELL{n,s,r} = Responses{n};
            end
            
        end
    end
    
    % LL neurons responses
    for n = 1:NRO_LL_Neurons
        for s = 1: NROStimuli
            for r = 1:NROrepeats
                CELL{n+NRO_SP_Neurons,s,r} = LL{n}.Stimulus(s);
            end
        end
    end
    
    % NC responses
    for n = 1:NRO_NC_Neurons
        for s = 1: NROStimuli
            for r = 1:NROrepeats
                CELL{n+NRO_LL_Neurons+NRO_SP_Neurons,s,r} = NC{n}.Stimulus(s);
            end
        end
    end
    
    CellMatrix = CELL;
    t1 = 0;
    t2 = time;
    llPOP = [ 0 0 0 0 1 1 1 0 0 0 ];
    spPOP = [ 1 1 1 1 0 0 0 0 0 0 ];
    SPcoding = NRO_SP_Neurons;
    LLcoding = NRO_LL_Neurons;
    Neurons = NRO_SP_Neurons + NRO_LL_Neurons + NRO_NC_Neurons;
    
    %% Find SP discrimination performances
    
    % This is simply SP discrimination performance for single neurons
    [LLPOP,~,DiscriminationPerformances ] = ALGO_LL( CellMatrix,t1,t2 ,spPOP,llPOP);
    % Annealing
    [spAPOP,~,spAperformance] = ALGO_Annealing(CellMatrix,t1,t2,spPOP);
    % Greedy
    %top down
    [spGPOP,~,spGperformance,TDpopulations,TDSUBS] = ALGO(CellMatrix,t1,t2);
    %bottom up
    [spGBPOP,~,spGBperformance,BUpopulations,BUSUBS] = ALGO_Bottom(CellMatrix,t1,t2);
    % Brute
    [spBPOP,~,spBperformance] = ALGO_Brute(CellMatrix,t1,t2);
    
    
    %% Draw graphs
    DrawLLvsSP;
    DrawGreedyGraphs;
end
