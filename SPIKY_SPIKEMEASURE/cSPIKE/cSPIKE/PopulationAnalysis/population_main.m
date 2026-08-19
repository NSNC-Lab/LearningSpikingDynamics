%% This is the main file of the population package
% These files are not guaranteed to contain error free general algorithms,
% but rather examples and base for you to write apply the algorithms by
% modifying the code to you own needs.


% First we will plot the simple example of the labelled line example
FIGURE_LL;

% Next we will plot an example of SP where we run all algorithms and find
% the ones where top down and bottom up fail respectively. Running this
% file may take a long time.
FIGURE_TD_BU_fail;

% If you want to generate a test set for your own use you can use the
% neuron response unit function. Basically this represents an SP
% population, but with multiple response units of size 1 you can also get
% LL populations using the same code. Again you can see this in the file 
% FIGURE_TD_BU_fail.

%% Example test set

noiseLL = 0;
noiseSP = 0;
noiseNC = 0;

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

%% Analysis
% The main files you might want to use yourself are the five analysis
% algorithms:

% SP algorithms
ALGO_Brute(CellMatrix,t1,t2) % Brute force
ALGO(CellMatrix,t1,t2) % Greedy top down
ALGO_Bottom(CellMatrix,t1,t2) % Greedy bottom up
ALGO_Annealing(CellMatrix,t1,t2,spPOP) % Simulated annealing

% LL algorithm
[LLPOP,PopulationValues,DiscriminationPerformances ] = ALGO_LL( CellMatrix,t1,t2 ,spPOP,llPOP) 
PopulationValues{1} % Contains the identiry of the neuron with the best performance for each pair
PopulationValues{2} % Contains the performance values of the best neurons for each pair

% You can refer to how they work from file FIGURE_TD_BU_fail. They all take
% the same form of cell array with CELL{neuron,sstimulus,response}, where
% the contents of the cell are the spike times exactly as for the
% SpikeTrainSet (see documentation of cSPIKE). t1 and t2 are start and end
% times respectively for the recording (expecting recordings of uniform
% length) and spPOP and llPOP are variables where you can give the function
% the real population is those are known so they calculate performance
% values. This is mainly for deevelopment and debugging as well as testing
% if one makes modifications.


% For the inputs and putputs of each function please refer to the function
% in guestion.

    