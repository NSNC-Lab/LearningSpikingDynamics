function [population, Matrix,performance,Temperature,Correctness,Cost,TimeSpent,NeuronsInPOP,uniqueSets]  = ALGO_Annealing(CellMatrix, t1,t2,realPOP)
% The function takes in as argument a cell matrix of the size
% neurons x stimulus x repeats. Each row represents one neuron and each
% column one stimulus. Within each stimulus there are a number of trials.
% data must be of equal lengths within the dimensions. This means that
% there must be data for each neuron for each stimulus for each repeat. t1
% and t2 are the start and end times of the recordings. For now the
% recordings need to be all set to same time frame and be equally long.
KnowValues = containers.Map;
KnowMatrices = containers.Map;
BestPerformance = -1;
% Getting dimensions of the array
[Neurons,~,~] = size(CellMatrix);
Trials = 10*Neurons;
k = 0.9;
% At the beginning chose a random iteration.
population = rand(Neurons,1) > 0.5;
% Chosing one that has at least one neuron.
while sum(population) == 0
    population = rand(Neurons,1) > 0.5;
end
% Calculate the discrimination performance of the initial population

currentPopulation = population;
[P,M] = discriminationPerformance2( CellMatrix, currentPopulation,t1,t2 );
BestPerformance  = P;

currentPerformance = P;
currentMatrix = M;
SecondaryStop = zeros(Neurons,Trials);
notReady = 1;

neighbours = 500;

NearbyPoints = zeros(1,neighbours);
Testpopulation = currentPopulation;
Testperformance = currentPerformance;
for i = 1:neighbours
    Testpopulation = AnnealingTransformation(Testpopulation);
    TestperformanceNow = discriminationPerformance2( CellMatrix, Testpopulation,t1,t2 ); 
    
    if TestperformanceNow > BestPerformance
        BestPerformance = TestperformanceNow;
    end
    
    NearbyPoints(i) = TestperformanceNow-Testperformance;
    Testperformance = TestperformanceNow; 
end
deltaC = mean(abs(NearbyPoints));
lnP = log(k);
T0 = -deltaC/lnP;
T = T0;
Round = 1;

TimeSpent = [];
while notReady
    
    %Using a set number of iterations for this test
    for trial = 1:Trials
        time = tic();
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
        % Check if we have encountered this population before
        string = POP2str(trialpopulation);
        if KnowValues.isKey(string)
            % If we have take the population values from storage
            Pnext = KnowValues(string);
            Matrixnext = KnowMatrices(string);
        else
            % create a new distance measure and store them
            [Pnext,Matrixnext] = discriminationPerformance2( CellMatrix, trialpopulation,t1,t2 );
            KnowValues(string) = Pnext;
            KnowMatrices(string) = Matrixnext;
            
            % If the performance is better than the best found so far: store 
            if Pnext > BestPerformance
                BestPerformance = Pnext;
            end
        end
        
        % If it is better or if it exceeds probability of going backwards,
        % accept the new solution
        
        % Acceptance probability
        a = exp((-abs(Pnext-currentPerformance))/T);
        if (Pnext > currentPerformance)|| rand(1) < a
            currentPerformance = Pnext;
            currentPopulation = trialpopulation;
            currentMatrix = Matrixnext;
        end
        SecondaryStop(:,trial) = currentPopulation';
        
        CorrectlyIncluded = sum(currentPopulation(logical(realPOP)))/sum(realPOP);
        CorrectlyExcluded = (sum(~realPOP)-sum(currentPopulation(logical(~realPOP))))/sum(~realPOP);
        Correctness(Round,1) = (CorrectlyIncluded + CorrectlyExcluded)/2;
        Correctness(Round,2) = CorrectlyIncluded;
        Correctness(Round,3) = CorrectlyExcluded;
        Cost(Round) = currentPerformance;
        Temperature(Round) = T;
        Round = Round+1;
        TimeSpent(Round) = toc(time);
        NeuronsInPOP(Round) = sum(currentPopulation);
    end
    TestValues = sum(SecondaryStop,2);
    
    compared = TestValues == currentPopulation.*Trials;
    
    % If the last round was all values the same
    if sum(compared) == size(currentPopulation,1)
        % If the place we got stuck to is not the best found so far:
        % Reset annealing. If it is, we stop.
        if performance < BestPerformance
            T = T0;
        else
            notReady = 0;
        end
    end
    
    %Setting temperature for next round
    T = T*0.99;
    
    population = currentPopulation;
    Matrix = currentMatrix;
    performance = currentPerformance;
    
    
end
uniqueSets = size(KnowValues);
uniqueSets = uniqueSets(1);
end

