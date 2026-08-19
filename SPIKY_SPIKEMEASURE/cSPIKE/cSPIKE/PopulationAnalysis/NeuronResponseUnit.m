classdef NeuronResponseUnit < handle
    
    
    properties (Access = private)
        nroNeurons_ % number of neurons
        responseRATEs_ % array with rates
        timeCoding_ % bool
        noise_ % 0 <= N < 1
        Jitter_ % in ms how far can a spike deviate
        time_ % length of response window
        NROstimuli_ % how many stimuli are there in total
        timingResponses_ % base timings of time coding neurons
        baseRate_ % base rate for noncoding responses
        codedStimuli_ % bool array. 1 for coded stimulus, 0 for noncoded.
        refractoryPeriod_ %refractory period in [ms]
        noiseGraph_% contains the CDF of noise multiplier
        noiseboundaryX_ % lower boundary of noise multiplier
        noiseboundaryY_ % higher boundary of noise multiplier
        DEBUG_;
    end
    
    methods(Access = private)
        
        %% Gives multiplier x
        function x = giveNoiseMultiplier(obj)
            X = 1-obj.noise_;
            Y = 1/(1-obj.noise_);
            notReady = 1;
            while notReady
                A = rand(1)*(Y-X)+X;
                if rand(1) < (1/(A^3)) 
                    x = A;
                    notReady = 0;
                end
            end
        end
        
        %% Creating response spike trains for time coding 
        function Initialize(obj)
            % first we generate the timing responses if this is a time
            % coding neuron
            if obj.timeCoding_
                for s = 1:obj.NROstimuli_
                    if obj.codedStimuli_(s)
                        SPIKES = 0;
                        while SPIKES ~= round(obj.responseRATEs_(s)*obj.time_)
                            obj.timingResponses_{s} = poissonSpikeTrain(obj.time_,obj.responseRATEs_(s));
                            SPIKES = size(obj.timingResponses_{s},2);
                        end
                        
                    end
                end
                
            end
        end
        %% Jitter all spikes
        function response_J = jitter(obj,response)
            JitterInSeconds = obj.Jitter_/1000;
            randomArray = (rand(1,size(response,2))-0.5)*2;% +- jitter
            response = response + randomArray.*JitterInSeconds;
            %removing spikes that were moved outside of the window
            response = response(logical((response <= obj.time_ ).*( response >= 0)));
            response_J = response; 
        end
        
        %% add refractory period
        function response_R = refractory(obj,response)
            
            RF = obj.refractoryPeriod_/1000;%convert to [ms]
            for spike = 1:(size(response,2)-1)
               if response(spike+1)-response(spike) < RF
                   response(spike+1) = response(spike)+RF;
               end
            end
            %removing spikes that were moved outside of the window
            response = response(logical((response <= obj.time_ ).*( response >= 0)));
            response_R = response; % for now just pass
        end
        
        %% Divide spikes of the response between all neurons of the unit
        % NOTE!! the output is a cell array!
        function responses_N = DivideToPopulation(obj,response)
            randomArray = rand(size(response,2),1);
            bounds = (0:obj.nroNeurons_)/(obj.nroNeurons_);
            responses_N = cell(1,obj.nroNeurons_);
            for n = 1:obj.nroNeurons_
                responses_N{n} = response(logical((randomArray >= bounds(n)).*(randomArray < bounds(n+1))));
            end
        end
        %% Adding noise
        function responseWithNoise = addNoise(obj,response,stimulusIndex)
             
            if obj.noise_== 0
                responseWithNoise = response;
            else
                %% Disrupting timing information
                if(true) % gate variable for unit testing
                    P = obj.noise_;
                    % removing randomly
                    responseWithNoise = response(~(rand(size(response,2),1) < P));
                    ADD = sum(rand(1,size(response,2)) < P);
                    % adding randomly
                    responseWithNoise = unique([responseWithNoise rand(1,ADD)*obj.time_]);
                else
                    responseWithNoise = response;
                end
                %% Disrupting rate information
                if(true) % gate variable for unit testing
                    % Getting the multiplier x
                    x = obj.giveNoiseMultiplier();
                    if x <= 1
                        P = 1-x;
                        % removing randomly
                        responseWithNoise = response(~(rand(1,size(response,2)) < P));
                    else
                        % Since the multiplier is over 1, we will add
                        % spikes with Poisson statistics.
                        
                        NoiseSpikes = poissonSpikeTrain(obj.time_,obj.responseRATEs_(stimulusIndex)*(x-1));
                        % combining the arrays and removing if spikes have
                        % exactly the same time. This is very unlikely but
                        % could theoretically happen. The spike times are
                        % also sorted.
                        responseWithNoise = unique([responseWithNoise NoiseSpikes]);
                    end
                end
                
            end
        end
    end
    methods(Access = public)
        
        
        %{
        This is the constructor of the NeuronResponseUnit
        
        The class is meant to represent a group of neurons that responds to
        a group of stimuli.
        
        # Inputs
        nroNeurons (int >=1):  The number of neurons in this response unit. Use 1 for
        labelled line.
        
        responseRATEs (float array >= 0, -1): the rates at which the neuron responds
        to stimuli numbered as indices of the array. If rate is negative it is
        flagged as noncoding and base rate is used instead.
        
        timeCoding (bool):
            TRUE: the neuron codes in time and thus for each
            response in the array responseRATEs only one instanse is generated.
            FALSE: Each time a stimulus is presented, a new Poisson spike train
            with appropriate rate is generated.
        
        noise ( 0 <= float < 1 ): see paper (N)
        
        jitter (float): strength of jitter noise [ms]. Applied only if
        timeCoding is TRUE.
        
        time (float > 0): the length of the response interval in seconds
        
        refractoryPeriod (float > 0): length of refractory period before a
        neuron is able to fire a new spike in [ms]. Note that the "time"-
        input is in [s]. The conversion is made automatically by the script.
        %}
        
        function obj = NeuronResponseUnit(nroNeurons,responseRATEs,baseRate,timeCoding, noise, jitter, time, refractoryPeriod)
            obj.nroNeurons_ = nroNeurons;  
            obj.responseRATEs_ = responseRATEs;
            obj.timeCoding_ = timeCoding;
            % checking noise level
            if ~(noise >= 0 && noise < 1)
                error('noise value not accepted')
            end
            obj.noise_ = noise;
            obj.Jitter_ = jitter;
            obj.time_ = time;
            obj.NROstimuli_ = max(size(responseRATEs));
            obj.baseRate_ = baseRate;
            obj.refractoryPeriod_ = refractoryPeriod;
            obj.codedStimuli_ = responseRATEs >= 0;
            % Switching the flags of noncoding to base rate
            obj.responseRATEs_(~obj.codedStimuli_) = obj.baseRate_;
            obj.Initialize();
            obj.DEBUG_ = 1;
        end
        
        %{
        This function produces the response of the neuron to given stimulus
        Inputs
        
        MAKE IT ACCEPT MULTIPLE INDICES!!!
        
        %}
        function response = Stimulus(obj,stimulusIndex)

            %% Create the base response of the response unit: new or timed response
            % If the stimulus is time coded and this neuron codes for that
            % stimulus present the same response. Othervice respond with a
            % random spike train. If the neuron is time coding, the
            % response is baseline f and if rate coding it will be
            % appropriate f_c. Both are in obj.responseRATEs_ array.
            if obj.timeCoding_ && obj.codedStimuli_(stimulusIndex)
                Spiketrain = obj.timingResponses_{stimulusIndex};% TESTED
                Spiketrain = obj.jitter(Spiketrain);% TESTED
            else
                Spiketrain = poissonSpikeTrain(obj.time_,obj.responseRATEs_(stimulusIndex));% TESTED
            end
            %% Add noise to the response
            NoisedSpikeTrain = obj.addNoise(Spiketrain,stimulusIndex);
            
            if obj.nroNeurons_ == 1
                % Check for refractory periods of the neuron
                response = obj.refractory(NoisedSpikeTrain);
                
                % Converting empty from 0x1 to [];
                if isempty(response)
                    response = [];
                end
            else
                % Divide the response between the coding units
                responsesCELL = obj.DivideToPopulation(NoisedSpikeTrain);% TESTED
                
                % And last check for refractory periods of each neuron
                for n = 1:obj.nroNeurons_
                    responsesCELL{n} = obj.refractory(responsesCELL{n});% TESTED
                    % Converting empty cell from 0x1 to [];
                    if isempty(responsesCELL{n})
                        responsesCELL{n} = [];
                    end
                end
                response = responsesCELL;
            end
            
        end
        
        %% Set noise
        function setNoise(obj, noise)
            if ~(noise >= 0 && noise < 1)
                error('noise value not accepted')
            end
            obj.noise_ = noise;
        end      
    end
    

    
end

