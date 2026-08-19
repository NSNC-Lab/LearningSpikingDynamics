%% cSPIKE v 1.5 --- 30.6.2023: Eero Satuvuori, Thomas Kreuz
% This is the main part of the cSPIKE. The object offers the basic
% functionality of graphical user interface SPIKY as a command line
% version. The main functions are implemented with MEX files and C++
% backends. Methods are based on the ISI-distance, the SPIKE-distance
% and the SPIKE-synchronization by Thomas Kreuz et al. The class
% contains data and methods for storing spike trains as a single set as
% well as calculating the different distance measures.
%
%   To use SpikeTrainSet you need a C++ compiler for your Matlab. Some help
%   can be found in the following links
%   https://it.mathworks.com/help/matlab/matlab_external/what-you-need-to-build-mex-files.html
%   https://it.mathworks.com/support/compilers/R2016a/
%
%   SpikeTrainSet objects can be created without compiling MEX files, but
%   you can't call any distance measures without the appropriate MEX
%   compiled.
%
%   *** Important ***
%   Before you use cSPIKE for the very first time on a computer you
%   need to compile the MEX and the C++ backends. To do so go to the folder
%   cSPIKEmex and run the script "MEX_compile.m". This may take few minutes.
%
%   Every time you call the MEX files (most functions of the SpikeTrainSet
%   use them) the folder in which they are located needs to be in your
%   Matlab workspace. The "InitializecSPIKE"-script will add it to your path. It
%   needs to be called once every time after starting Matlab and any
%   additional calls are not needed. This is not called by the SpikeTrainSet!
%   *****************
%
%   The SpikeTrainSet implements the following functions
%
%   To create a spike train set object call the constructor:
%
%   spiketrains: A cell array with SpikeTrains{1} containing an
%                array of spike times [spike1 spike2 ...spikeN] for the
%                first spike train and respectively for the other spike
%                trains. The object accepts only spike data aligned as row
%                vectors
%   beginning: The start time of the recording
%   ending: The end time of the recording
%
%   STS = SpikeTrainSet(spiketrains, beginning, ending)
%
%   Then you can call methods for the spike train set:
%
%   time1: Start time of the analysis interval.
%   time2: End time of the analysis interval. If not given or if
%          time1 = time2 instantaneous dissimilarity value is given instead.
%   threshold: Threshold for the adaptive method. If not given the
%              threshold extracted from the data is used instead.
%   Profile: Profile object
%
%   STS.ISIdistance(time1, time2)
%   STS.ISIdistanceMatrix(time1, time2)
%   Profile = STS.ISIdistanceProfile(time1, time2)
%
%   STS.AdaptiveISIdistance(time1, time2, threshold)
%   STS.AdaptiveISIdistanceMatrix(time1, time2, threshold)
%   Profile = STS.AdaptiveISIdistanceProfile(time1, time2, threshold)
%
%   STS.SPIKEdistance(time1, time2)
%   STS.SPIKEdistanceMatrix(obj, time1, time2)
%   Profile = STS.SPIKEdistanceProfile(obj, time1, time2)
%
%   STS.AdaptiveSPIKEdistance(time1, time2, threshold)
%   STS.AdaptiveSPIKEdistanceMatrix(obj, time1, time2,threshold)
%   Profile = STS.AdaptiveSPIKEdistanceProfile(obj, time1, time2, threshold)
%
%   STS.RateIndependentSPIKEdistance(time1, time2)
%   STS.RateIndependentSPIKEdistanceMatrix(obj, time1, time2) % ######
%   Profile = STS.RateIndependentSPIKEdistanceProfile(obj, time1, time2) % ######
%
%   STS.AdaptiveRateIndependentSPIKEdistance(time1, time2, threshold)
%   STS.AdaptiveRateIndependentSPIKEdistanceMatrix(obj, time1, time2, threshold) % ######
%   STS.AdaptiveRateIndependentSPIKEdistanceProfile(obj, time1, time2, threshold) % ######
%
%   STS.SPIKEsynchro(time1, time2, max_dist)
%   STS.AdaptiveSPIKEsynchro(time1, time2, max_dist, threshold)
%
%   SPIKESM: Spike synchronization matrix
%   SPIKEOM: Spike order matrix
%   SPIKETOM: Spike train order matrix % ######
%
%   [SPIKESM,SPIKEOM,SPIKETOM]= STS.SPIKESynchroMatrix(time1, time2, max_dist) % ######
%   [SPIKESM,SPIKEOM,SPIKETOM] = STS.AdaptiveSPIKESynchroMatrix(time1, time2, max_dist, threshold) % ######
%
%   synchro: a cell array identical to spike train set but instead of spike
%            times it contains SPIKE-synchronization values of each spike
%   Sorder: a cell array identical to spike train set but instead of spike
%           times it contains SPIKE-order values of each spike
%   STOrder: a cell array identical to spike train set but instead of spike
%            times it contains spike-train-order values of each spike
%
%   [synchro, Sorder, STOrder, synchProfile, SorderProfile, STorderProfile] = STS.SPIKEsynchroProfile(time1, time2, max_dist)
%   [synchro, Sorder, STOrder, synchProfile, SorderProfile, STorderProfile] = STS.AdaptiveSPIKESynchroProfile(time1, time2, max_dist, threshold)
%
%   [ss_profs,so_profs,sto_profs,SpikeTrainOfASpike] = STS.Ss_So_Sto_profs(obj, time1, time2, max_dist, threshold)
%         Pairwise Spike Order and Spike Train Order profiles
%   [ss_profs,so_profs,sto_profs,SpikeTrainOfASpike] = STS.Adaptive_Ss_So_Sto_profs(obj, time1, time2, max_dist, threshold)
%                          Pairwise Adaptive Spike Order and Spike Train Order profiles
%
%
%   Triggered: varargin: Each time moment is given as a separate parameter. Returns
%             the dissimilarity averaged over all time points.
%
%   STS.TriggeredISImatrix(varargin)
%   STS.TriggeredAdaptiveISImatrix(threshold, varargin)
%   STS.TriggeredRateIndependentISImatrix(varargin) % ######
%   STS.TriggeredAdaptiveRateIndependentISImatrix(threshold, varargin) % ######
%   STS.TriggeredSPIKEmatrix(varargin)
%   STS.TriggeredAdaptiveSPIKEmatrix(threshold, varargin)
%   STS.TriggeredRateIndependentSPIKEmatrix(varargin) % ######
%   STS.TriggeredAdaptiveRateIndependentSPIKEmatrix(threshold, varargin) % ######
%
%   Selective Averaging:
%             varargin: Each pair of inputs defines a new interval over which the
%             distance is defined. A unique union of the intervals is
%             used when defining the intervals over which the average is
%             taken.
%
%   STS.AveragedISIdistanceMatrix(varargin)
%   STS.AveragedAdaptiveISIdistanceMatrix(threshold, varargin)
%   STS.AveragedSPIKEdistanceMatrix(varargin)
%   STS.AveragedAdaptiveSPIKEdistanceMatrix(threshold, varargin)
%   STS.AveragedRateIndependentSPIKEdistanceMatrix(varargin) % ######
%   STS.AveragedAdaptiveRateIndependentSPIKEdistanceMatrix(threshold, varargin) % ######
%
%   Auxiliary functions:
%   STS.SetData(spiketrains, beginning, ending )
%       Replaces the data in the object with new data
%   STS.giveDATA()
%       Gives the a copy of the data array of the object. Data{1} contains
%       the edge corrected data and Data{2} the original spikes.
%   STS.giveTHR()
%       Returns the threshold value obtained from the set
%   [time1,time2] = STS.giveTIMES()
%       Gives the beginning and the end of the recording
%   STS.plotSpikeTrainSet(colour,width)
%       Plots the spike trains to current axis with colour and spike width
%       given. Default colour is black.
%
%
%   STS.SpikeTrainOrderWithSurrogates(numberOfSurrogates)
%       Performs the Spike Train Order analysis which yields the synfire
%       indicator and its significance
%
% BSD license:
%
% Copyright (c) 2016, Eero Satuvuori, extended 2023, Thomas Kreuz
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% * Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
% * Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
% * Neither the name of the author nor the names of its contributors may be
% used to endorse or promote products derived from this software without
% specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
% AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
% TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%% Class for enclosing spiketrains into one class and handling the set.
classdef SpikeTrainSet < handle
    
    
    properties (Access = private)
        
        % Internal datastores.
        DataArray;
        PooledArray;
        timeStart;
        timeEnd;
        threshold;
        max_dist = 10^12;
        timeShift;
        realISIs;
        edges;
        
        % Internal accuracy used by the program. This is the maximum
        % accuracy used. Increasing the accuracy may lead to malfunctions.
        accuracy = 10^-12;
        %ACC= 10^12;
    end
    %% PRIVATE METHODS ****************************************************
    methods (Access = private)
        
        %% Checking data availability
        % Function is given a bool value if the information is crucial for
        % the calling function. In case it is an error is thrown. If it is
        % not, just a bool value of 0 is returned if there is no data yet.
        function bool = HaveData(obj, crucial)
            
            % Returning 1 if there is data and 0 if there is not.
            if isempty(obj.DataArray)
                bool = 0;
                if crucial
                    error('No data available');
                end
            else
                bool = 1;
            end
            
        end
        %% Unique Union
        % Takes a set of intervals as a cell array so that each pair of two
        % values is an interval. The output is a unique union of the
        % intervals.
        function ReturnArray = UniqueUnion(obj, CellArray )
            starts = [];
            ends = [];
            for i = 1:2:length(CellArray)
                starts(end+1) = CellArray{i};
                ends(end+1) = CellArray{i+1};
            end
            
            UniqueArray = 0;
            
            while ~UniqueArray
                moves = 0;
                % Taking each first spike and if it is in an interval, move
                % it to the border of the interval.
                for i = 1:length(starts)
                    for ii = 1:length(starts)
                        if (i ~= ii)
                            if (starts(i) > starts(ii) && starts(i)< ends(ii))
                                starts(i) = starts(ii);
                                moves = moves + 1;
                            end
                        end
                    end
                end
                % Taking each last spike and doing the same.
                for i = 1:length(ends)
                    for ii = 1:length(ends)
                        
                        if (i ~= ii)
                            
                            if (ends(i) > starts(ii) && ends(i) < ends(ii))
                                
                                ends(i) = ends(ii);
                                moves = moves + 1;
                            end
                        end
                        
                    end
                end
                
                % If there were no moves this round, all the spikes are at
                % the edges of their intervals. There may be multiple same
                % intervals.
                if moves  == 0
                    
                    % now the array is unique. No need to continue.
                    UniqueArray = 1;
                    
                    % Taking each interval only once.
                    Ustarts = unique(starts);
                    Uends = unique(ends);
                    
                    % Since the intervals are in time order and unique,
                    % they will be in correct time order after unique
                    % command. The output is formed based on the intervals
                    ReturnArray = cell(0);
                    for i = 1:length(Ustarts)
                        ReturnArray{end+1} = Ustarts(i);
                        ReturnArray{end+1} = Uends(i);
                    end
                end
            end
        end
        
        %% Checking that the DATA given is valid
        % Function checks the data and gives an error
        % if the data is not in a correct form
        function CheckData(obj,Data)
            
            if ~iscell(Data)
                error('Data is not a cell array');
            end
            if length(Data) == 1
                error('One spiketrain');
            end
            for i = 1:length(Data)
                
                if ~ismatrix(Data) || size(Data,2) ~= 1
                    error('Data has incorrectly aligned cell array');
                end
                
                if ~ismatrix(Data{i}) && (size(Data{i},1)||size(Data{i},2))
                    error('Data has too many dimensions');
                end
                
                if size(Data{i},1) ~= 1
                    if  ~isempty(Data{i})
                        error('Data has incorrectly aligned vectors');
                    end
                end
                
                if ~isnumeric(Data{i})
                    error('Data is not all numeric');
                end
            end
            
            
        end
        %% Cutting function
        % This function cuts the data so that spikes outside of boundaries
        % are removed.
        function Data = CutData(obj,Data, START, END)
            
            if isnan(START) || isnan(END)
                beep
                pause
            end
            if any(cellfun(@(x) any(isnan(x)), Data))
                beep
                pause
            end
            
            for i = 1:length(Data)
                inds = not(abs(sign(sign(START-obj.accuracy - Data{i})...
                    + sign(END+obj.accuracy - Data{i}))));
                %inds = not(abs(sign(sign(START - Data{i})...
                %    + sign(END - Data{i}))));
                Data{i} = Data{i}(inds);
            end
            
        end
        %         %%
        %         function Data = RoundData(obj, Data)
        %
        %             %for i = 1:length(Data)
        %             %    Data{i} = round(Data{i}*obj.ACC)/obj.ACC;                  % ############
        %             %end
        %
        %         end
        %% Check input times
        % Checking that times given are in correct order and within boundaries
        function CheckTime(obj,Time1,Time2)
            if Time1  > Time2
                error('Time interval boundaries incorrect');
            end
            
            %if ~((Time1 >= obj.timeStart) && (Time2 <= obj.timeEnd))
            if ~((Time1 >= obj.timeStart) && (Time2 <= obj.timeEnd+10^-10))     % Matlab precision problem #########
                error('time interval not within data bounds');
            end
            
        end
        
        
        %% Correct the DATA Edges
        % Adds edge correction and makes sure that there is only 1 spike
        % at each time instant. Also rearranges the data into time series.
        
        function [Corrected,Edges] = EdgeCorrection(obj,Data,time1,time2)
            
            PooledISIs = [];
            Corrected = cell(1,length(Data))';
            Edges = cell(1,length(Data))';
            for i = 1:length(Data)
                Edges{i}=0;
                % 0 or 1 spikes cases
                if isempty(Data{i}) || length(Data{i}) == 1
                    if isempty(Data{i})
                        first = -(time2-time1)*1.2;                            % ###################
                        last = (time2-time1)*2.2;                              % ###################
                        Corrected{i} = [first obj.timeStart obj.timeEnd last]; % ################### xxx This adds 4 auxiliary spikes, later only 2 are removed xxx
                        %Corrected{i} = [first last];                          % ###################
                        %Corrected{i} = [obj.timeStart obj.timeEnd];           % ###################
                        Corrected{i} = unique(Corrected{i});
                        thisST = obj.timeEnd-obj.timeStart;
                        Edges{i}=0;
                    else
                        fullST = 0; %flag
                        first = obj.timeStart;
                        last = obj.timeEnd;
                        if Data{i} == first
                            first = -(time2-time1)*1.2;                           % ###################
                            thisST = obj.timeEnd-obj.timeStart;
                            fullST = 1;
                            Edges{i}=1;
                        end
                        if Data{i} == last
                            last = (time2-time1)*2.2;                             % ###################
                            thisST = obj.timeEnd-obj.timeStart;
                            fullST = 1;
                            Edges{i}=2;
                        end
                        
                        if ~fullST
                            thisST = [last-Data{i} Data{i}-first];
                        end
                        
                        Corrected{i} = [first Data{i} last];
                        Corrected{i} = unique(Corrected{i});
                    end
                    
                else
                    %The rest
                    thisST = Data{i}(2:end) - Data{i}(1:end-1);
                    if(Data{i}(1) ~= time1)
                        first = Data{i}(1) - max( Data{i}(2)-Data{i}(1), Data{i}(1) - obj.timeStart );
                        thisST = [thisST Data{i}(1)-first];
                    else
                        % In case the first spike is at the border it is
                        % not treated as auxiliary spike. For calculation
                        % the next spike does not matter so we add an
                        % "auxiliary" spike very far from the dataset to
                        % mark that the border is a real one.
                        first = -(time2-time1)*1.2;
                        Edges{i}=Edges{i}+1;
                    end
                    if(Data{i}(end)~= time2)
                        last = Data{i}(end) + max( Data{i}(end)-Data{i}(end-1),obj.timeEnd - Data{i}(end));
                        thisST = [thisST last-Data{i}(end)];
                    else
                        % Same for the other end.
                        last = (time2-time1)*2.2;
                        Edges{i}=Edges{i}+2;
                    end
                    Corrected{i} = [first last Data{i}];
                    
                    % At the end making sure that each dataset has only one
                    % spike at each time instant and that they are in time
                    % order
                    Corrected{i} = unique(Corrected{i});
                end
                PooledISIs = [PooledISIs thisST];
            end
            
            if(isempty (PooledISIs))
                obj.threshold = 0;
            else
                obj.threshold = sqrt(mean(PooledISIs.^2));
            end
            obj.realISIs = PooledISIs;
            
        end
        
        %% Form pooled spiketrain of all spiketrains
        % Pools all spiketrains into one. Uses unique, so that multiple
        % spikes at same times are filtered out.
        function Pooled = Pool(obj , Data)
            
            Pooled = [];
            for i = 1:length(Data)
                Pooled = [Pooled Data{i}];
            end
            Pooled = unique(Pooled);
            
        end
        %% Time shift functions.
        % These functions take care of the difference between "real" times
        % and internal times. The object computes using positive spiketimes
        % and in case the recording begins at negative times, all data is
        % shifted so that the recording begins from 0 internally. For
        % output the times are again shifted back to original.
        function  [time1,time2,Data] = CreateTimeShift(obj,time1,time2,Data)
            
            obj.timeShift = time1;
            time1 = time1-obj.timeShift;
            time2 = time2-obj.timeShift;
            for i = 1:length(Data)
                Data{i}=Data{i}-obj.timeShift;
            end
            
        end
        
        function  [time] = InputTimeShift(obj,time)
            
            time = time-obj.timeShift;
            
        end
        
        
        function  [time] = outputTimeShift(obj,time)
            
            time = time+obj.timeShift;
            
        end
        
        
    end
    %% PUBLIC METHODS *****************************************************
    methods (Access = public)
        %% Constructor
        % Builds a spiketrain set with the spiketrains given and beginning
        % and end times.
        function obj = SpikeTrainSet(spiketrains, beginning, ending)
            obj.timeShift = 0;
            obj.SetData(spiketrains, beginning, ending);
        end
        
        %% Setting data
        % Function sets data to the classes own variables. Calls
        % appropriate checking functions to input. Data is only accepted
        % in a single cell array containing spiketrains. If called to a
        % spiketrain set that has already been set, the user is asked for
        % input if it is to be replaced.
        function SetData(obj, Data, time1, time2)
            Data = Data';
            % checking that the data is in correct form.
            obj.CheckData(Data);
            %Rounding data to 12 digits accuracy
            % Data = obj.RoundData(Data);                                    %% ##############
            
            % time1 = round(time1*obj.ACC)/obj.ACC;                          %% ##############
            % time2 = round(time2*obj.ACC)/obj.ACC;                          %% ##############
            
            % Cutting spikes outside of interval
            Data = obj.CutData(Data, time1, time2);
            % If there was no data, add the data
            if obj.HaveData(0)
                
                %presetting answer variable into not n or y
                answer = 'h';
                % If there was data, ask if the data is to be replaced
                while answer ~= 'y' && answer ~= 'Y' && answer ~= 'n' && answer ~= 'N'
                    answer = input('Do you want to replace the data?(y/n)','s');
                end
                %
                if answer == 'y' || answer ~= 'Y'
                    
                    % Swiping old arrays clean
                    obj.DataArray = [];
                    obj.PooledArray= [];
                    
                    % Adding new values to the array
                    %obj.DataArray{2} = obj.RoundData(Data);                         % ###################
                    obj.DataArray{2} = Data;                         % ###################
                    [time1,time2,Data] = obj.CreateTimeShift(time1,time2,Data);
                    obj.timeStart = time1;
                    obj.timeEnd = time2;
                    [corr,obj.edges]=obj.EdgeCorrection(Data);
                    %obj.DataArray{1} = obj.RoundData(corr);     % ###################
                    obj.DataArray{1} = corr;     % ###################
                    obj.PooledArray = obj.Pool(Data);
                    
                end
            else
                % Setting the arrays for the first time
                %obj.DataArray{2} = obj.RoundData(Data);                         % ################### Copy of original data
                obj.DataArray{2} = Data;                                         % ################### Copy of original data
                [time1,time2,Data] = obj.CreateTimeShift(time1,time2,Data);
                obj.timeStart = time1;
                obj.timeEnd = time2;
                [corr,obj.edges]=obj.EdgeCorrection(Data,time1,time2);
                %obj.DataArray{1} = obj.RoundData(corr);   % ##### Edge-corrected data #####
                obj.DataArray{1} = corr;   % ##### Edge-corrected data #####
                
                obj.PooledArray = obj.Pool(Data);
                
            end
            
        end
        
        %% ISI METHODS ----------------------------------------------------
        % The methods in this section are applying ISI distance measure to
        % the data. The section also contains Adaptive extension.
        
        %% ISI-distance for interval/point
        % Calculates ISI distance for given interval between time1 and
        % time2. time1 < time2. If no input times are given return value
        % for whole data set interval that is defined. If only one time is
        % given or both are the same, return dissimilarity of the
        % spiketrains at that time instant.
        function  ISId = ISIdistance(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            
            ISId = AdaptiveISIdistance(obj, time1, time2, 0);
            
        end
        %% Adaptive ISI-distance for interval/point
        % Calculates SPIKE distance for given interval between time1 and
        % time2. time1 < time2. If no input times are given return value
        % for whole data set interval that is defined. If only one time is
        % given or both are the same, return dissimilarity of the
        % spiketrains at that time instant.
        function  ISId = AdaptiveISIdistance(obj, time1, time2, threshold)
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            elseif nargin == 2
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time1);
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            if(time1 == time2)
                ISId = mexAdaptiveISIDistance(obj.DataArray{1}, threshold, time1);
            else
                ISId = mexAdaptiveISIDistance(obj.DataArray{1}, threshold, time1, time2);
            end
        end
        %% ISI-distance/dissimilarity matrix
        % Returns the distance matrix of the spike trains averaged over
        % time interval defined by time 1 and time 2. If not given,
        % averaged over whole interval. If given only one time or the same
        % one twice, the returned matrix is dissimilarity between all
        % pairs.
        function  ISIdM = ISIdistanceMatrix(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            
            ISIdM = AdaptiveISIdistanceMatrix(obj, time1, time2, 0);
        end
        %% Adaptive ISI-distance/dissimilarity matrix
        % Returns the adaptive distance matrix of the spike trains averaged
        % over time interval defined by time 1 and time 2. If not given,
        % averaged over whole interval. If given only one time or the same
        % one twice, the returned matrix is dissimilarity between all
        % pairs. Threshold can be given as a parameter. If not given the
        % one extracted from the data is used.
        function  ISIdM = AdaptiveISIdistanceMatrix(obj, time1, time2, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            elseif nargin == 2
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time1);
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            ISIdM = mexAdaptiveISIDistanceMatrix(obj.DataArray{1}, threshold ,time1, time2);
            
        end
        %% Triggered ISI-dissimilarity matrix
        % Returns the averaged dissimilarity matrix of the spike trains
        % averaged over time instants defined in inputs. Each instant is
        % given in a vector of variables: [1,2,5,6] being average over time
        % instants 1,2,5 and 6.
        function  ISIdM = TriggeredISImatrix(obj, vector)
            
            if nargin < 2
                error('Not enough input arguments');
            end
            ISIdM = TriggeredAdaptiveISImatrix(obj,0, vector);
        end
        
        %% Triggered Adaptive ISI-dissimilarity matrix
        % Returns the averaged dissimilarity matrix of the spike trains
        % averaged over time instants defined in inputs. Each instant is
        % given as a vector of variables: [1,2,5,6] being average over time
        % instants 1,2,5 and 6. If a negative threshold value is given the
        % data exctracted threshold is used.
        function  ISIdM = TriggeredAdaptiveISImatrix(obj,threshold, vector)
            if threshold < 0
                threshold = obj.threshold;
            end
            if nargin < 3
                error('Not enough input arguments');
            end
            numberOfPoints = max(size(vector));
            obj.HaveData(1);
            ISIdM = 0;
            for i = 1:numberOfPoints
                time = obj.InputTimeShift(vector(i));
                obj.CheckTime(time,time);
                ISIdM = ISIdM + mexAdaptiveISIDistanceMatrix(obj.DataArray{1}, threshold, time, time);
            end
            
            ISIdM = ISIdM/numberOfPoints;
        end
        
        %% Averaged ISI-distance matrix
        % Returns the distance matrix of the spike trains averaged over
        % time intervals defined as pairs of inputs. The inputs must be in
        % pairs. If the intervals overlap, the average is computed over
        % unique union of the intervals. Input is given as a vector
        % of variables: [1,2,5,6] being average over time intervals [1,2] and
        % [5,6].
        function  ISIdM = AveragedISIdistanceMatrix(obj, vector)
            if nargin < 2
                error('Not enough input arguments');
            end
            if mod(vector,2) == 1
                error('Not a paired number of time points');
            end
            
            ISIdM = AveragedAdaptiveISIdistanceMatrix(obj, 0, vector);
            
        end
        
        %% Averaged Adaptive ISI-distance matrix
        % Returns the distance matrix of the spike trains averaged over
        % time intervals defined as pairs of inputs. The inputs must be in
        % pairs. If the intervals overlap, the average is computed over
        % unique union of the intervals. Input is given as a vector
        % of variables: [1,2,5,6] being average over time intervals [1,2] and
        % [5,6]. Threshold can be given as a parameter. If a negative
        % threshold value is given the data exctracted threshold is used.
        function  ISIdM = AveragedAdaptiveISIdistanceMatrix(obj, threshold, vector)
            if threshold < 0
                threshold = obj.threshold;
            end
            if nargin < 3
                error('Not enough input arguments');
            end
            if mod(vector,2) == 1
                error('Not a paired number of time points');
            end
            for i = 1:max(size(vector))
                cellArray{i} = vector(i);
            end
            obj.HaveData(1);
            
            % Forming unique union of input
            Timeslices = obj.UniqueUnion(cellArray);
            time1 = Timeslices{1};
            time2 = Timeslices{2};
            time1 = obj.InputTimeShift(time1);
            time2 = obj.InputTimeShift(time2);
            obj.CheckTime(time1,time2);
            TotalTime = time2-time1;
            ISIdM = (time2-time1)*mexAdaptiveISIDistanceMatrix(obj.DataArray{1}, threshold, time1, time2);
            
            for i = 3:2:length(Timeslices)
                time1 = Timeslices{i};
                time2 = Timeslices{i+1};
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
                obj.CheckTime(time1,time2);
                ISIdM = ISIdM + (time2-time1)*mexAdaptiveISIDistanceMatrix(obj.DataArray{1}, threshold, time1, time2);
                TotalTime = TotalTime + (time2-time1);
            end
            
            % averaging over time. Each matrix is weighted
            ISIdM = ISIdM./TotalTime;
        end
        
        %% ISI-distance profile
        % Forms a profile between times 1 and 2. The output is a Matlab
        % class Profile, which contains the profile and can even perform a
        % simple plotting. See object Profile for more information.
        function  ISIp = ISIdistanceProfile(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            
            ISIp = AdaptiveISIdistanceProfile(obj, time1, time2,0);
        end
        %% Adaptive ISI-distance profile
        % Forms a profile between times 1 and 2. The output is a Matlab
        % class Profile, which contains the profile and can even perform a
        % simple plotting. See object Profile for more information.
        function  ISIp = AdaptiveISIdistanceProfile(obj, time1, time2, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            internalProfile = mexAdaptiveISIDistanceProfile(obj.DataArray{1}, threshold, time1, time2);
            internalProfile(2,:) = obj.outputTimeShift(internalProfile(2,:));
            ISIp = Profile(internalProfile);
            
        end
        %% SPIKE METHODS --------------------------------------------------
        % The methods in this section are applying the SPIKE-distance
        % measure to the data. The section also contains Adaptive and rate
        % independent extensions.
        
        %% SPIKE-distance for interval/point
        % Calculates SPIKE distance for given interval between time1 and
        % time2. time1 < time2. If no input times are given return value
        % for whole data set interval that is defined. If only one time is
        % given or both are the same, return dissimilarity of the
        % spike trains at that time instant.
        function  SPIKEd = SPIKEdistance(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            
            SPIKEd = AdaptiveSPIKEdistance(obj, time1, time2, 0);
        end
        %% Adaptive SPIKE-distance for interval/point
        % Calculates SPIKE distance for given interval between time1 and
        % time2. time1 < time2. If no input times are given return value
        % for whole data set interval that is defined. If only one time is
        % given or both are the same, return dissimilarity of the
        % spiketrains at that time instant.
        function  SPIKEd = AdaptiveSPIKEdistance(obj, time1, time2, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            elseif nargin == 2
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time1);
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            if(time1 == time2)
                SPIKEd = mexAdaptiveSPIKEDistance(obj.DataArray{1}, threshold, time1);
            else
                SPIKEd = mexAdaptiveSPIKEDistance(obj.DataArray{1}, threshold, time1, time2);
            end
        end
        
        %% Rate independent SPIKE distance for interval/point
        % Calculates SPIKE distance for given interval between time1 and
        % time2. time1 < time2. If no input times are given return value
        % for whole data set interval that is defined. If only one time is
        % given or both are the same, returns dissimilarity of the
        % spiketrains at that time instant.
        
        
        function  SPIKEd = RateIndependentSPIKEdistance(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            SPIKEd = AdaptiveRateIndependentSPIKEdistance(obj, time1, time2, 0);
            
        end
        %% Adaptive Rate independent SPIKE distance for interval/point
        % Calculates SPIKE distance for given interval between time1 and
        % time2. time1 < time2. If no input times are given return value
        % for whole data set interval that is defined. If only one time is
        % given or both are the same, returns dissimilarity of the
        % spiketrains at that time instant. This will use
        % the predetermined threshold. If you then want to set threshold
        % yourself give it as third parameter.
        
        function  SPIKEd = AdaptiveRateIndependentSPIKEdistance(obj, time1, time2, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            elseif nargin == 2
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time1);
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            
            obj.HaveData(1);
            obj.CheckTime(time1, time2);
            
            if(time1 == time2)
                SPIKEd = mexAdaptiveRateIndependentSPIKEDistance(obj.DataArray{1}, threshold, time1);
            else
                SPIKEd = mexAdaptiveRateIndependentSPIKEDistance(obj.DataArray{1}, threshold, time1, time2);
            end
        end
        
        %% SPIKE distance/dissimilarity matrix
        % Returns the distance matrix of the spike trains averaged over
        % time interval defined by time 1 and time 2. If not given,
        % averaged over whole interval. If given only one time or the same
        % one twice, the returned matrix is dissimilarity between all
        % pairs.
        function  SPIKEdM = SPIKEdistanceMatrix(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            
            SPIKEdM = AdaptiveSPIKEdistanceMatrix(obj, time1, time2, 0);
        end
        
        %% Adaptive SPIKE distance/dissimilarity matrix
        % Returns the distance matrix of the spike trains averaged over
        % time interval defined by time 1 and time 2. If not given,
        % averaged over whole interval. If given only one time or the same
        % one twice, the returned matrix is dissimilarity between all
        % pairs.
        function  SPIKEdM = AdaptiveSPIKEdistanceMatrix(obj, time1, time2, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            elseif nargin == 2
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time1);
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            SPIKEdM = mexAdaptiveSPIKEDistanceMatrix(obj.DataArray{1}, threshold, time1, time2);
            
        end
        
        %% Rate independent SPIKE distance/dissimilarity matrix               % ######
        % Returns the distance matrix of the spike trains averaged over
        % time interval defined by time 1 and time 2. If not given,
        % averaged over whole interval. If given only one time or the same
        % one twice, the returned matrix is dissimilarity between all
        % pairs.
        function  SPIKEdM = RateIndependentSPIKEdistanceMatrix(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            
            SPIKEdM = AdaptiveRateIndependentSPIKEdistanceMatrix(obj, time1, time2, 0);
        end
        
        %% Adaptive Rate independent SPIKE distance/dissimilarity matrix               % ######
        % Returns the distance matrix of the spike trains averaged over
        % time interval defined by time 1 and time 2. If not given,
        % averaged over whole interval. If given only one time or the same
        % one twice, the returned matrix is dissimilarity between all
        % pairs.
        function  SPIKEdM = AdaptiveRateIndependentSPIKEdistanceMatrix(obj, time1, time2, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            elseif nargin == 2
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time1);
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            SPIKEdM = mexAdaptiveRateIndependentSPIKEDistanceMatrix(obj.DataArray{1}, threshold, time1, time2);
            
        end
        
        %% Triggered SPIKE-dissimilarity matrix
        % Returns the averaged dissimilarity matrix of the spike trains
        % averaged over time instants defined in inputs. Each instant is
        % given as a vector of variables: [1,2,5,6] being average over time
        % instants 1,2,5 and 6.
        function  SPIKEdM = TriggeredSPIKEmatrix(obj, vector)
            
            if nargin < 2
                error('Not enough input arguments');
            end
            SPIKEdM = TriggeredAdaptiveSPIKEmatrix(obj, 0, vector);
            
        end
        
        %% Triggered Adaptive SPIKE-dissimilarity matrix
        % Returns the averaged dissimilarity matrix of the spike trains
        % averaged over time instants defined in inputs. Each instant is
        % given as a vector of variables: [1,2,5,6] being average over time
        % instants 1,2,5 and 6. If a negative threshold value is given the
        % data exctracted threshold is used.
        
        function  SPIKEdM = TriggeredAdaptiveSPIKEmatrix(obj, threshold, vector)
            if threshold < 0
                threshold = obj.threshold;
            end
            if nargin < 3
                error('Not enough input arguments');
            end
            
            numberOfPoints = max(size(vector));
            obj.HaveData(1);
            
            SPIKEdM = 0;
            for i = 1:numberOfPoints
                time = obj.InputTimeShift(vector(i));
                obj.CheckTime(time,time);
                SPIKEdM = SPIKEdM + mexAdaptiveSPIKEDistanceMatrix(obj.DataArray{1}, threshold, time, time);
            end
            
            SPIKEdM = SPIKEdM/numberOfPoints;
        end
        
        %% Triggered Rate independent SPIKE-dissimilarity matrix               % ######
        % Returns the averaged dissimilarity matrix of the spike trains
        % averaged over time instants defined in inputs. Each instant is
        % given as a vector of variables: [1,2,5,6] being average over time
        % instants 1,2,5 and 6.
        function  SPIKEdM = TriggeredRateIndependentSPIKEmatrix(obj, vector)
            
            if nargin < 2
                error('Not enough input arguments');
            end
            SPIKEdM = TriggeredAdaptiveRateIndependentSPIKEmatrix(obj, 0, vector);
            
        end
        
        %% Triggered Adaptive Rate independent SPIKE-dissimilarity matrix               % ######
        % Returns the averaged dissimilarity matrix of the spike trains
        % averaged over time instants defined in inputs. Each instant is
        % given as a vector of variables: [1,2,5,6] being average over time
        % instants 1,2,5 and 6. If a negative threshold value is given the
        % data exctracted threshold is used.
        
        function  SPIKEdM = TriggeredAdaptiveRateIndependentSPIKEmatrix(obj, threshold, vector)
            if threshold < 0
                threshold = obj.threshold;
            end
            if nargin < 3
                error('Not enough input arguments');
            end
            
            numberOfPoints = max(size(vector));
            obj.HaveData(1);
            
            SPIKEdM = 0;
            for i = 1:numberOfPoints
                time = obj.InputTimeShift(vector(i));
                obj.CheckTime(time,time);
                SPIKEdM = SPIKEdM + mexAdaptiveRateIndependentSPIKEDistanceMatrix(obj.DataArray{1}, threshold, time, time);
            end
            
            SPIKEdM = SPIKEdM/numberOfPoints;
        end
        
        %% Averaged SPIKE-distance matrix
        % Returns the distance matrix of the spike trains averaged over
        % time intervals defined as pairs of inputs. The inputs must be in
        % pairs. If the intervals overlap, the average is computed over
        % unique union of the intervals. Input is given as a vector of
        % variables: [1,2,5,6] being average over time intervals [1,2] and
        % [5,6].
        function  SPIKEdM = AveragedSPIKEdistanceMatrix(obj, vector)
            if nargin < 2
                error('Not enough input arguments');
            end
            if mod(vector,2) == 1
                error('Not a paired number of time points');
            end
            SPIKEdM = AveragedAdaptiveSPIKEdistanceMatrix(obj, 0, vector);
        end
        
        %% Averaged Adaptive SPIKE-distance matrix
        % Returns the distance matrix of the spike trains averaged over
        % time intervals defined as pairs of inputs. The inputs must be in
        % pairs. If the intervals overlap, the average is computed over
        % unique union of the intervals. Input is given as a vector of
        % variables: [1,2,5,6] being average over time intervals [1,2] and
        % [5,6]. Threshold can be given as a parameter. If a negative
        % threshold value is given the data exctracted threshold is used.
        
        
        function  SPIKEdM = AveragedAdaptiveSPIKEdistanceMatrix(obj, threshold, vector)
            if threshold < 0
                threshold = obj.threshold;
            end
            if nargin < 3
                error('Not enough input arguments');
            end
            if mod(vector,2) == 1
                error('Not a paired number of time points');
            end
            for i = 1:max(size(vector))
                cellArray{i} = vector(i);
            end
            obj.HaveData(1);
            
            % Forming unique union of input
            Timeslices = obj.UniqueUnion(cellArray);
            
            time1 = Timeslices{1};
            time2 = Timeslices{2};
            time1 = obj.InputTimeShift(time1);
            time2 = obj.InputTimeShift(time2);
            obj.CheckTime(time1,time2);
            TotalTime = time2-time1;
            SPIKEdM = (time2-time1)*mexAdaptiveSPIKEDistanceMatrix(obj.DataArray{1}, threshold, time1, time2);
            
            for i = 3:2:length(Timeslices)
                
                time1 = Timeslices{i};
                time2 = Timeslices{i+1};
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
                obj.CheckTime(time1,time2);
                SPIKEdM = SPIKEdM + (time2-time1)*mexAdaptiveSPIKEDistanceMatrix(obj.DataArray{1}, threshold, time1, time2);
                TotalTime = TotalTime + (time2-time1);
            end
            
            % averaging over time. Each matrix is weighted
            SPIKEdM = SPIKEdM./TotalTime;
        end
        
        %% Averaged Rate independent SPIKE-distance matrix               % ######
        % Returns the distance matrix of the spike trains averaged over
        % time intervals defined as pairs of inputs. The inputs must be in
        % pairs. If the intervals overlap, the average is computed over
        % unique union of the intervals. Input is given as a vector of
        % variables: [1,2,5,6] being average over time intervals [1,2] and
        % [5,6].
        function  SPIKEdM = AveragedRateIndependentSPIKEdistanceMatrix(obj, vector)
            if nargin < 2
                error('Not enough input arguments');
            end
            if mod(vector,2) == 1
                error('Not a paired number of time points');
            end
            SPIKEdM = AveragedAdaptiveRateIndependentSPIKEdistanceMatrix(obj, 0, vector);
        end
        
        %% Averaged Adaptive Rate independent SPIKE-distance matrix               % ######
        % Returns the distance matrix of the spike trains averaged over
        % time intervals defined as pairs of inputs. The inputs must be in
        % pairs. If the intervals overlap, the average is computed over
        % unique union of the intervals. Input is given as a vector of
        % variables: [1,2,5,6] being average over time intervals [1,2] and
        % [5,6]. Threshold can be given as a parameter. If a negative
        % threshold value is given the data exctracted threshold is used.
        
        
        function  SPIKEdM = AveragedAdaptiveRateIndependentSPIKEdistanceMatrix(obj, threshold, vector)
            if threshold < 0
                threshold = obj.threshold;
            end
            if nargin < 3
                error('Not enough input arguments');
            end
            if mod(vector,2) == 1
                error('Not a paired number of time points');
            end
            for i = 1:max(size(vector))
                cellArray{i} = vector(i);
            end
            obj.HaveData(1);
            
            % Forming unique union of input
            Timeslices = obj.UniqueUnion(cellArray);
            
            time1 = Timeslices{1};
            time2 = Timeslices{2};
            time1 = obj.InputTimeShift(time1);
            time2 = obj.InputTimeShift(time2);
            obj.CheckTime(time1,time2);
            TotalTime = time2-time1;
            SPIKEdM = (time2-time1)*mexAdaptiveRateIndependentSPIKEDistanceMatrix(obj.DataArray{1}, threshold, time1, time2);
            
            for i = 3:2:length(Timeslices)
                
                time1 = Timeslices{i};
                time2 = Timeslices{i+1};
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
                obj.CheckTime(time1,time2);
                SPIKEdM = SPIKEdM + (time2-time1)*mexAdaptiveRateIndependentSPIKEDistanceMatrix(obj.DataArray{1}, threshold, time1, time2);
                TotalTime = TotalTime + (time2-time1);
            end
            
            % averaging over time. Each matrix is weighted
            SPIKEdM = SPIKEdM./TotalTime;
        end
        
        %% SPIKE-distance profile
        % Forms a profile between times 1 and 2. The output is a Matlab
        % class Profile, which contains the profile and can even perform a
        % simple plotting.
        function  SPIKEp = SPIKEdistanceProfile(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            
            SPIKEp = AdaptiveSPIKEdistanceProfile(obj, time1, time2, 0);
            
        end
        %% Adaptive SPIKE-distance profile
        % Forms a profile between times 1 and 2. The output is a Matlab
        % class Profile, which contains the profile and can even perform a
        % simple plotting.
        function  SPIKEp = AdaptiveSPIKEdistanceProfile(obj, time1, time2, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            internalProfile = mexAdaptiveSPIKEDistanceProfile(obj.DataArray{1}, threshold, time1, time2);
            internalProfile(2,:) = obj.outputTimeShift(internalProfile(2,:));
            SPIKEp = Profile(internalProfile);
            
        end
        
        %% Rate independent SPIKE-distance profile               % ######
        % Forms a profile between times 1 and 2. The output is a Matlab
        % class Profile, which contains the profile and can even perform a
        % simple plotting.
        function  SPIKEp = RateIndependentSPIKEdistanceProfile(obj, time1, time2)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            
            SPIKEp = AdaptiveRateIndependentSPIKEdistanceProfile(obj, time1, time2, 0);
            
        end
        %% Adaptive Rate independent SPIKE-distance profile               % ######
        % Forms a profile between times 1 and 2. The output is a Matlab
        % class Profile, which contains the profile and can even perform a
        % simple plotting.
        function  SPIKEp = AdaptiveRateIndependentSPIKEdistanceProfile(obj, time1, time2, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            internalProfile = mexAdaptiveRateIndependentSPIKEDistanceProfile(obj.DataArray{1}, threshold, time1, time2);
            internalProfile(2,:) = obj.outputTimeShift(internalProfile(2,:));
            SPIKEp = Profile(internalProfile);
            
        end
        
        
        
        %% SPIKE-Synchronization METHODS ----------------------------------
        % The methods in this section are applying SPIKE-synchronization
        % measure to the data
        
        %% SPIKE-Synchronization
        % This function returns the value of the method from the
        % interval given in the input.
        function  synchro = SPIKEsynchro(obj, time1, time2, max_dist)
            if nargin == 1
                time1 = outputTimeShift(obj, obj.timeStart);
                time2 = outputTimeShift(obj, obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            if nargin < 4
                max_dist = obj.max_dist;                                                % #############
            end
            synchro = AdaptiveSPIKEsynchro(obj, time1, time2, max_dist, 0);
        end
        
        
        %% Adaptive SPIKE-synchronization
        % This function returns the value of the method from the
        % interval given in the input. If no threshold value is given the
        % data exctracted value is used.
        function  synchro = AdaptiveSPIKEsynchro(obj, time1, time2, max_dist, threshold)
            obj.HaveData(1);
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                max_dist = obj.max_dist;
            end
            if nargin < 5
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            synchro = mexAdaptiveSPIKESynchro(obj.DataArray{1}, [obj.edges{:}], threshold/2, obj.timeStart, obj.timeEnd, time1, time2, max_dist);
            
        end
        
        
        %% SPIKE-synchronization profile
        % Forms a profile between times 1 and 2. The output is a set of
        % spike values for SPIKE-synchronization, Spike Order and Spike
        % Train Order, followed by their time profiles
        function  [synchro, Sorder, STOrder, synchProfile, SorderProfile, STorderProfile] = SPIKEsynchroProfile(obj, time1, time2, max_dist)
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            if nargin < 4
                max_dist = obj.max_dist;
            end
            [synchro, Sorder, STOrder, synchProfile, SorderProfile, STorderProfile] = AdaptiveSPIKESynchroProfile(obj, time1, time2, max_dist, 0);
            
        end
        
        
        %% Adaptive SPIKE-synchronization profile
        % Forms a profile between times 1 and 2. The output is a set of
        % spike values for SPIKE-synchronization, Spike Order and Spike
        % Train Order, followed by their time profiles
        function  [synchro, Sorder, STOrder, synchProfile, SorderProfile, STorderProfile] = AdaptiveSPIKESynchroProfile(obj, time1, time2, max_dist, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                max_dist = obj.max_dist;
            end
            if nargin < 5
                threshold = obj.threshold;
            end
            
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            % Change later into a real profile
            
            STs = obj.giveDATA{1};        % #######
            num_trains = length(STs);
            
            [synchro, Sorder, STOrder] = mexAdaptiveSPIKESynchroProfile(obj.DataArray{1}, [obj.edges{:}], ...
                threshold/2, obj.timeStart, obj.timeEnd, time1, time2, max_dist);
            synchroPool = [synchro{:}];
            SorderPool = [Sorder{:}];
            STOrderPool = [STOrder{:}];
            len1=length(synchroPool);
            
            if len1==0
                synchProfile=[];
                SorderProfile=[];
                STorderProfile=[];
                return
            end
            
            % Removing auxiliary spikes
            for i = 1:num_trains
                STs{i} = STs{i}(2:end-1);
            end
            
            spikes = [STs{:}];
            len2=length(spikes);
            if len1~=len2
                empties=intersect(find([obj.edges{:}]==0),...
                    find(cellfun(@(x) x==1, cellfun(@(x) length(x)==2 && x(1)==time1 && x(2)==time2, STs, 'UniformOutput', false))));
                for ec=1:length(empties)
                    STs{empties(ec)}=[];
                end
                spikes = [STs{:}];
                len2=length(spikes);
            end
            
            [OrderedSpikes,order] = sort(spikes);
            
            synchProfile=zeros(2,len2);
            SorderProfile=zeros(2,len2);
            STorderProfile=zeros(2,len2);
            synchProfile(1,: ) = synchroPool(order);  % ####  Matlab precision problem #####               Row118
            synchProfile(2,: ) = OrderedSpikes;
            SorderProfile(1,: ) = SorderPool (order);
            SorderProfile(2,: ) = OrderedSpikes;
            STorderProfile(1,: ) = STOrderPool(order);
            STorderProfile(2,: ) = OrderedSpikes;
        end
        
        %% Pairwise Spike Order and Spike Train Order profiles
        % Forms a profile between times 1 and 2. The output is a set of
        % spike values for SPIKE-synchronization, Spike Order and Spike
        % Train Order, followed by their time profiles
        function  [ss_profs,so_profs,sto_profs,SpikeTrainOfASpike] = Ss_So_Sto_profs(obj, time1, time2, max_dist)
            
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            if nargin < 4
                max_dist = obj.max_dist;
            end
            [ss_profs,so_profs,sto_profs,SpikeTrainOfASpike] = Adaptive_Ss_So_Sto_profs(obj, time1, time2, max_dist, 0);
            
        end
        
        
        %% Pairwise Adaptive Spike Order and Spike Train Order profiles
        function  [ss_profs,so_profs,sto_profs,SpikeTrainOfASpike] = Adaptive_Ss_So_Sto_profs(obj, time1, time2, max_dist, threshold)
            
            if nargin < 3
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 4
                max_dist = obj.max_dist;
            end
            if nargin < 5
                threshold = obj.threshold;
            end
            
            % Creating pooled spike train and info on what spike train a spike belongs to
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            STs = obj.giveDATA{1};
            %STs1=STs
            num_trains = size(STs,1);
            % Removing auxiliary spikes
            for i = 1:num_trains
                STs{i} = STs{i}(2:end-1);
            end
            %STs2=STs
            
            empties=intersect(find([obj.edges{:}]==0),...
                find(cellfun(@(x) x==1, cellfun(@(x) length(x)==2 && x(1)==time1 && x(2)==time2, STs, 'UniformOutput', false))));
            if ~isempty(empties)
                for ec=1:length(empties)
                    STs{empties(ec)}=[];
                end
            end
            %STs3=STs
            
            all_spikes = [STs{:}];
            indices = [];
            for trc = 1:num_trains
                indices = [indices ones(1,length(STs{trc}))*trc];
            end
            [OrderedSpikeTrains,order] = sort(all_spikes);
            SpikeTrainOfASpike = indices(order);
            
            % Creating the auxiliary matrices for spike order and spike train order
            ss_profs = mexFind_ss_profs(obj.DataArray{1}, [obj.edges{:}], threshold/2, obj.timeStart, obj.timeEnd, time1, time2, max_dist, ...
                OrderedSpikeTrains, SpikeTrainOfASpike);
            so_profs = mexFind_so_profs(obj.DataArray{1}, [obj.edges{:}], threshold/2, obj.timeStart, obj.timeEnd, time1, time2, max_dist, ...
                OrderedSpikeTrains, SpikeTrainOfASpike);
            sto_profs = mexFind_sto_profs(obj.DataArray{1}, [obj.edges{:}], threshold/2, obj.timeStart, obj.timeEnd, time1, time2, max_dist, ...
                OrderedSpikeTrains, SpikeTrainOfASpike);
        end
        
        
        
        %% Surrogate test for Spike Train Order
        % This function performs the analysis as described in the paper:
        % "Leaders and followers: Quantifying consistency in spatio-temporal
        % propagation patterns"
        % The input is the number of surrogates used. Default is 19 for 5%
        % confidence. Script SpikeTrainOrderExample shows one way of
        % obtaining the (very close to) Fig. 6 of the paper from the output.
        function  [initialIteration,optimalIteration,synf,so_profs,sto_profs,SpikeTrainOfASpike] = ...
                SpikeTrainOrderWithSurrogates(obj, numberOfSurrogates, time1, time2, max_dist)    % ###### SpikeTrainOfASpike ######
            
            if nargin < 2
                numberOfSurrogates = 19;
            end
            if nargin < 4
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin < 5
                max_dist = obj.max_dist;
            end
            %if nargin < 5
                thresh = 0;
                %threshold = obj.giveTHR   % ##################################### adaptive with data-dependent threshold
            %end
            
            % Creating pooled spike train and info on what spike train a spike belongs to
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            % STs = obj.giveDATA{1};            ########
            STs = obj.giveDATA{1};
            num_trains = size(STs,1);
            for i = 1:num_trains
                STs{i} = STs{i}(2:end-1);
            end
            empties=intersect(find([obj.edges{:}]==0),...
                find(cellfun(@(x) x==1, cellfun(@(x) length(x)==2 && x(1)==time1 && x(2)==time2, STs, 'UniformOutput', false))));
            if ~isempty(empties)
                for ec=1:length(empties)
                    STs{empties(ec)}=[];
                end
            end
            spikes = [STs{:}];
            
            indices = [];
            for trc = 1:num_trains
                indices = [indices ones(1,length(STs{trc}))*trc];
            end
            [OrderedSpikeTrains,order] = sort(spikes);
            SpikeTrainOfASpike = indices(order);
            
            % Creating the auxiliary matrices for spike order and spike train order
            so_profs = mexFind_so_profs(obj.DataArray{1}, [obj.edges{:}], thresh/2, obj.timeStart, obj.timeEnd, time1, time2, max_dist, ...
                OrderedSpikeTrains, SpikeTrainOfASpike);
            sto_profs = mexFind_sto_profs(obj.DataArray{1}, [obj.edges{:}], thresh/2, obj.timeStart, obj.timeEnd, time1, time2, max_dist, ...
                OrderedSpikeTrains, SpikeTrainOfASpike);
            
            % Calculating synfire indicator for original
            [Synch,SO,STO] = obj.AdaptiveSPIKESynchroProfile(time1, time2, max_dist, thresh);
            SYNCsum = 0;
            STOsum = 0;
            num_all_spikes  = 0;
            for i = 1:num_trains
                num_all_spikes = num_all_spikes + size(STs{i},2);
                STOsum = STOsum + sum(STO{i});
                SYNCsum = SYNCsum + sum(Synch{i});
            end
            ini = STOsum/num_all_spikes;
            
            % Calculating surrogates
            num_surros = numberOfSurrogates;
            if numberOfSurrogates > 0
                %disp('Start surrogates')
                
                all_trains=f_all_trains(STs');
                %synfNotNormalized = f_generate_spike_order_surros_plots(num_surros,sto_profs,all_trains,ini)
                
               synfNotNormalized = f_generate_spike_order_surros(sto_profs,num_surros);
                %disp('Surrogates ready')
            else
                synfNotNormalized = [];
            end
            
            
            % Normalizing the synfire indicators of the surrogates
            synf = synfNotNormalized*2/(num_trains-1)/num_all_spikes;
            
            % Calculating the optimal order by pairwise matrix
            matr_entries=sum(sto_profs,2)/2;
            matr = tril(ones(num_trains),-1);
            matr(~~matr) = matr_entries';
            matr=matr'-matr;
            %disp('Start simulated annealing')
            [st_indy_simann,~,~]=Spike_train_order_sim_ann_MEX(matr);
            %disp('Simulated annealing done')
            
            % Calculating the pairwise matrix for optimal order
            OPTOrderSpikes=cell(1,num_trains);
            OPTOrderWithAUX=cell(1,num_trains);
            OPTEdges=zeros(1,num_trains);
            for i = 1:num_trains
                OPTOrderSpikes{i} = STs{st_indy_simann(i)};
                OPTOrderWithAUX{i} = obj.DataArray{1}{st_indy_simann(i)};
                OPTEdges(i) = obj.edges{st_indy_simann(i)};            % [obj.edges{:}]
            end
            STs_opt = OPTOrderSpikes;
            opt_indices = [];
            opt_spikes = [STs_opt{:}];
            for ST = 1:num_trains
                opt_indices = [opt_indices ones(size(STs_opt{ST},2),1)'*ST];
            end
            [opt_OrderedSpikeTrains,opt_order] = sort(opt_spikes);
            opt_SpikeTrainOfASpike = opt_indices(opt_order);
            %disp('Start mexFind_sto_profs')
            sto_profs_opt = mexFind_sto_profs(OPTOrderWithAUX, OPTEdges, thresh/2, obj.timeStart, obj.timeEnd, time1, time2, max_dist, ...
                opt_OrderedSpikeTrains, opt_SpikeTrainOfASpike);
            %disp('EndmexFind_sto_profs')
            
            matr_entries_opt=sum(sto_profs_opt,2)/2;
            matrOpt = tril(ones(num_trains),-1);
            matrOpt(~~matrOpt) = matr_entries_opt';
            matrOpt=matrOpt'-matrOpt;
            
            
            % Synfire indicator of the optimal order
            STS = SpikeTrainSet(OPTOrderSpikes,obj.timeStart,obj.timeEnd);
            [~,SOopt,STOopt] = STS.AdaptiveSPIKESynchroProfile(time1, time2, max_dist, thresh);
            STOoptsum = 0;
            for i = 1:num_trains
                STOoptsum = STOoptsum + sum(STOopt{i});
                %sum(STOopt{i});
            end
            
            % Calculate the synfire indicator values
            fin = STOoptsum/num_all_spikes;
            SYNCvalue = SYNCsum/num_all_spikes;
            
            % Pooling STO values
            initPool = [STO{:}];
            %optPool_old = [];
            optPool = [STOopt{:}];
            SyncPool = [Synch{:}];
            %for i = 1:num_trains                
                %Concatenate optimal in original order for profile
                %optPool_old = [optPool_old STOopt{st_indy_simann(i)}];          % wrong, STOopt is already sorted according to the optimized order
            %end
            
            %bbb=sum(cellfun('length',STs))             % TTTTTTTTTTT
            profile(1,: ) = initPool(order);       % ##########
            profile(2,: ) = OrderedSpikeTrains;
            % OPTprofile(1,: ) = optPool_old(order);       % wrong
            OPTprofile(1,: ) = optPool(opt_order);       % TTTTTTTTTT
            OPTprofile(2,: ) = OrderedSpikeTrains;
            SpikeSynchroProfile(1,: ) = SyncPool(order);
            SpikeSynchroProfile(2,: ) = OrderedSpikeTrains;
            
            [~,rank]=sort(st_indy_simann);
            
            % Construct output
            initialIteration = Iteration;
            initialIteration.Name = 'Initial';
            initialIteration.Order = 1:num_trains;
            initialIteration.SpikeTrains = STs';
            initialIteration.SynfireIndicatorF = ini;
            initialIteration.SynchronizationValueC = SYNCvalue;
            initialIteration.SynchronizationProfile = SpikeSynchroProfile;
            initialIteration.SpikeTrainOrderSpikeValues = STO;
            initialIteration.SpikeOrderSpikeValues = SO;
            initialIteration.TimeProfileE = profile;
            initialIteration.PairwiseMatrixE = matr;
            initialIteration.SpikeTrainOfASpike = SpikeTrainOfASpike;       % TTTTTTTTTT
            
            optimalIteration = Iteration;
            optimalIteration.Name = 'optimal';
            optimalIteration.Order = st_indy_simann;
            optimalIteration.SpikeTrains = OPTOrderSpikes;
            optimalIteration.SynfireIndicatorF = fin;
            optimalIteration.SynchronizationValueC = SYNCvalue;
            optimalIteration.SynchronizationProfile = SpikeSynchroProfile;
            optimalIteration.SpikeTrainOrderSpikeValues = STOopt;
            optimalIteration.SpikeOrderSpikeValues = SOopt;
            optimalIteration.TimeProfileE = OPTprofile;
            optimalIteration.PairwiseMatrixE = matrOpt;
            optimalIteration.SpikeTrainOfASpike = rank(SpikeTrainOfASpike);       % TTTTTTTTTT
        end
        
        
        
        %% SPIKE-synchronization matrix
        % Returns the distance matrix of the spike trains averaged over
        % time interval defined by time 1 and time 2. If not given,
        % averaged over whole interval. If given only one time or the same
        % one twice, the returned matrix is dissimilarity between all
        % pairs.
        
        function  [SPIKESM,SPIKEOM,normSPIKEOM] = SPIKESynchroMatrix(obj, time1, time2, max_dist)
            
            if nargin == 1
                time1 = outputTimeShift(obj,obj.timeStart);
                time2 = outputTimeShift(obj,obj.timeEnd);
            end
            if nargin == 2
                time2 = time1;
            end
            if nargin < 4
                max_dist = obj.max_dist;
            end
            
            [SPIKESM,SPIKEOM,normSPIKEOM] = AdaptiveSPIKESynchroMatrix(obj, time1, time2, max_dist, 0);
            
        end
        
        %% Adaptive SPIKE-synchronization matrix
        % Returns the distance matrix of the spike trains averaged over
        % time interval defined by time 1 and time 2. If not given,
        % averaged over whole interval. If given only one time or the same
        % one twice, the returned matrix is dissimilarity between all
        % pairs. If no threshold value is given thedata exctracted value
        % is used.
        function  [SPIKESM,SPIKEOM,SPIKETOM] = AdaptiveSPIKESynchroMatrix(obj, time1, time2, max_dist, threshold)
            
            if nargin == 1
                time1 = obj.timeStart;
                time2 = obj.timeEnd;
            else
                time1 = obj.InputTimeShift(time1);
                time2 = obj.InputTimeShift(time2);
            end
            if nargin == 2
                time2 = time1;
            end
            if nargin < 4
                max_dist = obj.max_dist;
            end
            if nargin < 5
                threshold = obj.threshold;
            end
            obj.HaveData(1);
            obj.CheckTime(time1,time2);
            
            num_trains = length(obj.DataArray{1});
            SPIKESM = ones(num_trains);
            SPIKEOM = zeros(num_trains);
            SPIKETOM = zeros(num_trains);
            %normSPIKEOM = zeros(num_trains);
            for trc = 1:num_trains-1
                for trc2 = trc+1:num_trains
                    pair{1} = obj.DataArray{2}{trc};
                    pair{2} = obj.DataArray{2}{trc2};
                    num_pair_spikes = length(obj.DataArray{1}{trc})+length(obj.DataArray{1}{trc2})-4;
                    if num_pair_spikes > 0
                        STS = SpikeTrainSet(pair,obj.timeStart,obj.timeEnd);
                        SPIKESM(trc,trc2) = STS.AdaptiveSPIKEsynchro(time1, time2, max_dist, threshold);
                        SPIKESM(trc2,trc) = SPIKESM(trc,trc2);
                        
                        %[~,so_profile,sto_profile] = STS.AdaptiveSPIKESynchroProfile(time1, time2, max_dist, threshold);
                        [~,~,sto_profile] = STS.AdaptiveSPIKESynchroProfile(time1, time2, max_dist, threshold);
                        
                        %SPIKEOM(trc,trc2) = (sum(so_profile{1})+sum(so_profile{2}))/num_pair_spikes;
                        %SPIKEOM(trc2,trc) = (sum(so_profile{1})+sum(so_profile{2}))/num_pair_spikes;                        
                        SPIKETOM(trc,trc2) = (sum(sto_profile{1})+sum(sto_profile{2}))/num_pair_spikes;
                        SPIKETOM(trc2,trc) = -SPIKETOM(trc,trc2); % ### Spike Train Order is asymmetric (makes sense: when relative order changes, sign changes) ####
                    end
                end
            end
            
        end
        
        
        %% Auxiliary functions --------------------------------------------
        
        %% Get function
        % Gives a copy of the spike data contained in the spike train set
        function  DATA = giveDATA(obj)
            obj.HaveData(1);
            DATA = obj.DataArray;
        end
        %% Threshold value extracted from the data
        % returns the threshold value extracted from this dataset
        function  [threshold, Mean] = giveTHR(obj)
            obj.HaveData(1);
            threshold = obj.threshold;
            Mean = mean(obj.realISIs);
        end
        
        %% Start and end of the data set
        % returns an array with start and end times of the recording.
        function  [time1,time2] = giveTIMES(obj)
            obj.HaveData(1);
            time1 = obj.outputTimeShift(obj.timeStart);
            time2 = obj.outputTimeShift(obj.timeEnd);
        end
        
        %% Plotting function
        % Plots the spike trains to current axis
        function  plotSpikeTrainSet(obj,colour,width)
            obj.HaveData(1);
            if nargin == 1
                colour = 'k';
            end
            if nargin < 3
                width = 1;
            end
            
            beginning = obj.outputTimeShift(obj.timeStart);
            ending = obj.outputTimeShift(obj.timeEnd);
            hold on;
            for i = 1:length(obj.DataArray{2})
                
                SpikeTrain = obj.DataArray{2}{i};
                plotSpikeTrain(SpikeTrain,0.8,(length(obj.DataArray{2})+1-i)-0.5,colour,width)
                
            end
            axis([beginning,ending,0,length(obj.DataArray{2})])
            set(gca,'ytick',[]);
        end
        
    end
    
end

