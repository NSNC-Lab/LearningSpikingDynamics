%% Compile file for MEX functions.
% The file compiles the MEXes and the C++ classes mplementing the MEXes.

% First run as it is, if you get an error message that involves the two
% variable types "char16_t"  and "unsigned short" please set problem to 1
% and try again. There are some known compiler incompatabilities but one
% of these two variants should usually work.

problem=0;

if problem==0
    
    %% Distance measure functions
    mex -compatibleArrayDims mexAdaptiveISIDistance.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims mexAdaptiveSPIKEDistance.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims mexAdaptiveRateIndependentSPIKEDistance.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp

    mex -compatibleArrayDims mexAdaptiveSPIKESynchro.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    %% Matrix functions
    mex -compatibleArrayDims mexAdaptiveISIDistanceMatrix.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims mexAdaptiveSPIKEDistanceMatrix.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims mexAdaptiveRateIndependentSPIKEDistanceMatrix.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    %% Profile functions
    mex -compatibleArrayDims mexAdaptiveISIDistanceProfile.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims mexAdaptiveSPIKEDistanceProfile.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims mexAdaptiveRateIndependentSPIKEDistanceProfile.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims mexAdaptiveSPIKESynchroProfile.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    %% Surrogate auxiliary functions
    
    mex -compatibleArrayDims mexFind_ss_profs.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp

    mex -compatibleArrayDims mexFind_so_profs.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims mexFind_sto_profs.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -compatibleArrayDims Spike_train_order_sim_ann_MEX.c
    
    mex -compatibleArrayDims Spike_train_order_surro_MEX.c

else
    
    %% Distance measure functions
    mex -Dchar16_t=uint16_T mexAdaptiveISIDistance.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T mexAdaptiveSPIKEDistance.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T mexAdaptiveRateIndependentSPIKEDistance.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp

    mex -Dchar16_t=uint16_T mexAdaptiveSPIKESynchro.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
        
    %% Matrix functions
    mex -Dchar16_t=uint16_T mexAdaptiveISIDistanceMatrix.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T mexAdaptiveSPIKEDistanceMatrix.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T mexAdaptiveRateIndependentSPIKEDistanceMatrix.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    %% Profile functions
    mex -Dchar16_t=uint16_T mexAdaptiveISIDistanceProfile.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T mexAdaptiveSPIKEDistanceProfile.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T mexAdaptiveRateIndependentSPIKEDistanceProfile.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T mexAdaptiveSPIKESynchroProfile.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    %% Surrogate auxiliary functions
    
    mex -Dchar16_t=uint16_T mexFind_ss_profs.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp

    mex -Dchar16_t=uint16_T mexFind_so_profs.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T mexFind_sto_profs.cpp Spiketrain.cpp DataReader.cpp Spiketrains.cpp Pair.cpp ISIProfile.cpp SPIKEProfile.cpp
    
    mex -Dchar16_t=uint16_T Spike_train_order_sim_ann_MEX.c
    
    mex -Dchar16_t=uint16_T Spike_train_order_surro_MEX.c
    
end

