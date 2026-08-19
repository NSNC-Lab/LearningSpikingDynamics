/* Object: MEX-file
 * Creator: Eero Satuvuori
 * Date: 18.8.2016
 * Version: 1.0
 * Purpose: This is a MEX function interfacing Matlab variables into C++
 *          back end implementing the method and gating the result back to
 *          Matlab.
 */

#include <mex.h>
#include <matrix.h>
#include <vector>
#include "Spiketrains.h"
#include "DataReader.h"
#include<iostream>

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: mexFind_so_profs
 * Input: SpikeData,threshold,RecordingStartTime, RecordingEndTime, startTime, endTime
 * output: A-SPIKE-Synchronization spike values,
 *
 * Other: -
 */
void mexFunction(int outputN, mxArray* output[], int inputN, const mxArray* input[] )
{
    
    int* valueIn;
    double* valueOut;
    // Defining input and output
#define InN inputN
#define OutN outputN
#define DATA input[0]
#define EDGES input[1]
#define thr input[2]   
#define start input[3]
#define end input[4]
#define time3 input[5]
#define time4 input[6]
#define max_dist input[7]
#define spikes input[8]
#define indices input[9]
    double* THR = mxGetPr(thr);
    double* T1 = mxGetPr(start);
    double* T2 = mxGetPr(end);
    double* T3 = mxGetPr(time3);
    double* T4 = mxGetPr(time4);
    double* T5 = mxGetPr(max_dist);
    const mxArray* Data = DATA;
    double* Edges = mxGetPr(EDGES);
    double* Spikes = mxGetPr(spikes);
    double* SpikeTrainOfASpike = mxGetPr(indices);
    
    DataReader* Reader = new DataReader;
    
    Spiketrains* STs = Reader->ReadSpiketrains(Data);
    
    const int M = mxGetNumberOfElements(spikes);
    const int N = STs->NumberOfSpikeTrains()*(STs->NumberOfSpikeTrains()-1)/2;
    //mexPrintf("Number of spikes M %d, Number of spike trains %d, Number of spike train pairs N %d\n",M,STs->NumberOfSpikeTrains(),N);
    
    std::vector< unsigned int > Counters;
    std::vector< std::vector < double > > Matrix;
    double maxwindow = T2[0]-T1[0];
    //mexPrintf("\nstart: %f, end: %f, time3: %f, time4: %f, maxdistance %f, threshold %f, maxwindow %f\n\n",T1[0],T2[0],T3[0],T4[0],T5[0],THR[0],maxwindow);
    
    //mexPrintf("\n\n");
    //for (unsigned int Train1 = 0; Train1 < STs->NumberOfSpikeTrains(); Train1++)
    //   for (unsigned int spikeIndex = 0; spikeIndex < STs.at(Train1)->Length(); spikeIndex++)
    //       mexPrintf("%i %i: %f\n",Train1+1,spikeIndex+1,STs.at(Train1)->GiveSpikeAtIndex(spikeIndex));
    //mexPrintf("\n");
    
    //for (unsigned int Train1 = 0; Train1 < STs->NumberOfSpikeTrains(); Train1++)
    //    mexPrintf("Edges: %i: %i\n",Train1,(int)Edges[Train1]);
    //mexPrintf("\n\n");

    //for (unsigned int Train1 = 0; Train1 < M; Train1++)
    //    mexPrintf("Spikes: %i: %f\n",Train1,Spikes[Train1]);
    //mexPrintf("\n\n");
    
    //for (unsigned int Train1 = 0; Train1 < M; Train1++)
    //    mexPrintf("SpikeTrainOfASpike: %i: %i\n",Train1,(int)SpikeTrainOfASpike[Train1]);
    //mexPrintf("\n\n");

    // Initializing counters
    for (unsigned int Index = 0; Index < STs->NumberOfSpikeTrains(); Index++)
    {
        Counters.push_back(1);
        //std::cout << Index, Counters.at(Index) << '              ';
    }
    
    for (unsigned int spikeIndex = 0; spikeIndex < M; spikeIndex++)
    {
        unsigned int FromSpikeTrain = SpikeTrainOfASpike[spikeIndex]-1;
        std::vector <double> spikeValues;
        
        for (unsigned int Train1 = 0; Train1 < STs->NumberOfSpikeTrains()-1; Train1++)
        {
            for (unsigned int Train2 = Train1+1; Train2 < STs->NumberOfSpikeTrains(); Train2++)
            {
                unsigned int NthSpike = Counters.at(FromSpikeTrain);
                //mexPrintf("1) Trains %d and %d with spike %u of spike train %d\n",Train1+1,Train2+1,NthSpike,FromSpikeTrain+1);

                if (Train1 == FromSpikeTrain)
                {
                    double Coincidence = STs->AdaptiveCoincidence(NthSpike,Train1,Train2,Edges,T1[0],T2[0],maxwindow,T5[0],THR[0]);
                    spikeValues.push_back(Coincidence);
                    //mexPrintf("1) Coincidence %f for Spike %u of train %d with train %d, \n",Coincidence,NthSpike,Train1+1,Train2+1);
                }
                else if (Train2 == FromSpikeTrain)
                {
                    double Coincidence = STs->AdaptiveCoincidence(NthSpike,Train2,Train1,Edges,T1[0],T2[0],maxwindow,T5[0],THR[0]);
                    spikeValues.push_back(Coincidence);
                    //mexPrintf("2) Coincidence %f for Spike %u of train %d with train %d\n",Coincidence,NthSpike,Train1+1,Train2+1);
                }
                else
                {
                    //mexPrintf("3) Coincidence 0.0, spike %u not involved\n",NthSpike);
                    spikeValues.push_back(0.0);
                }
            }
        }
        
        Counters.at(FromSpikeTrain)++;
        Matrix.push_back(spikeValues);
    }

#define output1 output[0]
    
    // One dimensional cell arrays for the values
    output1 = mxCreateDoubleMatrix(N,M, mxREAL);
    
    valueOut = (double *)mxGetPr(output1);
    //mexPrintf("\n\n\nSS matrix\n\n");
    unsigned int index = 0;
    for (unsigned int m = 0;m < M; m++)
    {
        for (unsigned int n = 0;n < N; n++)
        {
            valueOut[index] = Matrix.at(m).at(n);
            index++;
            //mexPrintf("| %f",Matrix.at(m).at(n));
        }
        //mexPrintf("\n");
    }
    
    
    delete Reader;
    delete STs;
    
}