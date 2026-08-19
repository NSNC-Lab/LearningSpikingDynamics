/* Object: Spiketrains
 * Creator: Eero Satuvuori
 * Date: 18.8.2016
 * Version: 1.0
 * Purpose: 
 */

#include "Spiketrains.h"
#include "Spiketrain.h"
#include <cstdlib>
#include <mex.h>
#include <sstream>
#include <string>
#include <cmath>
#include "ISIProfile.h"
#include "SPIKEProfile.h"

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: Constructor
 * Input: -
 * output: -
 *
 * Other: -
 */ 
Spiketrains::Spiketrains()
{
    
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: Destructor
 * Input: -
 * output: -
 *
 * Other: deleting all spike trains associated with the spike trains object
 */
Spiketrains::~Spiketrains()
{
    for (unsigned int i = 0; i < STs.size(); i++)
    {
        delete STs.at(i);
    }
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: Second constructor
 * Input: vector of spiketrain pointers
 * output: -
 *
 * Other: Spike trains are only stored as pointers. Deleting the spike 
 *        trains is the responsibility of this object.
 */
Spiketrains::Spiketrains(std::vector<Spiketrain*> STvector):STs(STvector){}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: Add a spike train 
 * Input: spike train pointer
 * output: -
 *
 * Other: Spike trains are only stored as pointers. Deleting the spike 
 *        trains is the responsibility of this object.
 */
void Spiketrains::AddSpiketrain(Spiketrain* ST)
{

    STs.push_back(ST);

}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: get pointer to spike train at index
 * Input: index
 * output: spike train pointer
 *
 * Other: no checks are made if index exists. 
 */
Spiketrain* Spiketrains::GetST(int index)
{
    return STs.at(index);
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-ISI-distance
 * Input: time at, threshold
 * output: distance
 *
 * Other: A-ISI-distance of the whole set. Forms pairs and computes average
 */
double Spiketrains::AdaptiveISIdistance( double time,double threshold)
{

    int pairs = 0;
    double sum = 0;
    
    for (int i = 0; i < STs.size();i++)
    {
        for(int i2 = i+1;i2 < STs.size(); i2++)
        {
            pairs++;
            Pair tmp(STs.at(i),STs.at(i2));  
            sum += tmp.AdaptiveISIdistance(time,threshold);

        }
    }
   
    double out = sum/(double)pairs;

    return out;
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-ISI-distance
 * Input: timefrom, time to, threshold
 * output: distance
 *
 * Other: A-ISI-distance of the whole set. Forms pairs and computes average
 */
double Spiketrains::AdaptiveISIdistance( double from, double to ,double threshold)
{	
    int pairs = 0;
    double sum = 0;
    for (int i = 0; i < STs.size()-1;i++)
    {
        for(int i2 = i+1;i2 < STs.size(); i2++)
        {
            pairs++;
            Pair tmp(STs.at(i),STs.at(i2));
            sum += tmp.AdaptiveISIdistance(from, to, threshold);
        }
    }
    return sum/(double)pairs;   
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-SPIKE-synchronization of the set
 * Input: time from, time to, recording start, recording end, threshold
 * output: A-SPIKE-synchronization value
 *
 * Other: -
 */
double Spiketrains::AdaptiveSPIKEsynchro(double* Edges, double start,double end,double T1,double T2,double max_dist,double threshold)
{	
    std::vector<std::vector < double> > Synchronized = 
                               AdaptiveSPIKEsynchroProfile(Edges,start,end,T1,T2,max_dist,threshold);
    bool allEmpty = true;
    double sum = 0;
    double Spikes = 0;
    
    //mexPrintf("ASS - Edges0: %i, Edges1: %i \n",(int)Edges[0],(int)Edges[1]);
    for(int i = 0; i < Synchronized.size();i++)
    {
        if (!Synchronized.at(i).empty())
        {
            allEmpty = false;
        }        
        for(int ii = 0; ii < Synchronized.at(i).size(); ii++)
        {
            sum += Synchronized.at(i).at(ii);
        }
        Spikes += Synchronized.at(i).size();
    }
    if (allEmpty)
    {
        return 1;
    }
    else
    {
        return sum/Spikes;
    }
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-SPIKE-synchronization profile 
 * Input: time from, time to, recording start, recording end, threshold
 * output: vector<vector> containing the synchronization values of each spike
 *
 * Other: The formation of the profile is left to the user.       
 */
std::vector< std::vector<double> > Spiketrains::AdaptiveSPIKEsynchroProfile(double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold)
{
    std::vector< std::vector<double> > Synchronized;
    
    //mexPrintf("ASSP - Edges0: %i, Edges1: %i \n",(int)Edges[0],(int)Edges[1]);
    for (int i = 0; i < STs.size();i++)
    {
            Synchronized.push_back( AdaptiveSynchro(i,Edges,T1,T2,T3,T4,max_dist,threshold) );
    }
    return Synchronized;
    
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: SPIKE-order profile 
 * Input: time from, time to, recording start, recording end, threshold
 * output: vector<vector> containing the order values of each spike
 *
 * Other: The formation of the profile is left to the user.       
 */
std::vector< std::vector<double> > Spiketrains::AdaptiveSPIKEorderProfile(double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold)
{
    std::vector< std::vector<double> > Synchronized;
    for (int i = 0; i < STs.size();i++)
    {
            Synchronized.push_back( AdaptiveSpikeOrder(i,Edges,T1,T2,T3,T4,max_dist,threshold) );
    }
    return Synchronized;
    
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: spiketrain-order profile 
 * Input: time from, time to, recording start, recording end, threshold
 * output: vector<vector> containing the order values of each spike
 *
 * Other: The formation of the profile is left to the user.       
 */
std::vector< std::vector<double> > Spiketrains::AdaptiveSPIKEtrainOrderProfile(double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold)
{
    std::vector< std::vector<double> > Synchronized;
    for (int i = 0; i < STs.size();i++)
    {
            Synchronized.push_back( AdaptiveSpikeTrainOrder(i,Edges,T1,T2,T3,T4,max_dist,threshold) );
    }
    return Synchronized;
    
}    

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-SPIKE-synchronization of a single spike train in the set 
 * Input: time from, time to, recording start, recording end, threshold
 * output: vector<vector> containing the synchronization values of each spike
 *
 * Other: -      
 */
std::vector<double> Spiketrains::AdaptiveSynchro(int spiketrainINDEX,double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold)
{
    double maxwindow = T2-T1;
    std::vector<double> coincidencevalues; 
    
    // mexPrintf("AS1 - Spike Train: %i, Length: %i,  Edges1: %i \n",spiketrainINDEX,STs.at(spiketrainINDEX)->Length(),(int)Edges[spiketrainINDEX]);
    if (!STs.at(spiketrainINDEX)->isempty(T1,T2) || Edges[spiketrainINDEX] == 3)
    {
        // mexPrintf("AS2 - Spike Train: %i, Length: %i,  Edges1: %i \n",spiketrainINDEX,STs.at(spiketrainINDEX)->Length(),(int)Edges[spiketrainINDEX]);
        for (int spike = 1; spike < STs.at(spiketrainINDEX)->Length()-1;spike++)
        {
            double spiketime = STs.at(spiketrainINDEX)->GiveSpikeAtIndex(spike);

            if (T3 <= spiketime && spiketime <= T4)
            { 
                double sum = 0;
                for (int otherSTINDEX = 0;otherSTINDEX < STs.size();otherSTINDEX++)
                {
                    if ((!STs.at(otherSTINDEX)->isempty(T1,T2) || Edges[otherSTINDEX] == 3) && otherSTINDEX != spiketrainINDEX)
                    {
                        sum += AdaptiveCoincidence(spike, spiketrainINDEX,otherSTINDEX,Edges,T1,T2,maxwindow,max_dist,threshold);
                    }
                }
                // For each spike push one value or no value at all
                coincidencevalues.push_back(sum/(STs.size()-1));
            }
        }
    }
    
    //mexPrintf("sts: %i,      cvs: %i \n",STs.size(),coincidencevalues.size());
    return coincidencevalues;   
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: SPIKE-order of a single spike train in the set 
 * Input: time from, time to, recording start, recording end, threshold
 * output: vector<vector> containing the order values of each spike
 *
 * Other: -      
 */
std::vector<double> Spiketrains::AdaptiveSpikeOrder(int spiketrainINDEX, double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold)
{
    double maxwindow = T2-T1;
    std::vector<double> coincidencevalues;
    if (!STs.at(spiketrainINDEX)->isempty(T1,T2) || Edges[spiketrainINDEX] == 3)
    {
        for (int spike = 1; spike < STs.at(spiketrainINDEX)->Length()-1;spike++)
        {
            double spiketime = STs.at(spiketrainINDEX)->GiveSpikeAtIndex(spike);
            
            if (T3 <= spiketime && spiketime <= T4)
            {
                double sum = 0;
                for (int otherSTINDEX = 0;otherSTINDEX < STs.size();otherSTINDEX++)
                {
                    if ((!STs.at(otherSTINDEX)->isempty(T1,T2) || Edges[otherSTINDEX] == 3) && otherSTINDEX != spiketrainINDEX)
                    {
                        sum += AdaptiveOrderOfSpikes(spike, spiketrainINDEX,otherSTINDEX,Edges,T1,T2,maxwindow,max_dist,threshold);
                    }
                }
                // For each spike push one value or no value at all
                coincidencevalues.push_back(sum/(STs.size()-1));
            }
        }
    }
    return coincidencevalues;
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: SPIKE Train Order of a single spike train in the set 
 * Input: time from, time to, recording start, recording end, threshold
 * output: vector<vector> containing the order values of each spike
 *
 * Other: -      
 */
std::vector<double> Spiketrains::AdaptiveSpikeTrainOrder(int spiketrainINDEX, double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold)
{
    double maxwindow = T2-T1;
    std::vector<double> coincidencevalues;
    if (!STs.at(spiketrainINDEX)->isempty(T1,T2) || Edges[spiketrainINDEX] == 3)
    {
        for (int spike = 1; spike < STs.at(spiketrainINDEX)->Length()-1;spike++)
        {
            double spiketime = STs.at(spiketrainINDEX)->GiveSpikeAtIndex(spike);
            
            if (T3 <= spiketime && spiketime <= T4)
            {
                double sum = 0;
                for (int otherSTINDEX = 0;otherSTINDEX < STs.size();otherSTINDEX++)
                {
                    if ((!STs.at(otherSTINDEX)->isempty(T1,T2) || Edges[otherSTINDEX] == 3) && otherSTINDEX != spiketrainINDEX)
                    {
                        sum += AdaptiveOrderOfSpikeTrains(spike, spiketrainINDEX,otherSTINDEX,Edges,T1,T2,maxwindow,max_dist,threshold);
                    }
                }
                // For each spike push one value or no value at all
                coincidencevalues.push_back(sum/(STs.size()-1));
            }
        }
    }
    return coincidencevalues;
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: Coincidence detection
 * Input: spike index of spike train 1, spiketrain 1 index, spiketrain 2
 *        index, maximum window size, threshold
 * output: 1 if coincidence exists between spike trains. 0 if not
 *
 * Other: -
 */
double Spiketrains::AdaptiveCoincidence(int NthSpike,int STindex1,int STindex2,double* Edges,double start,double end,double maxwindow,double max_dist,double threshold)
{    
    double spike11 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike-1);
    double spike12 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike);
    double spike13 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike+1);
    
    double spiketime = spike12;
    double closestSpike = STs.at(STindex2)->GiveSpikeAtIndex(1);
    int closestSpikeINDEX = 1;
    
    for( int spike = 1; spike < STs.at(STindex2)->Length()-1; spike++)
    {
        double candidateSpike = STs.at(STindex2)->GiveSpikeAtIndex(spike);
        if ( std::abs(candidateSpike-spiketime) < std::abs(closestSpike-spiketime) && !((int)Edges[STindex2] % 2==0 && candidateSpike==start) && !((int)Edges[STindex2]<2 && candidateSpike==end))
        {
            closestSpike = candidateSpike;
            closestSpikeINDEX = spike;
        }
    }
    // mexPrintf("AC - Train1: %i, Train2: %i, Spike: %i, closestSpike %f\n",STindex1,STindex2,NthSpike,closestSpike);
    // mexPrintf("AC - Test1: %i, Test2: %i \n",!((int)Edges[STindex2] % 2==0 && closestSpike==start),!((int)Edges[STindex2]<2 && closestSpike==end));
    
    if ((closestSpike == spiketime) && (!((int)Edges[STindex2] % 2==0 && closestSpike==start) && !((int)Edges[STindex2]<2 && closestSpike==end)))
    {
        return (double)1;
    }
    else
    {
        double spike21 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX-1);
        double spike22 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX);
        double spike23 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX+1);
        
        double ISI11 = spike12 - spike11;
        double ISI12 = spike13 - spike12;
        double ISI21 = spike22 - spike21;
        double ISI22 = spike23 - spike22;
        

        // Not using the first and last interval
        if( NthSpike-1 == 0)
        {
            ISI11 = maxwindow;
        }
        if( NthSpike+1 == STs.at(STindex1)->Length()-1)
        {
            ISI12 = maxwindow;
        }
        if( closestSpikeINDEX-1 == 0)
        {
            ISI21 = maxwindow;
        }
        if( closestSpikeINDEX+1 == STs.at(STindex2)->Length()-1)
        {
            ISI22 = maxwindow;
        }
        
        
        double TAUij = 0;
        double tau1 = std::min(ISI11,ISI12)/2;
        double tau2 = std::min(ISI21,ISI22)/2;
        
        if (spike12 <= spike22){
            
            double tau1F = std::min(std::max(threshold/2,tau1),ISI12/2);
            double tau2P = std::min(std::max(threshold/2,tau2),ISI21/2);
            TAUij = std::min(tau1F,tau2P);
            
        }else{
            
            double tau1P = std::min(std::max(threshold/2,tau1),ISI11/2);
            double tau2F = std::min(std::max(threshold/2,tau2),ISI22/2);
            TAUij = std::min(tau1P,tau2F);
            
        }

        if( std::abs(spiketime-closestSpike) < TAUij )
        {
			if (((max_dist < 0) || (std::abs(spiketime-closestSpike) < max_dist)) && (!((int)Edges[STindex2] % 2==0 && closestSpike==start) && !((int)Edges[STindex2]<2 && closestSpike==end)))
			{
				return (double)1;
			}
			else
			{
				return 0;
			}   
        }
        else
        {
            return 0;
        }   
    }   
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: Adaptive Order Of Spikes
 * Input: spike index of spike train 1, spiketrain 1 index, spiketrain 2
 *        index, maximum window size, threshold
 * output: 
 *
 * Other: -
 */
double Spiketrains::AdaptiveOrderOfSpikes(int NthSpike,int STindex1,int STindex2,double* Edges,double start,double end,double maxwindow,double max_dist,double threshold)
{
    double spike11 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike-1);
    double spike12 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike);
    double spike13 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike+1);
    
    double spiketime = spike12;
    double closestSpike = STs.at(STindex2)->GiveSpikeAtIndex(1);
    int closestSpikeINDEX = 1;
    
    //mexPrintf("AOS - Edges0: %i, Edges1: %i, Edges2: %i \n",(int)Edges[0],(int)Edges[1],(int)Edges[2]);
    
    for( int spike = 1; spike < STs.at(STindex2)->Length()-1; spike++)
    {
        double candidateSpike = STs.at(STindex2)->GiveSpikeAtIndex(spike);
        
        //mexPrintf("AOS - candidate %f, Edges: %i, Test1: %i, Test2: %i \n",candidateSpike,(int)Edges[STindex2],(int)Edges[STindex2] % 2==0,(int)Edges[STindex2]<2);
        //mexPrintf("AOS - Test3: %i, Test4: %i \n",candidateSpike==start,candidateSpike==end);
        //mexPrintf("AOS - Test5: %i, Test6: %i \n",!((int)Edges[STindex2] % 2==0 && candidateSpike==start),!((int)Edges[STindex2]<2 && candidateSpike==end));
        if ( std::abs(candidateSpike-spiketime) < std::abs(closestSpike-spiketime) && !((int)Edges[STindex2] % 2==0 && candidateSpike==start) && !((int)Edges[STindex2]<2 && candidateSpike==end))
        {
            closestSpike = candidateSpike;
            closestSpikeINDEX = spike;
        }
    }
    //mexPrintf("AOS - closestSpike %f\n",closestSpike);
    
    if ((closestSpike == spiketime) || ((int)Edges[STindex2] % 2==0 && closestSpike==start) || ((int)Edges[STindex2]<2 && closestSpike==end))
    {
        //mexPrintf("Spikes at the same time \n");
        return (double)0;                                                   // ############
    }
    else
    {
        double spike21 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX-1);
        double spike22 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX);
        double spike23 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX+1);
        
        double ISI11 = spike12 - spike11;
        double ISI12 = spike13 - spike12;
        double ISI21 = spike22 - spike21;
        double ISI22 = spike23 - spike22;
        
        // Not using the first and last interval
        if( NthSpike-1 == 0)
        {
            ISI11 = maxwindow;
        }
        if( NthSpike+1 == STs.at(STindex1)->Length()-1)
        {
            ISI12 = maxwindow;
        }
        if( closestSpikeINDEX-1 == 0)
        {
            ISI21 = maxwindow;
        }
        if( closestSpikeINDEX+1 == STs.at(STindex2)->Length()-1)
        {
            ISI22 = maxwindow;
        }
        
        
        double TAUij = 0;
        double tau1 = std::min(ISI11,ISI12)/2;
        double tau2 = std::min(ISI21,ISI22)/2;
        
        if (spike12 <= spike22){
            
            double tau1F = std::min(std::max(threshold/2,tau1),ISI12/2);
            double tau2P = std::min(std::max(threshold/2,tau2),ISI21/2);
            TAUij = std::min(tau1F,tau2P);
            
        }else{
            
            double tau1P = std::min(std::max(threshold/2,tau1),ISI11/2);
            double tau2F = std::min(std::max(threshold/2,tau2),ISI22/2);
            TAUij = std::min(tau1P,tau2F);
            
        }        
                
        if( std::abs(spiketime-closestSpike) < TAUij)
        {
            if(std::abs(spiketime-closestSpike) < max_dist)
            {
                if (spiketime == closestSpike)
                {
                    return 0;
                }
                else if(spiketime < closestSpike)
                {
                    //mexPrintf("1 \n");
                    return (double)1;
                }
                else
                {
                    //mexPrintf("-1 \n");
                    return (double)-1;
                }
            }
            else
            {
                //mexPrintf("Out of window \n");
                return 0;
            }
        }
        else
        {
            //mexPrintf("Out of window \n");
            return 0;
        }   
    }   
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: Adaptive Order Of Spike Trains
 * Input: spike index of spike train 1, spiketrain 1 index, spiketrain 2
 *        index, maximum window size, threshold
 * output: 
 *
 * Other: -
 */
double Spiketrains::AdaptiveOrderOfSpikeTrains(int NthSpike,int STindex1,int STindex2,double* Edges,double start,double end,double maxwindow,double max_dist,double threshold)
{    
    double spike11 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike-1);
    double spike12 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike);
    double spike13 = STs.at(STindex1)->GiveSpikeAtIndex(NthSpike+1);
    
    double spiketime = spike12;
    double closestSpike = STs.at(STindex2)->GiveSpikeAtIndex(1);
    int closestSpikeINDEX = 1;
    
    for( int spike = 1; spike < STs.at(STindex2)->Length()-1; spike++)
    {
        double candidateSpike = STs.at(STindex2)->GiveSpikeAtIndex(spike);
        if ( std::abs(candidateSpike-spiketime) < std::abs(closestSpike-spiketime) && !((int)Edges[STindex2] % 2==0 && candidateSpike==start) && !((int)Edges[STindex2]<2 && candidateSpike==end))
        {
            closestSpike = candidateSpike;
            closestSpikeINDEX = spike;
        }
    }
    
    if ((closestSpike == spiketime) || ((int)Edges[STindex2] % 2==0 && closestSpike==start) || ((int)Edges[STindex2]<2 && closestSpike==end))
    {
        //mexPrintf("Spikes at the same time \n");
        return (double)0;                                                   // ############
    }
    else
    {
        double spike21 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX-1);
        double spike22 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX);
        double spike23 = STs.at(STindex2)->GiveSpikeAtIndex(closestSpikeINDEX+1);
        
        double ISI11 = spike12 - spike11;
        double ISI12 = spike13 - spike12;
        double ISI21 = spike22 - spike21;
        double ISI22 = spike23 - spike22;
        
        // Not using the first and last interval
        if( NthSpike-1 == 0)
        {
            ISI11 = maxwindow;
        }
        if( NthSpike+1 == STs.at(STindex1)->Length()-1)
        {
            ISI12 = maxwindow;
        }
        if( closestSpikeINDEX-1 == 0)
        {
            ISI21 = maxwindow;
        }
        if( closestSpikeINDEX+1 == STs.at(STindex2)->Length()-1)
        {
            ISI22 = maxwindow;
        }       
        double TAUij = 0;
        double tau1 = std::min(ISI11,ISI12)/2;
        double tau2 = std::min(ISI21,ISI22)/2;
        
        if (spike12 <= spike22){
            
            double tau1F = std::min(std::max(threshold/2,tau1),ISI12/2);
            double tau2P = std::min(std::max(threshold/2,tau2),ISI21/2);
            TAUij = std::min(tau1F,tau2P);
            
        }else{
            
            double tau1P = std::min(std::max(threshold/2,tau1),ISI11/2);
            double tau2F = std::min(std::max(threshold/2,tau2),ISI22/2);
            TAUij = std::min(tau1P,tau2F);
            
        }        
                
        if( std::abs(spiketime-closestSpike) < TAUij)
        {
            if(std::abs(spiketime-closestSpike) < max_dist)
            {
                if (spiketime == closestSpike)
                {
                    return 0;
                }
                else if(spiketime < closestSpike)
                {
                    if (STindex1 < STindex2)
                    {
                        return (double)1;
                    }
                    else
                    {
                        return (double)-1;
                    }
                }
                else
                {
                    if (STindex1 > STindex2)
                    {
                        return (double)1;
                    }
                    else
                    {
                        return (double)-1;
                    }
                }
                
            }
            else
            {
                //mexPrintf("Out of window \n");
                return 0;
            }
        }
        else
        {
            //mexPrintf("Out of window \n");
            return 0;
        }   
    }   
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-SPIKE-distance
 * Input: time at, threshold (Triggered Averaging)
 * output: distance
 *
 * Other: A-SPIKE-distance of the whole set. Forms pairs and computes average
 */
double Spiketrains::AdaptiveSPIKEdistance( double time, double threshold)
{

    int pairs = 0;
    double sum = 0;
    
    for (int i = 0; i < STs.size();i++)
    {
        for(int i2 = i+1;i2 < STs.size(); i2++)
        {
            
            pairs++;
            Pair tmp(STs.at(i),STs.at(i2));
            sum += tmp.AdaptiveSPIKEdistance(time,threshold);
        }
    }
   
    double out = sum/(double)pairs;

    return out;
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-SPIKE-distance
 * Input: time from, time to, threshold (Selective Averaging)
 * output: distance
 *
 * Other: A-SPIKE-distance of the whole set. Forms pairs and computes average
 */
double Spiketrains::AdaptiveSPIKEdistance( double from, double to, double threshold)
{	

    int pairs = 0;
    double sum = 0;

    for (int i = 0; i < STs.size()-1;i++)
    {
        for(int i2 = i+1;i2 < STs.size(); i2++)
        {
           
            pairs++;
            Pair tmp(STs.at(i),STs.at(i2)); 
            sum += tmp.AdaptiveSPIKEdistance(from, to, threshold);
        }
    }
    
    double out = sum/(double)pairs;
    
    return out;
    
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: ARI-SPIKE-distance
 * Input: time at, threshold (Triggered Averaging)
 * output: distance
 *
 * Other: ARI-SPIKE-distance of the whole set. Forms pairs and computes average
 */
double Spiketrains::AdaptiveRateIndependentSPIKEdistance( double time, double threshold)
{

    int pairs = 0;
    double sum = 0;
    
    for (int i = 0; i < STs.size();i++)
    {
        for(int i2 = i+1;i2 < STs.size(); i2++)
        {
            
            pairs++;
            Pair tmp(STs.at(i),STs.at(i2));
            sum += tmp.AdaptiveRateIndependentSPIKEdistance(time,threshold);
        }
    }
   
    double out = sum/(double)pairs;

    return out;
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: ARI-SPIKE-distance
 * Input: time from, time to, threshold (Selective Averaging)
 * output: distance
 *
 * Other: ARI-SPIKE-distance of the whole set. Forms pairs and computes average
 */
double Spiketrains::AdaptiveRateIndependentSPIKEdistance( double from, double to, double threshold)
{	

    int pairs = 0;
    double sum = 0;

    for (int i = 0; i < STs.size()-1;i++)
    {
        for(int i2 = i+1;i2 < STs.size(); i2++)
        {
           
            pairs++;
            Pair tmp(STs.at(i),STs.at(i2)); 
            sum += tmp.AdaptiveRateIndependentSPIKEdistance(from, to, threshold);
        }
    }
    
    double out = sum/(double)pairs;
    
    return out;
    
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: Give numver of spiketrains in teh set
 * Input: -
 * output: the number
 *
 * Other: -
 */
int Spiketrains::NumberOfSpikeTrains()
{
    return STs.size();
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-ISI-distance matrix
 * Input: reference to matrix, time from, time to, threshold
 * output: -
 *
 * Other: output through reference to the matrix.
 */
void Spiketrains::AdaptiveISIDistanceMatrix(double* &matrix, double T1,double T2,double threshold)
{
    bool point = (T1==T2);
    int pairs = 0;
    int nroSTs = STs.size();
    for (int n = 0; n < nroSTs-1;n++)
    {
        for(int m = n+1;m < nroSTs; m++)
        {
            double distance;
            
            Pair tmp(STs.at(n),STs.at(m));
            
            if(point)
            {
                distance = tmp.AdaptiveISIdistance(T1,threshold);
            }
            else
            {
                distance = tmp.AdaptiveISIdistance(T1, T2,threshold);
            }
            matrix[n + nroSTs*m] = distance;
            matrix[m + nroSTs*n] = distance;
        }
    }

}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-SPIKE-distance matrix
 * Input: reference to matrix, time from, time to, threshold
 * output: -
 *
 * Other: output through reference to the matrix.
 */
void Spiketrains::AdaptiveSPIKEDistanceMatrix(double* &matrix, double T1,double T2,double threshold)
{
    bool point = (T1==T2);
    int pairs = 0;
    int nroSTs = STs.size();
    for (int n = 0; n < nroSTs-1;n++)
    {
        for(int m = n+1;m < nroSTs; m++)
        {
            double distance;
            
            Pair tmp(STs.at(n),STs.at(m));
            
            if(point)
            {
                distance = tmp.AdaptiveSPIKEdistance(T1,threshold);
            }
            else
            {
                distance = tmp.AdaptiveSPIKEdistance(T1, T2,threshold);
            }
            
            matrix[n + nroSTs*m] = distance;
            matrix[m + nroSTs*n] = distance;
        }
    }
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-Rate-Independent-SPIKE-distance matrix
 * Input: reference to matrix, time from, time to, threshold
 * output: -
 *
 * Other: output through reference to the matrix.
 */
void Spiketrains::AdaptiveRateIndependentSPIKEDistanceMatrix(double* &matrix, double T1,double T2,double threshold)
{
    bool point = (T1==T2);
    int pairs = 0;
    int nroSTs = STs.size();
    for (int n = 0; n < nroSTs-1;n++)
    {
        for(int m = n+1;m < nroSTs; m++)
        {
            double distance;
            
            Pair tmp(STs.at(n),STs.at(m));
            
            if(point)
            {
                distance = tmp.AdaptiveRateIndependentSPIKEdistance(T1,threshold);
            }
            else
            {
                distance = tmp.AdaptiveRateIndependentSPIKEdistance(T1, T2,threshold);
            }
            
            matrix[n + nroSTs*m] = distance;
            matrix[m + nroSTs*n] = distance;
        }
    }
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-ISI-distance profile
 * Input: reference to X values, reference to Y values,  from time, to time
 *        threshold
 * output: -
 *
 * Other: output is given through reference variables
 */
void Spiketrains::AdaptiveISIDistanceProfile(std::vector<double> &Xprofile,std::vector<double> &Yprofile, double T1,double T2,double threshold)
{
    std::vector<ISIProfile*> PairwiseProfiles;
    int nroSTs = STs.size();
    for (int n = 0; n < nroSTs-1;n++)
    {
        for(int m = n+1;m < nroSTs; m++)
        {       
            std::vector<double>  Y ,X;
            Pair tmpPair(STs.at(n),STs.at(m));
            // Returns profile in X and Y
            tmpPair.AdaptiveISIprofile(T1,T2,X,Y,threshold);
            
            ISIProfile* tmpProfile = new ISIProfile(X,Y);
            
            PairwiseProfiles.push_back(tmpProfile);    
        }   
    }

    ISIProfile* Combined = PairwiseProfiles.at(0);
    for(int i = 1; i < PairwiseProfiles.size();i++)
    {
        Combined->AddISIprofile( PairwiseProfiles.at(i) );
    }
    Xprofile = Combined->GetProfileX();
    Yprofile = Combined->GetProfileY();

    for (int i = 0; i < PairwiseProfiles.size();i++)
    {
        delete PairwiseProfiles.at(i);
    }
}

/* Date: 18.8.2016
 * Version: 1.0
 *
 * Function: A-SPIKE-distance profile
 * Input: reference to X values, reference to Y values,  from time, to time
 *        threshold
 * output: -
 *
 * Other: output is given through reference variables
 */
void Spiketrains::AdaptiveSPIKEDistanceProfile(std::vector<double> &Xprofile,std::vector<double> &Yprofile, double T1,double T2, double threshold)
{
    std::vector<SPIKEProfile*> PairwiseProfiles;
    int nroSTs = STs.size();
    for (int n = 0; n < nroSTs-1;n++)
    {
        for(int m = n+1;m < nroSTs; m++)
        {
            
            std::vector<double>  Y ,X;
            Pair tmpPair(STs.at(n),STs.at(m));
           
            // Returns profile in X and Y
            tmpPair.AdaptiveSPIKEprofile(T1,T2,X,Y,threshold);
           
            SPIKEProfile* tmpProfile = new SPIKEProfile(X,Y);
            
            PairwiseProfiles.push_back(tmpProfile);
           
        }
      
    }

    SPIKEProfile* Combined = PairwiseProfiles.at(0);
    for(int i = 1; i < PairwiseProfiles.size();i++)
    {
        Combined->AddSPIKEprofile( PairwiseProfiles.at(i) );
    }
    Xprofile = Combined->GetProfileX();
    Yprofile = Combined->GetProfileY();

    for (int i = 0; i < PairwiseProfiles.size();i++)
    {
        delete PairwiseProfiles.at(i);
    } 
}

/* Date: 16.1.2018
 * Version: 1.0
 *
 * Function: RIA-SPIKE-distance profile
 * Input: reference to X values, reference to Y values,  from time, to time
 *        threshold
 * output: -
 *
 * Other: output is given through reference variables
 */
void Spiketrains::AdaptiveRateIndependentSPIKEDistanceProfile(std::vector<double> &Xprofile,std::vector<double> &Yprofile, double T1,double T2, double threshold)
{
    std::vector<SPIKEProfile*> PairwiseProfiles;
    int nroSTs = STs.size();
    for (int n = 0; n < nroSTs-1;n++)
    {
        for(int m = n+1;m < nroSTs; m++)
        {
            
            std::vector<double>  Y ,X;
            Pair tmpPair(STs.at(n),STs.at(m));
           
            // Returns profile in X and Y
            tmpPair.AdaptiveRateIndependentSPIKEprofile(T1,T2,X,Y,threshold);
           
            SPIKEProfile* tmpProfile = new SPIKEProfile(X,Y);
            
            PairwiseProfiles.push_back(tmpProfile);
           
        }
      
    }

    SPIKEProfile* Combined = PairwiseProfiles.at(0);
    for(int i = 1; i < PairwiseProfiles.size();i++)
    {
        Combined->AddSPIKEprofile( PairwiseProfiles.at(i) );
    }
    Xprofile = Combined->GetProfileX();
    Yprofile = Combined->GetProfileY();

    for (int i = 0; i < PairwiseProfiles.size();i++)
    {
        delete PairwiseProfiles.at(i);
    }
}