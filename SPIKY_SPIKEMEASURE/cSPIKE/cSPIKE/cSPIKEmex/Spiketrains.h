/* Object: Spiketrains
 * Creator: Eero Satuvuori
 * Date: 18.8.2016
 * Version: 1.0
 * Purpose: 
 */

#ifndef Spiketrains_H
#define Spiketrains_H

#include <vector>
#include <cmath>
#include "Spiketrain.h"
#include "Pair.h"
#include "ISIProfile.h"
#include "SPIKEProfile.h"

class Spiketrains
{
private:
    std::vector<Spiketrain*> STs;
    
    std::vector<double> AdaptiveSynchro(int spiketrain,double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold);
    std::vector<double> AdaptiveSpikeTrainOrder(int spiketrainINDEX,double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold);
    std::vector<double> AdaptiveSpikeOrder(int spiketrainINDEX,double* Edges, double T1,double T2,double T3,double T4,double max_dist,double threshold);
    
public:
    Spiketrains();
    ~Spiketrains();
    
    Spiketrains(std::vector<Spiketrain*> STvector);
    
    void AddSpiketrain(Spiketrain* ST);
    int NumberOfSpikeTrains();
    
    
    double AdaptiveSPIKEsynchro(double* Edges, double start,double end ,double T1,double T2,double max_dist,double threshold);
    
    std::vector< std::vector<double> > AdaptiveSPIKEsynchroProfile(double* Edges, double start, double end,double T1,double T2,double max_dist,double threshold);
    std::vector< std::vector<double> > AdaptiveSPIKEorderProfile(double* Edges, double start, double end,double T1,double T2,double max_dist,double threshold);
    std::vector< std::vector<double> > AdaptiveSPIKEtrainOrderProfile(double* Edges, double start, double end,double T1,double T2,double max_dist,double threshold);
    
    double AdaptiveISIdistance(double time, double threshold);
    double AdaptiveISIdistance(double from, double to, double threshold);
    
    double AdaptiveCoincidence(int NthSpike, int STindex1, int STindex2,double* Edges,double start,double end,double maxwindow,double max_dist,double threshold);
    double AdaptiveOrderOfSpikeTrains(int NthSpike, int STindex1, int STindex2,double* Edges,double start,double end,double maxwindow,double max_dist,double threshold);
    double AdaptiveOrderOfSpikes(int NthSpike, int STindex1, int STindex2,double* Edges,double start,double end,double maxwindow,double max_dist,double threshold);
    
    double AdaptiveSPIKEdistance(double time, double threshold);
    double AdaptiveSPIKEdistance(double from, double to, double threshold);
    
    double AdaptiveRateIndependentSPIKEdistance(double time, double threshold);
    double AdaptiveRateIndependentSPIKEdistance(double from, double to, double threshold);
        
    void AdaptiveISIDistanceMatrix(double* &matrix ,double T1,double T2,double threshold);
    void AdaptiveSPIKEDistanceMatrix(double* &matrix ,double T1,double T2,double threshold);
    void AdaptiveRateIndependentSPIKEDistanceMatrix(double* &matrix ,double T1,double T2,double threshold);
    
    void AdaptiveISIDistanceProfile(std::vector<double> &Xprofile,std::vector<double> &Yprofile ,double T1,double T2, double threshold);
    void AdaptiveSPIKEDistanceProfile(std::vector<double> &Xprofile,std::vector<double> &Yprofile ,double T1,double T2, double threshold);
    void AdaptiveRateIndependentSPIKEDistanceProfile(std::vector<double> &Xprofile,std::vector<double> &Yprofile ,double T1,double T2, double threshold);
    
    Spiketrain* GetST(int index);
};
#endif