%% Make Figure
% neurons = size(CellMatrix,1);
% stimuli = size(CellMatrix,2);
% repeats = size(CellMatrix,3);
% % [sortedPerformances,NeuronIndices] = sort(DiscriminationPerformances,'descend');
% % for labels = 1:size(NeuronIndices,1)
% %     XLABEL{labels} = num2str(NeuronIndices(labels));
% % end
% % SortedspPOP = logical(spPOP(NeuronIndices));
% % SortedllPOP = logical(llPOP(NeuronIndices));
% % SortedNoPOP = logical(ones(1,neurons)-SortedspPOP-SortedllPOP);
% placements = 1:neurons;
% % sortedSPAperformance = (spAperformance'*spAPOP)';
% % sortedSPAperformance = sortedSPAperformance(NeuronIndices);
% % sortedSPGperformance = (spGperformance'*spGPOP)';
% % sortedSPGperformance = sortedSPGperformance(NeuronIndices);
% % sortedSPBperformance = (spBperformance'*spBPOP)';
% % sortedSPBperformance = sortedSPBperformance(NeuronIndices);
% % AllPerformances = [sortedPerformances';sortedSPAperformance;sortedSPGperformance;sortedSPBperformance];
% spPOP = logical(spPOP);
% llPOP = logical(llPOP);
% NoPOP = logical(ones(1,neurons)-spPOP-llPOP);
% AllPerformances = [DiscriminationPerformances';spAperformance'*spAPOP';spGperformance'*spGPOP';spGBperformance'*spGBPOP';spBperformance'*spBPOP' ];
% 
% for labels = 1:neurons
%     XLABEL{labels} = num2str(labels);
% end
% 
% figure;
% subplot(1,2,2);
% SPplot = bar(placements(spPOP),AllPerformances(:,spPOP)',1);
% SPplot(1).FaceColor = 'red';
% SPplot(2).FaceColor = 'black';
% SPplot(3).FaceColor = 'yellow';
% SPplot(4).FaceColor = 'Cyan';
% SPplot(5).FaceColor = 'Green';
% hold on;
% LLplot = bar(placements(llPOP)',AllPerformances(:,llPOP)',1);
% LLplot(1).FaceColor = 'red';
% LLplot(2).FaceColor = 'black';
% LLplot(3).FaceColor = 'yellow';
% LLplot(4).FaceColor = 'Cyan';
% LLplot(5).FaceColor = 'Green';
% NCplot = bar(placements(NoPOP)',AllPerformances(:,NoPOP)',1);
% NCplot(1).FaceColor = 'red';
% NCplot(2).FaceColor = 'black';
% NCplot(3).FaceColor = 'yellow';
% NCplot(4).FaceColor = 'Cyan';
% NCplot(5).FaceColor = 'Green';
% set(gca, 'XTick',placements)
% set(gca, 'XTickLabel', XLABEL)
% ylim([-0.1,0.5]);
% xlabel('Neuron Index')
% ylabel('Discrimination Performance: D_S')
% title('Discrimination performances and found populations')
% legend(LLplot,'LL single neuron: Fig on the left','Population: Annealing','Population: Greedy top-down','Population: Greedy bottom-up','Population: Brute','location','northeast')
% 
% subplot(1,2,1);
% SPplot = bar(placements(spPOP),DiscriminationPerformances(spPOP), 'r');
% hold on;
% LLplot = bar(placements(llPOP),DiscriminationPerformances(llPOP), 'b');
% NCplot = bar(placements(NoPOP),DiscriminationPerformances(NoPOP), 'g');
% set(gca, 'XTick',placements)
% set(gca, 'XTickLabel', XLABEL)
% ylim([-0.1,0.5]);
% xlabel('Neuron Index')
% ylabel('LL Discrimination Performance: D_S')
% title('LL discrimination')
% legend([LLplot, SPplot, NCplot],'Labelled line coding neurons','Summed population coding neurons','Non-coding neurons','location','northeast')
% 

%% Sorted

neurons = size(CellMatrix,1);
stimuli = size(CellMatrix,2);
repeats = size(CellMatrix,3);
[sortedPerformances,NeuronIndices] = sort(DiscriminationPerformances,'descend');
for labels = 1:size(NeuronIndices,1)
    XLABEL{labels} = num2str(NeuronIndices(labels));
end
SortedspPOP = logical(spPOP(NeuronIndices));
SortedllPOP = logical(llPOP(NeuronIndices));
SortedNoPOP = logical(ones(1,neurons)-SortedspPOP-SortedllPOP);
placements = 1:neurons;
sortedSPAperformance = (spAperformance'*spAPOP)';
sortedSPAperformance = sortedSPAperformance(NeuronIndices);
sortedSPGperformance = (spGperformance'*spGPOP)';
sortedSPGperformance = sortedSPGperformance(NeuronIndices);
sortedSPGBperformance = (spGBperformance'*spGBPOP)';
sortedSPGBperformance = sortedSPGBperformance(NeuronIndices);
sortedSPBperformance = (spBperformance'*spBPOP)';
sortedSPBperformance = sortedSPBperformance(NeuronIndices);
AllPerformances = [sortedPerformances';sortedSPAperformance;sortedSPGperformance;sortedSPGBperformance;sortedSPBperformance];

figure('Position',[100 100 800 500]);
subplot(1,2,2);
SPplot = bar(placements(SortedspPOP),AllPerformances(:,SortedspPOP)',1);
SPplot(1).FaceColor = 'red';
SPplot(2).FaceColor = 'black';
SPplot(3).FaceColor = 'yellow';
SPplot(4).FaceColor = 'Cyan';
SPplot(5).FaceColor = 'Green';
hold on;
LLplot = bar(placements(SortedllPOP)',AllPerformances(:,SortedllPOP)',1);
LLplot(1).FaceColor = 'red';
LLplot(2).FaceColor = 'black';
LLplot(3).FaceColor = 'yellow';
LLplot(4).FaceColor = 'Cyan';
LLplot(5).FaceColor = 'Green';
NCplot = bar(placements(SortedNoPOP)',AllPerformances(:,SortedNoPOP)',1);
NCplot(1).FaceColor = 'red';
NCplot(2).FaceColor = 'black';
NCplot(3).FaceColor = 'yellow';
NCplot(4).FaceColor = 'Cyan';
NCplot(5).FaceColor = 'Green';
set(gca, 'XTick',placements)
set(gca, 'XTickLabel', XLABEL)
ylim([-0.1,0.5]);
xlabel('Neuron Index')
ylabel('Discrimination Performance: D_S')
title('Discrimination performances and found populations')
legend(LLplot,'LL single neuron: Fig on the left','Population: Annealing','Population: Greedy top-down','Population: Greedy bottom-up','Population: Brute','location','northeast')

subplot(1,2,1);
SPplot = bar(placements(SortedspPOP),sortedPerformances(SortedspPOP), 'r','BarWidth',0.8);
hold on;
LLplot = bar(placements(SortedllPOP),sortedPerformances(SortedllPOP), 'b','BarWidth',0.8);
NCplot = bar(placements(SortedNoPOP),sortedPerformances(SortedNoPOP), 'g','BarWidth',0.8);
set(gca, 'XTick',placements)
set(gca, 'XTickLabel', XLABEL)
ylim([-0.1,0.5]);
xlabel('Neuron Index')
ylabel('LL Discrimination Performance: D_S')
title('LL discrimination')
legend([LLplot, SPplot, NCplot],'Labelled line coding neurons','Summed population coding neurons','Non-coding neurons','location','northeast')
