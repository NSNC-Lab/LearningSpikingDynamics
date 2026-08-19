clear all;
close all;
InitializecSPIKE;
stimulusIndices = [2 3 10 12];

NROrepeats = 5;
NROStimuli = 2;% (coding, noncoding)
NRONeurons = 4;
%% Creating spike trains
%{
rate = 20;% Hz
nroNeurons = 1;%
responseRATEs = [ones(NRONeurons,1)*-1 ones(NRONeurons,1)*rate];
timeCoding = 1;
baseRate = rate;
noise = 0;
jitter = 5;%[ms]
time = 1;%[s]
refractoryPeriod = 2;%[ms]

for n = 1:NRONeurons
    RU{n} = NeuronResponseUnit(nroNeurons,responseRATEs(n,:),baseRate,timeCoding, noise, jitter, time, refractoryPeriod);
end

for n = 1:NRONeurons
   for s = 1:NROStimuli
       for r = 1:NROrepeats
           CELL{n,s,r} = RU{n}.Stimulus(s);
       end
   end
end

figure('rend','painters','pos',[50 50 900 600]);
for n = 1:NRONeurons
    for s = 1:NROStimuli
        for r = 1:NROrepeats
            STs{r} = CELL{n,s,r};
        end
        STS = SpikeTrainSet(STs,0,time);
        subplot(4,4,s+(n-1)*NRONeurons)
        if s == 2
            STS.plotSpikeTrainSet('g');
        else
            STS.plotSpikeTrainSet('r');
        end
        set(gca,'xtick',[])
        if s == 1
            h = text(-0.2,0.5,['Neuron ' num2str(n)] );
            set(h,'Rotation',90);
        end
        if n == 1 && s == 1
            text(0,6,'Non-coded s ' )
        end
        if n == 1 && s == 2
            text(0,6,'Coded s ' )
        end
    end
end

%% actual coding
NROrepeats = 5;
NROStimuli = 4;
NRONeurons = 4;
% 1 = car
% 2 = ship
% 3 = red
% 4 = white
%% First fill cell array with noise
for n = 1:NRONeurons
   for s = 1:NROStimuli
       for r = 1:NROrepeats
           CELL{n,s,r} = RU{n}.Stimulus(1);
       end
   end
end

s1 = [1 3];% neurons responding
s2 = [1 4];
s3 = [2 3];
s4 = [2 4];

% then fill the responses in

for n = s1
    s = 1;
    for r = 1:NROrepeats
        CELL{n,s,r} = RU{n}.Stimulus(2);
    end
end

for n = s2
    s = 2;
    for r = 1:NROrepeats
        CELL{n,s,r} = RU{n}.Stimulus(2);
    end
end

for n = s3
    s = 3;
    for r = 1:NROrepeats
        CELL{n,s,r} = RU{n}.Stimulus(2);
    end
end

for n = s4
    s = 4;
    for r = 1:NROrepeats
        CELL{n,s,r} = RU{n}.Stimulus(2);
    end
end

%gathering spike trains of single neurons

for n = 1:NRONeurons
   index = 1;
   for s = 1:NROStimuli
       for r = 1:NROrepeats
           STs{index} = CELL{n,s,r};
           index = index+1;
       end
   end
    STS = SpikeTrainSet(STs,0,time);
    DistMatrix{n} = STS.SPIKEdistanceMatrix;
end
save('LL')
%}

%% Plotting the spike trains
load('LL')
%figure('rend','painters','pos',[50 50 1314.3 1000]);
figure;
num_rows=5; num_cols=7; fs=10;


subplot(num_rows,num_cols,1)
imshow('White_Car_Pixnio.jpg')
title('Stimulus 1','FontWeight','bold','FontSize',fs)
xl=xlim; yl=ylim;
text(xl(1)-0.4*diff(xl),yl(1)-0.2*diff(yl),'A','FontWeight','bold','FontSize',fs+1);

subplot(num_rows,num_cols,2)
imshow('Red_Car_Pixnio.jpg')
title('Stimulus 2','FontWeight','bold','FontSize',fs)

subplot(num_rows,num_cols,3)
imshow('White_Ship_Pixnio.jpg')
title('Stimulus 3','FontWeight','bold','FontSize',fs)

subplot(num_rows,num_cols,4)
imshow('Red_Ship_Pixnio.jpg')
title('Stimulus 4','FontWeight','bold','FontSize',fs)


Colormap = [0,0,0;
    0,0,1;
    1,0,0;
    0,0.5,0;
    1,1,0];

for i = 1:NRONeurons
    M = zeros(NROStimuli);
    D = DiscriminationMatrix{i};
    for S1 = 1:NROStimuli
        for S2 = S1:NROStimuli
            inter = mean(Statistics{i}{S1,S2});
            intra = mean([Statistics{i}{S1,S1}  Statistics{i}{S2,S2}]);
            M(S1,S2) = D(S1,S2)*(inter-intra);
            M(S2,S1) = M(S1,S2);
        end
    end
    M_n{i} = M;
end

maxi1=0;
maxi2=0;
for n = 1:NRONeurons
    Matrix = DistMatrix{n};
    [performance,SMatrix,rMatrix,Distances,Statistics{n}] = PerformanceValue( Matrix,NROStimuli,NROrepeats);
    DiscriminationMatrix{n} = SMatrix;
    DiscriminationPerformanceMatrix{n} = rMatrix;
    maxi1=max([maxi1 max(max(DistMatrix{n}))]);
    maxi2=max([maxi2 max(max(M_n{n}))]);   
end



for n = 1:NRONeurons
    for i = 1:NROStimuli
        subplot(num_rows,num_cols,n*num_cols+i)
        %subplot(5,6,n*6-6+i)
        for r = 1:NROrepeats
            STs2{r} = CELL{n,i,r};
        end
        STS = SpikeTrainSet(STs2,0,time);
        STS.plotSpikeTrainSet;
        set(gca,'xtick',[])
        if n==NRONeurons
            xlabel('Time','FontWeight','bold','FontSize',fs)
        end
        if i == 1
            ylabel(['Neuron ' num2str(n)],'FontWeight','bold','FontSize',fs)
            %h = text(-0.2,0.5,['Neuron ' num2str(n)] );
            %set(h,'Rotation',90);
            if n==1
                xl=xlim; yl=ylim;
                text(xl(1)-0.4*diff(xl),yl(2)+0.2*diff(yl),'B','FontWeight','bold','FontSize',fs+1);
            end
        end
    end
    subplot(num_rows,num_cols,n*num_cols+5)
    imagesc(DistMatrix{n});
    caxis([0 maxi1])
    axis square
    set(gca,'YTick',[])
    set(gca,'XTick',[])
    if n==1
        xl=xlim; yl=ylim;
        text(xl(1)-0.4*diff(xl),yl(1)-0.2*diff(yl),'C','FontWeight','bold','FontSize',fs+1);
    elseif n==NRONeurons
        xl=xlim; yl=ylim;
        text(xl(1)+0.4*diff(xl),yl(2)+0.19*diff(yl),'D_S','FontWeight','bold','FontSize',fs);
    end
    
    
    subplot(num_rows,num_cols,n*num_cols+6)
    imagesc(DiscriminationMatrix{n}*n);
    colormap(gca,Colormap)
    caxis([0 4])
    axis square
    set(gca,'YTick',[])
    set(gca,'XTick',[])
    xl=xlim; yl=ylim;
    text(xl(1)+1.11*diff(xl),yl(1)+0.52*diff(yl),sprintf('M_{%i}', n),'FontWeight','bold','FontSize',fs);
    %title(sprintf('Discrimination M_{%i}', n),'FontWeight','bold','FontSize',fs);
    linewid = 1.5;
    DrawSeparators(NROStimuli,1,linewid)
    if n==1
        xl=xlim; yl=ylim;
        text(xl(1)-0.4*diff(xl),yl(1)-0.2*diff(yl),'D','FontWeight','bold','FontSize',fs+1);
    elseif n==NRONeurons
        text(xl(1)-0.2*diff(xl),yl(2)+0.19*diff(yl),'Discrimination','FontWeight','bold','FontSize',fs);        
    end
    
    subplot(num_rows,num_cols,n*num_cols+7)
    imagesc(M_n{n});
    caxis([0 maxi2])
    colormap(gca,'jet')
    axis square
    set(gca,'YTick',[])
    set(gca,'XTick',[])
    xl=xlim; yl=ylim;
    text(xl(1)+1.11*diff(xl),yl(1)+0.52*diff(yl),sprintf('P_{%i}', n),'FontWeight','bold','FontSize',fs);
    %title(sprintf('Discrimination M_{%i}', n),'FontWeight','bold','FontSize',fs);
    linewid = 1.5;
    DrawSeparators(NROStimuli,1,linewid)
    if n==1
        xl=xlim; yl=ylim;
        text(xl(1)-0.4*diff(xl),yl(1)-0.2*diff(yl),'E','FontWeight','bold','FontSize',fs+1);
    elseif n==NRONeurons
        text(xl(1)-0.15*diff(xl),yl(2)+0.19*diff(yl),'Performance','FontWeight','bold','FontSize',fs);
    end
end



P = zeros(NROStimuli);
LL_Neurons = [];
BestNeuronM = zeros(NRONeurons);

% Calculating the LL population by selecting the best neuron for each
% stimulus pair into the population
for S1 = 1:NROStimuli
    for S2 = S1:NROStimuli
        BestNeuron = -1;
        BestResult = 0;
        for i = 1:NRONeurons
            Matrix = M_n{i};
            
            if Matrix(S1,S2) > BestResult
                BestNeuron = i;
                BestResult = Matrix(S1,S2);
            end
            
        end
        if BestNeuron ~= -1
            LL_Neurons(end+1) = BestNeuron;
            P(S1,S2) = BestResult;
            BestNeuronM(S1,S2) = BestNeuron;
            BestNeuronM(S2,S1) = BestNeuron;
        end
    end
end
LL_Neurons = unique(LL_Neurons);
Performance = mean(P(P > 0));
P=P+P';

% Population Performance matrix
subplot(num_rows,num_cols,7)
imagesc(P);
%colormap(gca,Colormap)
caxis([0 maxi2])
colormap(gca,'jet')
axis square
set(gca,'YTick',[])
set(gca,'XTick',[])
xl=xlim; yl=ylim;
text(xl(1)+1.11*diff(xl),yl(1)+0.5*diff(yl),'P','FontWeight','bold','FontSize',fs);
%text(xl(1)+0.45*diff(xl),yl(1)-0.15*diff(yl),'P','FontWeight','bold','FontSize',fs);
linewid = 1.5;
DrawSeparators(NROStimuli,1,linewid)
xl=xlim; yl=ylim;
text(xl(1)-0.4*diff(xl),yl(1)-0.2*diff(yl),'F','FontWeight','bold','FontSize',fs+1);

% Population matrix
subplot(num_rows,num_cols,6)
imagesc(BestNeuronM);
colormap(gca,Colormap)
caxis([0 4])
axis square
set(gca,'YTick',[])
set(gca,'XTick',[])
xl=xlim; yl=ylim;
%text(xl(1)-0.2*diff(xl),yl(1)-0.19*diff(yl),'Discrimination','FontWeight','bold','FontSize',fs);
text(xl(1)+1.11*diff(xl),yl(1)+0.5*diff(yl),'M','FontWeight','bold','FontSize',fs);
%text(xl(1)+0.45*diff(xl),yl(1)-0.15*diff(yl),'M','FontWeight','bold','FontSize',fs);
linewid = 1.5;
DrawSeparators(NROStimuli,1,linewid)
xl=xlim; yl=ylim;
text(xl(1)-0.4*diff(xl),yl(1)-0.2*diff(yl),'G','FontWeight','bold','FontSize',fs+1);

annotation('arrow',0.35*ones(1,2),[0.8 0.76],'LineWidth',3,'Color','k')
annotation('arrow',[0.565 0.59],0.43*ones(1,2),'LineWidth',3,'Color','k')
annotation('arrow',[0.685 0.71],0.43*ones(1,2),'LineWidth',3,'Color','k')
annotation('arrow',[0.795 0.82],0.43*ones(1,2),'LineWidth',3,'Color','k')
annotation('rectangle',[0.825 0.28 0.096 0.485],'LineWidth',2,'Color','k')
annotation('arrow',0.864*ones(1,2),[0.765 0.805],'LineWidth',3,'Color','k')
annotation('arrow',[0.83 0.805],0.865*ones(1,2),'LineWidth',3,'Color','k')

%set(gcf,'PaperOrientation','Portrait'); set(gcf,'PaperType','A4');
%set(gcf,'PaperUnits','Normalized','PaperPosition',[0.04 0.3 0.92 0.63]);
set(gcf,'PaperOrientation','Landscape'); set(gcf,'PaperType','A4');
set(gcf,'PaperUnits','Normalized','PaperPosition',[0.04 0.25 0.92 0.7]);
psname='LL_Coding_Illustration.ps';
print(gcf,'-dpsc',psname)


