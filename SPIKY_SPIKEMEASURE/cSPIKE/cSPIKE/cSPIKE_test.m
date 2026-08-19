%InitializecSPIKE
%clc
close all

tmin=0;
tmax=1000;
threshold=1000;

measures=48;               % +1:ISI,+2:SPIKE,+4:RI-SPIKE,+8:SPIKE-Synchro,+16:SPIKE-order,+32:Spike Train Order
adaptive_measures=0;       % +1:ISI,+2:SPIKE,+4:RI-SPIKE,+8:SPIKE-Synchro     % Adaptive
showing=0;                % +1:Spike Trains,+2:Distance,+4:Profile,+8:Matrix
plotting=15;               % +1:Spike Trains,+2:Distance,+4:Profile,+8:Matrix
sort_spike_trains=1;       % 0-no,1-yes

dataset=6;                % ##### 6 for testing of spike train sorting #####

if dataset==2
    tmax=100;
    num_trains=2;
    spikes=cell(1,num_trains);
    spikes{1} = [12 16 28 32 44 48 60 64 76 80];
    spikes{2} = [8 20 24 36 40 52 56 68 72 84];
elseif dataset==3
    tmax=1;
    num_trains=3;
    spikes=cell(1,num_trains);
    spikes{1} = [0.0001 0.7142];
    spikes{2} = [0.2858 0.9999];                    % Synfire Chain with second and third spike trains switched
    spikes{3} = [0.1429 0.8571];
elseif dataset==4
    num_trains=4;
    spikes=cell(1,num_trains);
    spikes{1} = [64.88600 305.81000 696.00000 800.0000];
    spikes{2} = [67.88600 302.81000 699.00000];
    spikes{3} = [164.88600 205.81000 796.00000 900.0000];
    spikes{4} = [263.76400 418.45000 997.48000];
elseif dataset==5
    tmax=1;
    num_trains=5;
    spikes=cell(1,num_trains);
    spikes{1} = [0.0001 0.2942 0.5882 0.8823];
    spikes{2} = [0.0589 0.3530 0.6470 0.9411];      % Synfire Chain with second and third as well as fourth and fifth spike trains switched
    spikes{3} = [0.0295 0.3236 0.6176 0.9117];
    spikes{4} = [0.1177 0.4118 0.7058 0.9999];
    spikes{5} = [0.0883 0.3824 0.6764 0.9705];
elseif dataset==6
    tmax=1000;
    spikes_mat=textread('Testdata6.txt');
    num_trains=size(spikes_mat,1);
    spikes=cell(1,num_trains);
    for stc=1:num_trains
        spikes{stc}=spikes_mat(stc,spikes_mat(stc,:)>0);
    end
    %spikes=spikes(1:4);
elseif dataset==7
    num_trains=7;
    spikes=cell(1,num_trains);
    spikes{1} = [64.88600 305.81000 696.00000];
    spikes{2} = 66.415;
    spikes{3} = 66.449;
    spikes{4} = 66.449;
    spikes{5} = [];
    spikes{6} = [];
    spikes{7} = [63.76400 318.45000 697.48000];
elseif dataset==40
    tmax=4000;
    spikes_mat=textread('Testdata.txt');
    num_trains=size(spikes_mat,1);
    spikes=cell(1,num_trains);
    for stc=1:num_trains
        spikes{stc}=spikes_mat(stc,spikes_mat(stc,:)>0);
    end
elseif dataset==55
    %tmin=4.5; tmax=4.8;
    tmax=613;
    spikes_mat=textread('../../Downloads/CyprianAdler/spike_times_PySpike.txt');
    num_trains=size(spikes_mat,1);
    spikes=cell(1,num_trains);
    for stc=1:num_trains
        spikes{stc}=spikes_mat(stc,spikes_mat(stc,:)>0);
    end
end
% num_trains


STS=SpikeTrainSet(spikes, tmin, tmax);

if mod(showing,2)>0
    for stc=1:num_trains
        spikes{stc}
    end
end
if mod(plotting,2)>0
    figure(1)
    STS.plotSpikeTrainSet()
end


if mod(measures,2)>0                                                        % ISI-distance
    if mod(showing,16)>1 || mod(plotting,16)>1
        isi_dist = STS.ISIdistance(tmin, tmax);
        if mod(showing,4)>1
            isi_dist
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        isi_dist_prof = STS.ISIdistanceProfile(tmin, tmax);
        x_isi_dist_prof = isi_dist_prof.PlotProfileX;
        y_isi_dist_prof = isi_dist_prof.PlotProfileY;
        if mod(showing,8)>3
            isi_dist_profile=[x_isi_dist_prof' y_isi_dist_prof']
        end
        if mod(plotting,8)>3
            figure(2)
            isi_dist_prof.Plot()
            title(['ISI-distance = ',num2str(isi_dist)])
            xlim([tmin tmax])
            ylim([0 1])
        end
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        isi_dist_mat = STS.ISIdistanceMatrix(tmin, tmax);
        if mod(showing,16)>7
            isi_dist_mat
        end
        if mod(plotting,16)>7
            figure(3)
            imagesc(isi_dist_mat)
            colormap jet
            set(gca,'XTick',1:num_trains,'YTick',1:num_trains)
            xlabel('Spike trains'); ylabel('Spike trains')
            title(['ISI-distance = ',num2str(isi_dist)])
            colorbar
        end
    end
end



if mod(measures,4)>1                                                        % SPIKE-distance
    if mod(showing,16)>1 || mod(plotting,16)>1
        spike_dist = STS.SPIKEdistance(tmin, tmax);
        if mod(showing,4)>1
            spike_dist
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        spike_dist_prof = STS.SPIKEdistanceProfile(tmin, tmax);
        x_spike_dist_prof = spike_dist_prof.PlotProfileX;
        y_spike_dist_prof = spike_dist_prof.PlotProfileY;
        if mod(showing,8)>3
            spike_dist_profile=[x_spike_dist_prof' y_spike_dist_prof']
        end
        if mod(plotting,8)>3
            figure(4)
            spike_dist_prof.Plot()
            title(['SPIKE-distance = ',num2str(spike_dist)])
            xlim([tmin tmax])
            ylim([0 1])
        end
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        spike_dist_mat = STS.SPIKEdistanceMatrix(tmin, tmax);
        if mod(showing,16)>7
            spike_dist_mat
        end
        if mod(plotting,16)>7
            figure(5)
            imagesc(spike_dist_mat)
            colormap jet
            set(gca,'XTick',1:num_trains,'YTick',1:num_trains)
            xlabel('Spike trains'); ylabel('Spike trains')
            title(['SPIKE-distance = ',num2str(spike_dist)])
            colorbar
        end
    end
end



if mod(measures,8)>3                                                        % RI-SPIKE-distance
    if mod(showing,16)>1 || mod(plotting,16)>1
        ri_spike_dist = STS.RateIndependentSPIKEdistance(tmin, tmax);
        if mod(showing,4)>1
            ri_spike_dist
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        ri_spike_dist_prof = STS.RateIndependentSPIKEdistanceProfile(tmin, tmax);
        x_ri_spike_dist_prof = ri_spike_dist_prof.PlotProfileX;
        y_ri_spike_dist_prof = ri_spike_dist_prof.PlotProfileY;
        ri_spike_dist_profile=[x_ri_spike_dist_prof' y_ri_spike_dist_prof']
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        ri_spike_dist_mat = STS.RateIndependentSPIKEdistanceMatrix(tmin, tmax)
    end
end

if mod(measures,64)>7
    if mod(showing,16)>1 || mod(plotting,16)>1
        [synchro, Sorder, STOrder, synchProfile, SorderProfile, STorderProfile] = STS.SPIKEsynchroProfile(tmin, tmax);
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        [SPIKESM, SPIKEOM, SPIKETOM]= STS.SPIKESynchroMatrix(tmin, tmax, inf);
    end
end

if mod(measures,16)>7                                                        % SPIKE-Synchro
    if mod(showing,4)>1 || mod(plotting,16)>1
        spike_sync = STS.SPIKEsynchro(tmin, tmax);
        if mod(showing,4)>1
            spike_sync
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        %[synchro, Sorder, STOrder, synchProfile, SorderProfile, STorderProfile] = STS.SPIKEsynchroProfile(tmin, tmax);
        % %num_spikes=cellfun(@length,spikes);
        %[all_spikes,sp_indy]=sort([spikes{:}]);
        %all_synchro=[synchro{:}];
        %x_spike_sync_prof = [tmin all_spikes tmax];
        %y_spike_sync_prof = all_synchro([sp_indy(1) sp_indy sp_indy(end)]);
        x_spike_sync_prof = synchProfile(2,:);
        y_spike_sync_prof = synchProfile(1,:);
        
        if mod(showing,8)>3
            spike_sync_profile=[x_spike_sync_prof' y_spike_sync_prof']
        end
        if mod(plotting,8)>3
            figure(8)
            plot(x_spike_sync_prof',y_spike_sync_prof','-*k')
            title(['SPIKE-synchronization = ',num2str(spike_sync)])
            xlim([tmin tmax])
            ylim([0 1])
        end
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        spike_sync_mat = SPIKESM;
        if mod(showing,16)>7
            spike_sync_mat
        end
        if mod(plotting,16)>7
            figure(9)
            imagesc(spike_sync_mat)
            colormap jet
            set(gca,'XTick',1:num_trains,'YTick',1:num_trains)
            xlabel('Spike trains'); ylabel('Spike trains')
            title(['SPIKE-synchronization = ',num2str(spike_sync)])
            colorbar
        end
    end
end

if mod(measures,32)>15                                                        % SPIKE-Order
    if mod(showing,4)>1 || mod(plotting,16)>1
        spike_order = 0; % mean(SorderProfile(1,:),2);           % always 0
        if mod(showing,4)>1
            spike_order
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        x_spike_order_prof = SorderProfile(2,:);
        y_spike_order_prof = SorderProfile(1,:);
        
        if mod(showing,8)>3
            spike_order_profile=[x_spike_order_prof' y_spike_order_prof']
        end
        if mod(plotting,8)>3
            figure(10)
            plot(x_spike_order_prof',y_spike_order_prof','-*k')
            title(['SPIKE-Order = ',num2str(spike_order)])
            xlim([tmin tmax])
            line([tmin tmax],zeros(1,2),'LineStyle',':')
            ylim([-1 1])
        end
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        spike_order_mat = SPIKEOM;
        if mod(showing,16)>7
            spike_order_mat
        end
        if mod(plotting,16)>7
            figure(11)
            imagesc(spike_order_mat)
            colormap jet
            set(gca,'XTick',1:num_trains,'YTick',1:num_trains)
            xlabel('Spike trains'); ylabel('Spike trains')
            title(['SPIKE-Order = ',num2str(spike_order)])
            colorbar
        end
    end
end

if mod(measures,64)>31                                                        % Spike Train Order
    if mod(showing,16)>1 || mod(plotting,16)>1
        spike_train_order = mean(STorderProfile(1,:),2);
        if mod(showing,4)>1
            spike_train_order
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        x_spike_train_order_prof = STorderProfile(2,:);
        y_spike_train_order_prof = STorderProfile(1,:);
        
        if mod(showing,8)>3
            spike_train_order_profile=[x_spike_train_order_prof' y_spike_train_order_prof']
        end
        if mod(plotting,8)>3
            figure(12)
            colormap jet
            plot(x_spike_train_order_prof',y_spike_train_order_prof','-*k')
            line([tmin tmax],zeros(1,2),'LineStyle',':')
            title(['Spike Train Order = ',num2str(spike_train_order)])
            xlim([tmin tmax])
            ylim([-1 1])
        end
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        spike_train_order_mat = SPIKETOM;
        if mod(showing,16)>7
            spike_train_order_mat
        end
        if mod(plotting,16)>7
            figure(13)
            colormap jet
            imagesc(spike_train_order_mat)
            set(gca,'XTick',1:num_trains,'YTick',1:num_trains)
            xlabel('Spike trains'); ylabel('Spike trains')
            title(['Spike Train Order = ',num2str(spike_train_order)])
            colorbar
        end
    end
end


if sort_spike_trains==1 % ############################################################################################
    num_surros=19;
    
    [initialIteration,optimalIteration,synf,so_profs,sto_profs,SpikeTrainOfASpike] = STS.SpikeTrainOrderWithSurrogates(num_surros);
    ss_profs=abs(so_profs);
    
    spike_order_plotting=1;
    if spike_order_plotting==1
        %% minimum and maximum boundaries for colour coding
        Cmin = min(min(min(initialIteration.PairwiseMatrixE)),min(min(optimalIteration.PairwiseMatrixE)));
        Cmax = max(max(max(initialIteration.PairwiseMatrixE)),max(max(optimalIteration.PairwiseMatrixE)));
        
        % Making figure at size 800x600 pixels
        FigureSize = [800 600];
        FontSize = 12;
        TickFontSize = 10;
        height = 0.13;
        PositionArray = [0.05 0.05+(1:4)*0.195];
        
        
        figure(14); clf
        %set(gcf,'Position', [200, 70, 100+FigureSize(1), 100+FigureSize(2)])
        set(gcf,'Units','normalized','Position',[0 0.0044 1.0000 0.8900])
        set(gcf,'name','SPIKE-Order')
        % Spike order profile
        sb1 = subplot('Position',[0.1 PositionArray(5) 0.7 height]);
        

        PlotColoredSpikes(initialIteration.SpikeTrains,initialIteration.SpikeOrderSpikeValues,tmin,tmax)
        
        if num_trains<=20
            set(sb1,'Ytick',(1:num_trains)-0.5,'YTickLabel', fliplr(initialIteration.Order));
        else
            cm=colormap;
            num_cols=size(cm,1);
            dcol_indy=round(num_cols:-(num_cols-1)/(num_trains-1):1);
            dcols=cm(dcol_indy,:);
            xlim([tmin-0.01*(tmax-tmin) tmax])
            coli=ceil(((initialIteration.Order-1)/(num_trains-1))*(num_cols-1))+1;
            for trc=1:num_trains
                patch([tmin-0.015*(tmax-tmin) tmin*ones(1,2) tmin-0.015*(tmax-tmin)],num_trains+1-trc-1+[0 0 1 1],...
                    dcols(initialIteration.Order(trc),:),'EdgeColor',dcols(initialIteration.Order(trc),:));
            end
            set(sb1,'Ytick',([1 num_trains])-0.5,'YTickLabel',fliplr(initialIteration.Order([1 end])));
        end
        
        title('Spike trains','FontSize',FontSize)
        box on
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',TickFontSize)
        %Adding colorbar
        axes('Position', [0.81 PositionArray(5) 0.01 height] ,'Visible', 'off');
        c = colorbar;
        % Adjusting colorbar width
        cpos = c.Position;
        cpos(3) = 0.03;
        c.Position = cpos;
        % Color bar range
        caxis([-1 1])
        % Adding D to indicate value range
        xl = xlim;
        yl = ylim;
        right = 7;
        up = 0.7;
        text(xl(2)+(xl(2)-xl(1))*right, (yl(2)-yl(1))*up,'D')
        
        % Spike Train Order profile
        sb2 = subplot('Position',[0.1 PositionArray(4) 0.7 height]);
        plot(initialIteration.TimeProfileE(2,:),initialIteration.TimeProfileE(1,:),'-ok','MarkerFaceColor','black')
        hold on;

        %Plotting the channel
        plot(initialIteration.SynchronizationProfile(2,:),initialIteration.SynchronizationProfile(1,:),'--k','MarkerFaceColor','black')
        plot(initialIteration.SynchronizationProfile(2,:),-initialIteration.SynchronizationProfile(1,:),'--k','MarkerFaceColor','black')
        if num_trains<20
            xlim([tmin tmax])
        else
            xlim([tmin-0.01*(tmax-tmin) tmax])
        end
        ylim([-1.15, 1.15])
        title('Time profile E','FontSize',FontSize)
        xl = xlim;
        yl = ylim;
        text(((xl(2)-xl(1))*1.05),yl(1)+(yl(2)-yl(1))*0.8,['C=' ,num2str(initialIteration.SynchronizationValueC, '%.3f')],'FontSize',FontSize)
        text(((xl(2)-xl(1))*1.05),yl(1)+(yl(2)-yl(1))*0.3,['F_u=',num2str(initialIteration.SynfireIndicatorF, '%.3f')],'FontSize',FontSize)
        hold on;
        plot([xl(1) xl(2)],[0 0], ':k')
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',TickFontSize)
        
        % D matrix for original
        sb3 = subplot('Position',[0.11 PositionArray(3) height height]);
        imagesc(initialIteration.PairwiseMatrixE)
        caxis([Cmin Cmax]);
        axis square
        set(gca,'YDir','reverse')
        title('Pairwise Matrix D','FontSize',FontSize)
        if num_trains<20
            set(sb3,'Xtick',1:num_trains,'XTickLabel', 1:num_trains);
            set(sb3,'Ytick',1:num_trains,'YTickLabel', 1:num_trains);
        end
        %Form separator line(zig-zag)
        for trc=1:num_trains-1
            line((trc+0.5)*ones(1,2),[trc-0.5 trc+0.5],'Color','k','LineWidth',2)
            line([trc+0.5 trc+1.5],(trc+0.5)*ones(1,2),'Color','k','LineWidth',2)
        end
        line((num_trains+0.5)*ones(1,2),[0.5 num_trains-0.5],'Color','k','LineWidth',2)
        line([1.5 num_trains+0.5],0.5*ones(1,2),'Color','k','LineWidth',2)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',TickFontSize)
        colormap(sb3,jet(100));
        
        % Sorted D matrix
        sb4 = subplot('Position',[0.27 PositionArray(3) height height]);
        imagesc(optimalIteration.PairwiseMatrixE)
        caxis([Cmin Cmax]);
        colormap(jet(100));
        axis square
        set(gca,'YDir','reverse')
        title('Sorted pairwise Matrix D','FontSize',FontSize)
        if num_trains<20
            set(sb4,'Xtick',1:num_trains,'XTickLabel', 1:num_trains);
            set(sb4,'Ytick',1:num_trains,'YTickLabel', 1:num_trains);
        end
        %Form separator line(zig-zag)
        for trc=1:num_trains-1
            line((trc+0.5)*ones(1,2),[trc-0.5 trc+0.5],'Color','k','LineWidth',2)
            line([trc+0.5 trc+1.5],(trc+0.5)*ones(1,2),'Color','k','LineWidth',2)
        end
        line((num_trains+0.5)*ones(1,2),[0.5 num_trains-0.5],'Color','k','LineWidth',2)
        line([1.5 num_trains+0.5],0.5*ones(1,2),'Color','k','LineWidth',2)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',TickFontSize)
        colormap(sb4,jet(100));
        
        % Colorbar for the matrices
        cb = axes('Position', [0.33 PositionArray(3) height height], 'Visible', 'off');
        c = colorbar();
        cpos = c.Position;
        cpos(3) = 0.03;
        c.Position = cpos;
        caxis([Cmin Cmax])
        colormap(cb,jet(100));
        
        % Synfire indicator graph
        if num_surros>0
            sb5 = subplot('Position',[0.60 PositionArray(3) height height]);
            histogram(synf,100);
            set(get(gca,'child'),'FaceColor','red','EdgeColor','r');
            yl = ylim;
            x = optimalIteration.SynfireIndicatorF;
            hold on;
            xlabel('F')
            axis square
            % Adding standard deviation etc
            ini = initialIteration.SynfireIndicatorF;
            [~,b]=sort([ini synf]);
            posi=num_surros+2-find(b==1);
            pval=posi/(num_surros+1);
            mean_surros=mean(synf);
            min_surros=min(synf);
            max_surros=max(synf);
            min_val=min([ini min_surros]);
            max_val=max([ini max_surros]);
            abs_max_val=max(abs([ini max_surros]));
            std_surros=std(synf);
            if std_surros>0
                z_score=(ini-mean_surros)/std_surros;
            else
                z_score=0;
            end
            line(mean_surros*ones(1,2),yl,'Color','r','LineWidth',3)
            line(mean_surros+[-std_surros std_surros],(yl(1)+0.65*(yl(2)-yl(1)))*ones(1,2),'Color','r','LineWidth',3)
            line((mean_surros-std_surros)*ones(1,2),yl(1)+[0.6 0.7]*(yl(2)-yl(1)),'Color','r','LineWidth',3)
            line((mean_surros+std_surros)*ones(1,2),yl(1)+[0.6 0.7]*(yl(2)-yl(1)),'Color','r','LineWidth',3)
            if pval<=0.05
                title(['z = ',num2str(z_score,3),'  ;  p = ',num2str(pval,2),'**'],'FontSize',FontSize,'FontWeight','bold')
            elseif pval<0.3
                title(['z = ',num2str(z_score,3),'  ;  p > 0.05'],'FontSize',FontSize,'FontWeight','bold')
            else
                title(['z = ',num2str(z_score,3),'  ;  p >> 0.05'],'FontSize',FontSize,'FontWeight','bold')
            end
            plot([x x],[yl(1) yl(2)],'--k','linewidth',1.5)
            a = get(gca,'XTickLabel');
            set(gca,'XTickLabel',a,'fontsize',TickFontSize)
        end
        
        % Time profile E for sorted spike trains
        sb6 = subplot('Position',[0.1 PositionArray(2) 0.7 height]);
        plot(optimalIteration.TimeProfileE(2,:),optimalIteration.TimeProfileE(1,:),'-ok','MarkerFaceColor','black')
        hold on
        if num_trains<20
            xlim([tmin tmax])
        else
            xlim([tmin-0.01*(tmax-tmin) tmax])
        end
        ylim([-1.15, 1.15])
        xl = xlim;
        yl = ylim;
        text(((xl(2)-xl(1))*1.05),yl(1)+(yl(2)-yl(1))*0.8,['C=' ,num2str(optimalIteration.SynchronizationValueC, '%.3f')],'FontSize',FontSize)
        text(((xl(2)-xl(1))*1.05),yl(1)+(yl(2)-yl(1))*0.3,['F_s=',num2str(optimalIteration.SynfireIndicatorF, '%.3f')],'FontSize',FontSize)
        hold on;
        % middle line
        plot([xl(1) xl(2)],[0 0], ':k')
        % Channel
        plot(optimalIteration.SynchronizationProfile(2,:),optimalIteration.SynchronizationProfile(1,:),'--k','MarkerFaceColor','black')
        plot(optimalIteration.SynchronizationProfile(2,:),-optimalIteration.SynchronizationProfile(1,:),'--k','MarkerFaceColor','black')
        title('Time profile E for sorted spike trains','FontSize',FontSize)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',TickFontSize)
        
        % Sorted spike trains
        sb7 = subplot('Position',[0.1 PositionArray(1) 0.7 height]);
        

        PlotColoredSpikes(optimalIteration.SpikeTrains,optimalIteration.SpikeOrderSpikeValues,tmin,tmax)
        
        if num_trains<=20
            set(sb7,'Ytick', (1:num_trains)-0.5,'YTickLabel',fliplr(optimalIteration.Order));
        else
            xlim([tmin-0.01*(tmax-tmin) tmax])
            coli=ceil(((optimalIteration.Order-1)/(num_trains-1))*(num_cols-1))+1;
            for trc=1:num_trains
                patch([tmin-0.015*(tmax-tmin) tmin*ones(1,2) tmin-0.015*(tmax-tmin)],num_trains+1-trc-1+[0 0 1 1],...
                    dcols(optimalIteration.Order(trc),:),'EdgeColor',dcols(optimalIteration.Order(trc),:));
            end
            set(sb7,'Ytick',([1 num_trains])-0.5,'YTickLabel',fliplr(initialIteration.Order([1 end])));
        end
        title('Sorted spike trains','FontSize',FontSize)
        box on
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'FontSize',TickFontSize)
        %Adding colorbar
        axes('Position', [0.81 PositionArray(1) 0.01 height] ,'Visible', 'off');
        c = colorbar;
        % Adjusting colorbar width
        cpos = c.Position;
        cpos(3) = 0.03;
        c.Position = cpos;
        % Color bar range
        caxis([-1 1])
        % Adding D to indicate value range
        xl = xlim;
        yl = ylim;
        right = 7;
        up = 0.7;
        text(xl(2)+(xl(2)-xl(1))*right, (yl(2)-yl(1))*up,'D')
    end
end












% ################ Adaptive versions, not needed right now ###############

if mod(adaptive_measures,2)>0                                                % A-ISI-distance
    if mod(showing,16)>1 || mod(plotting,16)>1
        % isi_dist = STS.ISIdistance(tmin, tmax)                           % the same as threshold = 0
        % adaptive_isi_dist_thr_0 = STS.AdaptiveISIdistance(tmin, tmax, 0)   % falls back to original ISI-distance
        adaptive_isi_dist_thr = STS.AdaptiveISIdistance(tmin, tmax, threshold);
        adaptive_isi_dist_auto = STS.AdaptiveISIdistance(tmin, tmax);
        auto_threshold = STS.giveTHR();
        % adaptive_isi_dist_auto_extr_thr = STS.AdaptiveISIdistance(tmin, tmax, threshold)
        if mod(showing,4)>1
            adaptive_isi_dist_thr
            adaptive_isi_dist_auto
            auto_threshold
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        adaptive_isi_dist_prof = STS.AdaptiveISIdistanceProfile(tmin, tmax, threshold);
        x_adaptive_isi_dist_prof = adaptive_isi_dist_prof.PlotProfileX;
        y_adaptive_isi_dist_prof = adaptive_isi_dist_prof.PlotProfileY;
        if mod(showing,8)>3
            adaptive_isi_dist=[x_adaptive_isi_dist_prof' y_adaptive_isi_dist_prof']
        end
        if mod(plotting,8)>3
            figure(101)
            adaptive_isi_dist_prof.Plot()
            title(['Adaptive ISI-distance = ',num2str(adaptive_isi_dist_thr)])
            xlim([tmin tmax])
            ylim([0 1])
        end
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        adaptive_isi_dist_mat = STS.AdaptiveISIdistanceMatrix(tmin, tmax, threshold);
        if mod(showing,16)>7
            adaptive_isi_dist_mat
        end
        if mod(plotting,16)>7
            figure(102)
            imagesc(adaptive_isi_dist_mat)
            colormap jet
            set(gca,'XTick',1:num_trains,'YTick',1:num_trains)
            xlabel('Spike trains'); ylabel('Spike trains')
            title(['Adaptive ISI-distance = ',num2str(adaptive_isi_dist_thr)])
            colorbar
        end
    end
end



if mod(adaptive_measures,4)>1                                                % A-SPIKE-distance
    if mod(showing,16)>1 || mod(plotting,16)>1
        adaptive_spike_dist_thr = STS.AdaptiveSPIKEdistance(tmin, tmax, threshold);
        adaptive_spike_dist_auto = STS.AdaptiveSPIKEdistance(tmin, tmax);
        if mod(showing,4)>1
            adaptive_spike_dist_thr
            adaptive_spike_dist_auto
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        adaptive_spike_dist_prof = STS.AdaptiveSPIKEdistanceProfile(tmin, tmax, threshold);
        x_adaptive_spike_dist_prof = adaptive_spike_dist_prof.PlotProfileX;
        y_adaptive_spike_dist_prof = adaptive_spike_dist_prof.PlotProfileY;
        if mod(showing,8)>3
            adaptive_spike_dist=[x_adaptive_spike_dist_prof' y_adaptive_spike_dist_prof']
        end
        if mod(plotting,8)>3
            figure(103)
            adaptive_spike_dist_prof.Plot()
            title(['Adaptive SPIKE-distance = ',num2str(adaptive_spike_dist_thr)])
            xlim([tmin tmax])
            ylim([0 1])
        end
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        adaptive_spike_dist_mat = STS.AdaptiveSPIKEdistanceMatrix(tmin, tmax, threshold);
        if mod(showing,16)>7
            adaptive_spike_dist_mat
        end
        if mod(plotting,16)>7
            figure(104)
            imagesc(adaptive_spike_dist_mat)
            colormap jet
            set(gca,'XTick',1:num_trains,'YTick',1:num_trains)
            xlabel('Spike trains'); ylabel('Spike trains')
            title(['Adaptive SPIKE-distance = ',num2str(adaptive_spike_dist_thr)])
            colorbar
        end
    end
end



if mod(adaptive_measures,8)>3                                                % A-RI-SPIKE-distance
    if mod(showing,16)>1 || mod(plotting,16)>1
        adaptive_ri_spike_dist_thr = STS.AdaptiveRateIndependentSPIKEdistance(tmin, tmax, threshold);
        adaptive_ri_spike_dist_auto = STS.AdaptiveRateIndependentSPIKEdistance(tmin, tmax);
        if mod(showing,4)>1
            adaptive_ri_spike_dist_thr
            adaptive_ri_spike_dist_auto
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        adaptive_ri_spike_dist_prof = STS.AdaptiveRateIndependentSPIKEdistanceProfile(tmin, tmax, threshold);
        x_adaptive_ri_spike_dist_prof = adaptive_ri_spike_dist_prof.PlotProfileX;
        y_adaptive_ri_spike_dist_prof = adaptive_ri_spike_dist_prof.PlotProfileY;
        adaptive_ri_spike_dist=[x_adaptive_ri_spike_dist_prof' y_adaptive_ri_spike_dist_prof']
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        adaptive_ri_spike_dist_mat = STS.AdaptiveRateIndependentSPIKEdistanceMatrix(tmin, tmax, threshold)
    end
end



if mod(adaptive_measures,16)>7                                                % A-SPIKE-Synchro
    if mod(showing,16)>1 || mod(plotting,16)>1
        adaptive_spike_sync_thr = STS.AdaptiveSPIKEsynchro(tmin, tmax, threshold);
        adaptive_spike_sync_auto = STS.AdaptiveSPIKEsynchro(tmin, tmax);
        if mod(showing,4)>1
            adaptive_spike_sync_thr
            adaptive_spike_sync_auto
        end
    end
    if mod(showing,8)>3 || mod(plotting,8)>3
        [adaptive_synchro, adaptive_Sorder, adaptive_STOrder] = STS.AdaptiveSPIKESynchroProfile(tmin, tmax, threshold);
        num_spikes=cellfun(@length,spikes);
        [all_spikes,sp_indy]=sort([spikes{:}]);
        all_adaptive_synchro=[adaptive_synchro{:}];
        x_adaptive_spike_sync_prof = [tmin all_spikes tmax];
        y_adaptive_spike_sync_prof = all_adaptive_synchro([sp_indy(1) sp_indy sp_indy(end)]);
        if mod(showing,8)>3
            adaptive_spike_sync=[x_adaptive_spike_sync_prof' y_adaptive_spike_sync_prof']
        end
        if mod(plotting,8)>3
            figure(105)
            plot(x_adaptive_spike_sync_prof',y_adaptive_spike_sync_prof','-*k')
            title(['Adaptive SPIKE-synchronization = ',num2str(adaptive_spike_sync_thr)])
            xlim([tmin tmax])
            ylim([0 1])
        end
    end
    if mod(showing,16)>7 || mod(plotting,16)>7
        [adaptive_SPIKESM, adaptive_SPIKEOM, adaptive_normSPIKEOM]= STS.AdaptiveSPIKESynchroMatrix(tmin, tmax, threshold, inf);
        adaptive_spike_sync_mat = adaptive_SPIKESM;
        if mod(showing,16)>7
            adaptive_spike_sync_mat
        end
        if mod(plotting,16)>7
            figure(106)
            imagesc(adaptive_spike_sync_mat)
            colormap jet
            set(gca,'XTick',1:num_trains,'YTick',1:num_trains)
            xlabel('Spike trains'); ylabel('Spike trains')
            title(['Adaptive SPIKE-synchronization = ',num2str(adaptive_spike_sync_thr)])
            colorbar
        end
    end
end


if mod(adaptive_measures,32)>15                                                % A-SPIKE-Order
end

if mod(adaptive_measures,64)>31                                                % A-Spike-Train-Order
end