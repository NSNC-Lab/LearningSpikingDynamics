function plotSpikeTrain( spiketimes, height, center, colour , width)

hold on;
for i = 1:length(spiketimes)

    plot([spiketimes(i) spiketimes(i)],[center-height/2 center+height/2],'Color',colour,'LineWidth',width);

end

end

