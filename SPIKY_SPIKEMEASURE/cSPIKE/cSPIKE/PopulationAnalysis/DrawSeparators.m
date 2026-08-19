function DrawSeparators(NROStimuli,NROrepeats,linewid)
totalWidth = NROStimuli*NROrepeats;
totalHeight = totalWidth;
separators = (1:NROStimuli-1)*NROrepeats;
hold on;
for i = 1:size(separators,2)
    plot([0,totalWidth]+0.5,[separators(i),separators(i)]+0.5,'color','k','linewidth',linewid)
    plot([separators(i),separators(i)]+0.5,[0,totalHeight]+0.5,'color','k','linewidth',linewid)
end
hold off;


end

