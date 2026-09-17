function selected=trace_embedding_region(mapID)
% Select any island/region and return its original cell rows and parameters.
root=fileparts(mfilename('fullpath'));
v=load(fullfile(root,'embedding_layer_results','embedding_analysis.mat'),'results'); r=v.results;
if nargin<1, mapID=r.bestMaps(2); end
m=r.maps(mapID); fig=figure('Color','w'); ax=axes(fig);
[~,labels]=ismember(r.cellTable.Layer,r.classes); labels(labels==0)=4;
palette=[.15 .45 .75;.9 .48 .15;.3 .64 .42;.5 .5 .5];
s=scatter(ax,m.Y(:,1),m.Y(:,2),35,palette(labels,:),'filled');
add_embedding_datatips(s,r.cellTable);
title(ax,{'Click polygon vertices around a group; press Enter when done',...
    sprintf('%s | map %d',m.method,mapID)});
[x,y]=ginput;
if numel(x)<3, selected=r.cellTable([],:); return; end
inside=inpolygon(m.Y(:,1),m.Y(:,2),x,y);
selected=r.cellTable(inside,:);
hold(ax,'on'); plot(ax,[x;x(1)],[y;y(1)],'k-');
scatter(ax,m.Y(inside,1),m.Y(inside,2),70,'k');
text(ax,m.Y(inside,1),m.Y(inside,2)," "+string(selected.CellID),'FontSize',8);
disp(selected);
end
