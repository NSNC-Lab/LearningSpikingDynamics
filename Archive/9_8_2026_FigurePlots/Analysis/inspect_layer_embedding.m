function [fig,cellTable]=inspect_layer_embedding(mapID,visible)
% Explore the SAME coordinates colored by layer, subject, or parameter.
% Click points with the Data Tips tool to recover original cell identities.
root=fileparts(mfilename('fullpath'));
v=load(fullfile(root,'embedding_layer_results','embedding_analysis.mat'),'results'); r=v.results;
if nargin<1 || isempty(mapID), mapID=r.bestMaps(2); end
if nargin<2, visible=true; end
assert(mapID>=1 && mapID<=numel(r.maps));
m=r.maps(mapID); cellTable=r.cellTable;
vis='off'; if visible, vis='on'; end
fig=figure('Visible',vis,'Color','w','Position',[60 60 1500 930]);
t=tiledlayout(fig,3,5,'TileSpacing','compact','Padding','compact');
colors=[.15 .45 .75;.9 .48 .15;.3 .64 .42;.5 .5 .5];
ax=nexttile(t); hold(ax,'on'); [~,labels]=ismember(cellTable.Layer,r.classes); labels(labels==0)=4;
hs=gobjects(4,1);
for k=1:4
    ix=labels==k; hs(k)=scatter(ax,m.Y(ix,1),m.Y(ix,2),20,colors(k,:),'filled');
    add_embedding_datatips(hs(k),cellTable(ix,:));
end
title(ax,'Cortical layer'); legend(ax,hs,[r.classes;"Unknown"],'Location','best','FontSize',7);
ax=nexttile(t); [~,~,subjectID]=unique(cellTable.Subject);
s=scatter(ax,m.Y(:,1),m.Y(:,2),20,subjectID,'filled');
add_embedding_datatips(s,cellTable); colormap(ax,hsv(max(subjectID))); colorbar(ax);
title(ax,'Subject (categorical index)');
for p=1:numel(r.dataset.names)
    ax=nexttile(t); values=r.dataset.X(:,p);
    s=scatter(ax,m.Y(:,1),m.Y(:,2),20,values,'filled');
    add_embedding_datatips(s,cellTable);
    s.DataTipTemplate.DataTipRows(end+1)=dataTipTextRow(r.dataset.names{p},values);
    limits=prctile(values,[2 98]); if limits(2)>limits(1), clim(ax,limits); end
    colormap(ax,parula); colorbar(ax);
    title(ax,strrep(r.dataset.names{p},'_',' '));
end
ax=nexttile(t); axis(ax,'off');
text(ax,0,1,{'Point coordinates are identical across panels.',...
    'Use Data Tips to identify cells.',...
    'Parameter colors clip at 2nd/98th percentiles.',...
    'Tables/data tips retain original values.',...
    sprintf('Map ID: %d | seed: %d',mapID,m.seed),...
    sprintf('Preprocessing: %d (1=z-score, 2=asinh)',m.preprocessing)},...
    'VerticalAlignment','top','FontSize',9,'Interpreter','none');
for ax=findall(fig,'Type','axes')'
    if ~isempty(ax.Children) && ~strcmp(ax.Visible,'off')
        xticks(ax,[]); yticks(ax,[]); box(ax,'off');
    end
end
warningText='Unsupervised geometry; color labels did not create this map.';
if m.supervised, warningText='Layer-guided geometry: layer labels CREATED some of this separation.'; end
title(t,{sprintf('%s — map %d: trace back to original cells and parameters',m.method,mapID),warningText});
end
