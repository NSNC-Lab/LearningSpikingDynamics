function add_embedding_datatips(scatterHandle,T)
% Preserve source metadata directly on every plotted point.
scatterHandle.UserData=T;
rows=[dataTipTextRow('Cell ID',T.CellID),...
    dataTipTextRow('Layer',T.Layer),dataTipTextRow('Subject',T.Subject),...
    dataTipTextRow('Selected batch',T.SelectedBatch)];
scatterHandle.DataTipTemplate.DataTipRows=rows;
end
