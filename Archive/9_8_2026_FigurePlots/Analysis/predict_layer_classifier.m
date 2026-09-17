function [labelIndex,scores] = predict_layer_classifier(model,Xnew)
% Xnew: rows=cells, columns=results.dataset.names, original parameter units.
% Map output indices to names with results.classes(labelIndex).
assert(size(Xnew,2)==numel(model.mu),'Unexpected number of parameter columns.');
assert(all(isfinite(Xnew),'all'),'New parameter values must be finite.');
if model.family==2, Xnew=asinh(Xnew./model.transformScale); end
[labelIndex,scores]=predict(model.classifier,(Xnew-model.mu)./model.sd);
end
