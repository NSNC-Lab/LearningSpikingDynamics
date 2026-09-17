function scores = project_layer_parameters(model,X)
% Apply a saved full-data projection to new parameter rows in dataset.names order.
if model.method==4, X=asinh(X./model.transformScale); end
F=(X-model.mu)./model.sd;
if model.method==5
    F=exp(-pdist2(F,model.centers,'squaredeuclidean')/(2*model.width^2));
end
scores=(F-model.featureMu)*model.W;
end
