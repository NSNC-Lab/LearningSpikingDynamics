function validate_layer_analysis(outdir)
% Independent checks against MATLAB's built-in LDA and saved row alignment.
if nargin<1, outdir=fullfile(fileparts(mfilename('fullpath')),'layer_parameter_results'); end
v=load(fullfile(outdir,'layer_analysis.mat'),'results'); r=v.results;
d=r.dataset; X=d.X(r.valid,:); [~,y]=ismember(d.layer(r.valid),r.classes);
assert(isequal(d.cellID,(1:size(d.X,1))'));
[~,best]=min(d.distance,[],2); assert(isequal(best,d.selection));
src=load(d.simfile,'params'); labels=load(d.datafile,'all_data');
for j=1:numel(d.names)
    A=src.params.(d.names{j});
    for k=1:size(d.X,1)
        assert(isequaln(d.X(k,j),A(d.selection(k),k,end)));
        assert(d.layer(k)==string(labels.all_data(k).layer));
    end
end
for m=1:numel(r.models)
    S=project_layer_parameters(r.models{m},d.X);
    assert(max(abs(S-r.scores{m}),[],'all')<1e-10);
end
agreement=nan(numel(r.seeds),1); nativeBA=agreement;
for rep=1:numel(r.seeds)
    rng(r.seeds(rep),'twister'); c=cvpartition(y,'KFold',5); pr=nan(size(y));
    for f=1:5
        tr=training(c,f); te=test(c,f);
        mu=mean(X(tr,:),1); sd=std(X(tr,:),0,1); sd(sd<eps)=1;
        mdl=fitcdiscr((X(tr,:)-mu)./sd,y(tr),'DiscrimType','linear','Prior','uniform');
        pr(te)=predict(mdl,(X(te,:)-mu)./sd);
    end
    agreement(rep)=mean(pr==r.predictions(:,rep,2));
    nativeBA(rep)=mean(arrayfun(@(k)mean(pr(y==k)==k),1:3));
end
T=table((1:numel(r.seeds))',agreement,nativeBA,r.repeatBA(:,2),...
    'VariableNames',{'Repeat','PredictionAgreement','MATLAB_LDA_BA','Custom_LDA_BA'});
disp(T); assert(all(agreement>.99),'Custom LDA differs materially from built-in LDA.');
writetable(T,fullfile(outdir,'validation_builtin_LDA.csv'));
fprintf('PASS: cell/parameter alignment, selected batches, saved projections, and built-in LDA agreement.\n');
end
