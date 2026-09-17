function results = explore_layer_parameter_space(repo,outdir,nPerm)
% Explore two-dimensional spaces that predict cortical layer from parameters.
% Run: results = explore_layer_parameter_space;
% Requires Statistics and Machine Learning Toolbox and the repository cSPIKE.
% All scaling, transforms, landmarks, and supervised projections are refit
% within training folds. Hyperparameters use inner CV only. Three known layers
% are the primary target; unknown layer is projected after fitting.
% Predictions are out-of-fold; full-data scatter plots are descriptive fits.
if nargin<1 || isempty(repo)
    repo = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics';
end
if nargin<2 || isempty(outdir), outdir=fullfile(fileparts(mfilename('fullpath')),'layer_parameter_results'); end
if nargin<3, nPerm=99; end
if ~isfolder(outdir), mkdir(outdir); end
diary(fullfile(outdir,'analysis_log.txt')); cleanup=onCleanup(@()diary('off'));
rng(20260914,'twister');
dataset=prepare_layer_parameters(repo,outdir);
classes=["L2/3";"L4";"L5/6"];
[known,yall]=ismember(dataset.layer,classes);
valid=known & all(isfinite(dataset.X),2);
X=dataset.X(valid,:); y=yall(valid); subjects=dataset.subject(valid);
K=numel(classes); counts=accumarray(y,1,[K 1]);
fprintf('Known finite cells: %d; excluded unknown: %d; excluded nonfinite known: %d\n',sum(valid),sum(~known),sum(known & ~valid));
disp(table(classes,counts,'VariableNames',{'Layer','N'}));
methods=["PCA 2D";"LDA 2D";"Shrinkage LDA 2D";"Asinh shrinkage LDA 2D";"RBF Fisher 2D"];
% A small declared search; do not choose a map by its visual separation.
configs={ [0 0], [0 0], [0 0;.25 0;.75 0;1 0], ...
    [0 0;.25 0;.75 0;1 0], [.25 .5;.75 .5;.25 1;.75 1;.25 2;.75 2] };
nRep=5; nFold=5; seeds=20260914+(0:nRep-1);
pred=nan(numel(y),nRep,numel(methods)); repeatBA=nan(nRep,numel(methods));
repeatAcc=repeatBA; choices=cell(nRep,numel(methods));
for r=1:nRep
    folds=stratified_folds(y,nFold,seeds(r));
    for m=1:numel(methods)
        [pred(:,r,m),choices{r,m}]=evaluate(X,y,folds,m,configs{m},seeds(r));
        repeatBA(r,m)=balanced_accuracy(y,pred(:,r,m),K);
        repeatAcc(r,m)=mean(y==pred(:,r,m));
        fprintf('Repeat %d: %-25s BA %.3f accuracy %.3f\n',r,methods(m),repeatBA(r,m),repeatAcc(r,m));
    end
end
results=struct('dataset',dataset,'valid',valid,'classes',classes,'methods',methods,...
    'configs',{configs},'predictions',pred,'repeatBA',repeatBA,'repeatAccuracy',repeatAcc,...
    'choices',{choices},'seeds',seeds,'counts',counts);
results.summary=table(methods,mean(repeatBA,1)',std(repeatBA,0,1)',mean(repeatAcc,1)',...
    'VariableNames',{'Method','BalancedAccuracy','SplitSD','Accuracy'});
disp(results.summary);
save(fullfile(outdir,'layer_analysis.mat'),'results');
writetable(results.summary,fullfile(outdir,'method_comparison.csv'));
% Each subject is held out entirely, to assess sensitivity to shared animals.
u=unique(subjects); groupedPred=nan(numel(y),numel(methods));
if numel(u)>2 && ~any(ismissing(subjects))
    [~,~,gf]=unique(subjects);
    for m=1:numel(methods)
        groupedPred(:,m)=evaluate(X,y,gf,m,configs{m},20261001,subjects);
    end
end
results.subjectPredictions=groupedPred;
% Label-free selection sensitivity: median across batches and initial values
% of final-selected batches. These are controls, not independent test data.
results.sensitivity=nan(2,numel(methods));
sources={dataset.Xinitial(valid,:),dataset.Xmedian(valid,:)};
for s=1:2
    for m=1:numel(methods)
        pr=evaluate(sources{s},y,stratified_folds(y,nFold,seeds(1)),m,configs{m},seeds(1));
        results.sensitivity(s,m)=balanced_accuracy(y,pr,K);
    end
end
% Full-data maps and loadings; their separation is explicitly training-fit.
models=cell(numel(methods),1); scores=cell(size(models));
for m=1:numel(methods)
    config=choose_config(X,y,m,configs{m},stratified_folds(y,5,seeds(1)));
    models{m}=fit_projection(X,y,m,config);
    scores{m}=project(models{m},dataset.X);
end
results.models=models; results.scores=scores;
save(fullfile(outdir,'layer_analysis.mat'),'results');
make_figures(results,outdir);
% Family-wise permutation comparison: rerun tuning and all five methods.
% Compare to the SAME first-repeat CV statistic, not the five-repeat average.
results.permutationBA=nan(nPerm,numel(methods));
for p=1:nPerm
    rng(72000+p,'twister'); yp=y(randperm(numel(y)));
    folds=stratified_folds(yp,nFold,seeds(1));
    for m=1:numel(methods)
        pr=evaluate(X,yp,folds,m,configs{m},seeds(1));
        results.permutationBA(p,m)=balanced_accuracy(yp,pr,K);
    end
    if mod(p,10)==0 || p==nPerm
        fprintf('Permutation %d/%d complete\n',p,nPerm);
        save(fullfile(outdir,'layer_analysis.mat'),'results');
    end
end
if nPerm>0
    results.permutationP=(1+sum(results.permutationBA>=repeatBA(1,:),1))/(nPerm+1);
    results.familywiseP=(1+sum(max(results.permutationBA,[],2)>=max(repeatBA(1,:))))/(nPerm+1);
end
save(fullfile(outdir,'layer_analysis.mat'),'results');
write_report(results,outdir);
fprintf('Analysis complete. Results: %s\n',outdir);
end

function folds=stratified_folds(y,nfold,seed)
rng(seed,'twister'); c=cvpartition(y,'KFold',nfold); folds=zeros(size(y));
for f=1:nfold, folds(test(c,f))=f; end
end

function [pred,chosen]=evaluate(X,y,folds,method,configs,seed,groups)
pred=nan(size(y)); fs=unique(folds); chosen=nan(numel(fs),2);
for f=1:numel(fs)
    tr=folds~=fs(f); te=~tr;
    if numel(unique(y(tr)))<numel(unique(y)), continue; end
    if nargin>=7
        [~,~,inner]=unique(groups(tr));
    else
        inner=stratified_folds(y(tr),3,seed+f);
    end
    chosen(f,:)=choose_config(X(tr,:),y(tr),method,configs,inner);
    model=fit_projection(X(tr,:),y(tr),method,chosen(f,:));
    pred(te)=classify_scores(model,project(model,X(te,:)));
end
end

function best=choose_config(X,y,method,configs,folds)
if size(configs,1)==1, best=configs(1,:); return; end
scores=nan(size(configs,1),1); K=numel(unique(y));
for c=1:size(configs,1)
    pred=nan(size(y));
    for f=unique(folds)'
        tr=folds~=f; te=~tr;
        if numel(unique(y(tr)))<K, continue; end
        model=fit_projection(X(tr,:),y(tr),method,configs(c,:));
        pred(te)=classify_scores(model,project(model,X(te,:)));
    end
    scores(c)=balanced_accuracy(y,pred,K);
end
[~,idx]=max(scores); best=configs(idx,:);
end

function model=fit_projection(X,y,method,config)
model.method=method; model.config=config;
model.transformScale=ones(1,size(X,2));
if method==4
    model.transformScale=median(abs(X),1);
    bad=model.transformScale<eps;
    backup=std(X,0,1); backup(backup<eps)=1;
    model.transformScale(bad)=backup(bad);
    X=asinh(X./model.transformScale);
end
model.mu=mean(X,1); model.sd=std(X,0,1); model.sd(model.sd<eps)=1;
F=(X-model.mu)./model.sd;
if method==5
    % Twenty-four training-only RBF landmarks, selected without layer labels.
    nc=min(24,size(F,1)); idx=zeros(nc,1);
    [~,idx(1)]=min(sum(F.^2,2)); mind=inf(size(F,1),1);
    for j=2:nc
        mind=min(mind,sum((F-F(idx(j-1),:)).^2,2));
        mind(idx(1:j-1))=-inf; [~,idx(j)]=max(mind);
    end
    model.centers=F(idx,:); d=pdist(F); d=d(d>0);
    model.width=config(2)*median(d); if isempty(d), model.width=1; end
    F=exp(-pdist2(F,model.centers,'squaredeuclidean')/(2*model.width^2));
end
model.featureMu=mean(F,1); F=F-model.featureMu;
K=max(y); n=size(F,1); p=size(F,2);
if method==1
    [~,~,V]=svd(F,'econ'); W=V(:,1:2);
else
    Sw=zeros(p); Sb=zeros(p); overall=mean(F,1);
    for k=1:K
        A=F(y==k,:); mu=mean(A,1); E=A-mu;
        Sw=Sw+E'*E; delta=mu-overall; Sb=Sb+size(A,1)*(delta'*delta);
    end
    Sw=Sw/(n-K); Sb=Sb/n;
    scale=max(trace(Sw)/p,eps);
    reg=(1-config(1))*Sw+config(1)*diag(diag(Sw))+1e-6*scale*eye(p);
    R=chol(reg); B=R'\Sb/R; B=(B+B')/2;
    [V,e]=eig(B,'vector'); [~,order]=sort(real(e),'descend');
    W=R\real(V(:,order(1:2)));
end
% Fix signs for repeatable full-data views, without changing classification.
for j=1:2, [~,ix]=max(abs(W(:,j))); if W(ix,j)<0, W(:,j)=-W(:,j); end; end
model.W=W; S=F*W; model.centroids=zeros(K,2); residual=zeros(size(S));
for k=1:K
    model.centroids(k,:)=mean(S(y==k,:),1);
    residual(y==k,:)=S(y==k,:)-model.centroids(k,:);
end
C=residual'*residual/(n-K);
model.invCov=pinv(C+1e-8*max(trace(C)/2,eps)*eye(2));
end

function S=project(model,X)
if model.method==4, X=asinh(X./model.transformScale); end
F=(X-model.mu)./model.sd;
if model.method==5
    F=exp(-pdist2(F,model.centers,'squaredeuclidean')/(2*model.width^2));
end
S=(F-model.featureMu)*model.W;
end

function pred=classify_scores(model,S)
K=size(model.centroids,1); d=zeros(size(S,1),K);
for k=1:K
    E=S-model.centroids(k,:); d(:,k)=sum((E*model.invCov).*E,2);
end
[~,pred]=min(d,[],2); % Equal class priors, pooled covariance classifier.
end

function b=balanced_accuracy(y,pred,K)
recall=nan(K,1);
for k=1:K
    ok=y==k & isfinite(pred); if any(ok), recall(k)=mean(pred(ok)==k); end
end
b=mean(recall,'omitnan');
end

function make_figures(r,outdir)
palette=[.15 .45 .75;.90 .48 .15;.30 .64 .42];
fig=figure('Visible','off','Color','w','Position',[50 50 1500 850]);
t=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
for m=1:numel(r.methods)
    ax=nexttile(t); hold(ax,'on'); S=r.scores{m}; model=r.models{m};
    good=r.valid; lo=min(S(good,:),[],1); hi=max(S(good,:),[],1); pad=.08*(hi-lo);
    [gx,gy]=meshgrid(linspace(lo(1)-pad(1),hi(1)+pad(1),150),linspace(lo(2)-pad(2),hi(2)+pad(2),150));
    q=classify_scores(model,[gx(:),gy(:)]); rgb=reshape(palette(q,:),[size(gx),3]);
    image(ax,'XData',[gx(1,1) gx(1,end)],'YData',[gy(1,1) gy(end,1)],'CData',rgb,'AlphaData',.12);
    hs=gobjects(4,1);
    for k=1:3
        ix=good & r.dataset.layer==r.classes(k);
        hs(k)=scatter(ax,S(ix,1),S(ix,2),26,palette(k,:),'filled','MarkerFaceAlpha',.8);
    end
    ix=~ismember(r.dataset.layer,r.classes) & all(isfinite(S),2);
    hs(4)=scatter(ax,S(ix,1),S(ix,2),24,[.45 .45 .45],'x');
    set(ax,'YDir','normal','FontSize',10); xlabel(ax,'Axis 1'); ylabel(ax,'Axis 2');
    title(ax,sprintf('%s | held-out BA %.1f%%',r.methods(m),100*mean(r.repeatBA(:,m))));
    if m==1, legend(ax,hs,[r.classes;"Unknown (projected only)"],'Location','best','FontSize',8); end
    axis(ax,'tight'); box(ax,'off');
end
ax=nexttile(t); bar(ax,100*mean(r.repeatBA,1),'FaceColor',[.35 .5 .65]); hold(ax,'on');
errorbar(ax,1:5,100*mean(r.repeatBA,1),100*std(r.repeatBA,0,1),'.k');
yline(ax,100/3,'--','Chance balanced accuracy'); ylim(ax,[0 100]);
xticks(ax,1:5); xticklabels(ax,r.methods); xtickangle(ax,25); ylabel(ax,'Held-out balanced accuracy (%)');
title(ax,'Five repeats of nested five-fold CV');
title(t,{'Layer prediction from fitted parameters','Maps/regions fit all known cells; reported accuracies are out-of-fold'});
exportgraphics(fig,fullfile(outdir,'01_projection_comparison.png'),'Resolution',170);
exportgraphics(fig,fullfile(outdir,'01_projection_comparison.pdf'),'ContentType','vector');
savefig(fig,fullfile(outdir,'01_projection_comparison.fig')); close(fig);
fig=figure('Visible','off','Color','w','Position',[50 50 1500 650]); t=tiledlayout(fig,2,5,'TileSpacing','compact');
[~,y]=ismember(r.dataset.layer(r.valid),r.classes);
for row=1:2
    for m=1:5
        ax=nexttile(t); C=zeros(3);
        if row==1
            for rep=1:size(r.predictions,2)
                C=C+confusionmat(y,r.predictions(:,rep,m),'Order',1:3);
            end
        else
            pr=r.subjectPredictions(:,m); good=isfinite(pr);
            if any(good), C=confusionmat(y(good),pr(good),'Order',1:3); end
        end
        C=100*C./max(sum(C,2),1); imagesc(ax,C,[0 100]);
        xticks(ax,1:3); yticks(ax,1:3); xticklabels(ax,r.classes); yticklabels(ax,r.classes);
        for i=1:3, for j=1:3, text(ax,j,i,sprintf('%.0f%%',C(i,j)),'HorizontalAlignment','center','Color','k'); end; end
        xlabel(ax,'Predicted'); ylabel(ax,'Actual'); title(ax,r.methods(m));
    end
end
colormap(fig,flipud(gray(256))); % Override with a light sequential palette for legibility.
colormap(fig,[linspace(1,.45,256)' linspace(1,.7,256)' ones(256,1)]);
title(t,{'Out-of-fold confusion matrices: percentage within each actual layer',...
    'Top: repeated cell CV | Bottom: leave-one-subject-out CV'});
exportgraphics(fig,fullfile(outdir,'02_heldout_confusions.png'),'Resolution',170);
savefig(fig,fullfile(outdir,'02_heldout_confusions.fig')); close(fig);
fig=figure('Visible','off','Color','w','Position',[50 50 1050 550]); t=tiledlayout(fig,1,2);
ax=nexttile(t); model=r.models{3}; imagesc(ax,model.W); colorbar(ax);
yticks(ax,1:numel(r.dataset.names)); yticklabels(ax,strrep(r.dataset.names,'_',' '));
xticks(ax,1:2); xticklabels(ax,{'Axis 1','Axis 2'}); title(ax,'Shrinkage LDA: standardized parameter coefficients');
ax=nexttile(t); grouped=nan(1,5);
for m=1:5, grouped(m)=balanced_accuracy(y,r.subjectPredictions(:,m),3); end
bar(ax,100*[mean(r.repeatBA,1);grouped;r.sensitivity]');
yline(ax,100/3,'--'); ylim(ax,[0 100]); xticks(ax,1:5); xticklabels(ax,r.methods); xtickangle(ax,30);
ylabel(ax,'Balanced accuracy (%)'); legend(ax,{'Cell CV','Subject CV','Initial selected parameters','Median final parameters'},'Location','best','FontSize',8);
title(ax,'Sensitivity checks');
exportgraphics(fig,fullfile(outdir,'03_coefficients_and_controls.png'),'Resolution',170); close(fig);
end

function write_report(r,outdir)
[~,y]=ismember(r.dataset.layer(r.valid),r.classes);
fid=fopen(fullfile(outdir,'README_results.md'),'w'); c=onCleanup(@()fclose(fid));
fprintf(fid,'# Layer separation from fitted model parameters\n\n');
fprintf(fid,'MATLAB %s. Source: `%s`.\n\n',version,r.dataset.simfile);
fprintf(fid,'One final-epoch model per cell, selected by the same mean cross-trial SPIKE distance as ParameterMDS.m. Known layers: %d cells; unknown: %d excluded from training.\n\n',sum(r.valid),sum(~ismember(r.dataset.layer,r.classes)));
fprintf(fid,'Class counts: L2/3 %d, L4 %d, L5/6 %d. Majority accuracy baseline %.1f%%; balanced-accuracy baseline 33.3%%.\n\n',r.counts,100*max(r.counts)/sum(r.counts));
fprintf(fid,'All five maps have two coordinates. Shrinkage and nonlinear hyperparameters are selected by three-fold inner CV; five-fold outer CV is repeated five times. Scaling, asinh scale, RBF landmarks, and projections are training-only in every split. Equal-prior pooled-covariance classification is used in projected space.\n\n');
fprintf(fid,'| Method | Cell CV balanced accuracy | Split SD | Accuracy | Subject CV BA | Initial BA | Median-batch BA |\n|---|---:|---:|---:|---:|---:|---:|\n');
for m=1:5
    gba=balanced_accuracy(y,r.subjectPredictions(:,m),3);
    fprintf(fid,'| %s | %.1f%% | %.1f points | %.1f%% | %.1f%% | %.1f%% | %.1f%% |\n',r.methods(m),100*mean(r.repeatBA(:,m)),100*std(r.repeatBA(:,m)),100*mean(r.repeatAccuracy(:,m)),100*gba,100*r.sensitivity(:,m));
end
fprintf(fid,'\nSplit SD measures sensitivity to fold assignments, not a confidence interval from independent datasets. Initial and median-batch controls use one outer CV repeat. Initial parameters use the batches selected by final output and therefore are not an independent initialization-only test.\n\n');
if isfield(r,'familywiseP')
    fprintf(fid,'Permutation test: %d label shuffles rerun the entire nested tuning procedure for all five methods. Observed first-repeat maximum BA %.3f; family-wise max-statistic p = %.4f. This test assumes exchangeable cells; it is not an animal-level significance test.\n\n',size(r.permutationBA,1),max(r.repeatBA(1,:)),r.familywiseP);
end
fprintf(fid,'Subject validation holds all cells from one subject out, with subject-grouped inner tuning. Missing-class folds are left unevaluated. Counts evaluated per method: ');
fprintf(fid,'%d ',sum(isfinite(r.subjectPredictions),1)); fprintf(fid,'of %d.\n\n',numel(y));
fprintf(fid,'Maps in 01_projection_comparison are fitted on all labeled cells, so visual separation is optimistic. Unknown cells are projected without training labels. Decision regions correspond to each plotted two-dimensional classifier. Never compare coordinates from different CV folds directly.\n\n');
fprintf(fid,'The RBF Fisher map uses 24 label-free farthest-point training landmarks, Gaussian widths 0.5/1/2 times the median training distance, and covariance shrinkage 0.25/0.75. It is a nonlinear feature-map discriminant projection, not t-SNE or UMAP.\n\n');
fprintf(fid,'Files: layer_analysis.mat (models, scores, predictions and permutations), selected_parameters.mat (cell IDs, subjects, labels, selected batches and parameter matrices), method_comparison.csv, and PNG/PDF/FIG plots. To project new rows in the original parameter order, use project_layer_parameters.m with results.models{methodIndex}.\n');
end
