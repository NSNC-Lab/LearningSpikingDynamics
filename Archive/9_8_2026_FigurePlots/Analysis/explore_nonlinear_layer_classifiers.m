function results = explore_nonlinear_layer_classifiers(nRep)
% Full-parameter nonlinear classification; no two-dimensional bottleneck.
% Run from Analysis: results = explore_nonlinear_layer_classifiers;
% Same cells and outer cell folds as explore_layer_parameter_space.
% Every preprocessing step, model family and hyperparameter choice is
% training-only. Main metric: balanced accuracy (mean of three recalls).
% nRep=0 runs a quick API smoke test without writing classification results.
if nargin<1, nRep=5; end
root=fileparts(mfilename('fullpath'));
old=load(fullfile(root,'layer_parameter_results','layer_analysis.mat'),'results');
previous=old.results; dataset=previous.dataset; valid=previous.valid;
X=dataset.X(valid,:); [~,y]=ismember(dataset.layer(valid),previous.classes);
subjects=dataset.subject(valid); configs=candidate_configs;
names=["Gaussian SVM";"Asinh Gaussian SVM";"Polynomial SVM";...
    "Nearest neighbors";"Random forest";"Boosted trees";"Neural network"];
for m=1:numel(names)
    model=train_model(X,y,configs{m}(1,:),m,108+m);
    pr=apply_model(model,X); assert(all(ismember(pr,1:3)));
end
fprintf('All seven nonlinear classifier smoke tests passed.\n');
if nRep==0, results=[]; return; end
outdir=fullfile(root,'nonlinear_layer_results');
if ~isfolder(outdir), mkdir(outdir); end
diary(fullfile(outdir,'analysis_log.txt')); cleanup=onCleanup(@()diary('off'));
assert(all(isfinite(X),'all') && size(X,1)==numel(y));
results=struct('dataset',dataset,'valid',valid,'classes',previous.classes,...
    'methods',[names;"Inner-CV-selected family"],'configs',{configs});
results.seeds=20260914+(0:nRep-1);
results.predictions=nan(numel(y),nRep,8);
results.repeatBA=nan(nRep,8); results.repeatAccuracy=nan(nRep,8);
results.details=cell(nRep,1);
results.previousBA=previous.repeatBA;
results.previousMethods=previous.methods;
fprintf('Analyzing %d cells, %d parameters; %d candidate configurations.\n',...
    size(X,1),size(X,2),sum(cellfun(@(c)size(c,1),configs)));
for rep=1:nRep
    folds=cell_folds(y,5,results.seeds(rep));
    [pr,detail]=nested_evaluation(X,y,folds,configs,results.seeds(rep),[]);
    results.predictions(:,rep,:)=reshape(pr,numel(y),1,8);
    results.details{rep}=detail;
    for m=1:8
        results.repeatBA(rep,m)=ba(y,pr(:,m));
        results.repeatAccuracy(rep,m)=mean(y==pr(:,m));
    end
    fprintf('Repeat %d held-out balanced accuracies:\n',rep);
    disp(table(results.methods,results.repeatBA(rep,:)','VariableNames',{'Method','BA'}));
    save(fullfile(outdir,'nonlinear_analysis.mat'),'results');
end
% Five subject-grouped folds, also grouped for inner tuning. Metadata defines
% groups; this does not assert each unique subject string is a distinct animal.
gfolds=group_folds(y,subjects,5,20261201);
[results.subjectPredictions,results.subjectDetails]=nested_evaluation(...
    X,y,gfolds,configs,20261201,subjects);
results.subjectFolds=gfolds;
groupBA=arrayfun(@(m)ba(y,results.subjectPredictions(:,m)),1:8);
results.summary=table(results.methods,mean(results.repeatBA,1)',std(results.repeatBA,0,1)',...
    mean(results.repeatAccuracy,1)',groupBA',...
    'VariableNames',{'Method','BalancedAccuracy','SplitSD','Accuracy','SubjectGroupedBA'});
disp(results.summary);
% Final deployable classifier: select family AND configuration by inner CV
% on all known cells; its apparent accuracy is not an independent estimate.
[bestConfig,innerBA]=tune_models(X,y,cell_folds(y,5,20261215),configs,20261215);
[~,family]=max(innerBA);
results.finalModel=train_model(X,y,bestConfig{family},family,20261216);
results.finalMethod=names(family); results.finalInnerBA=innerBA;
[results.finalPredictions,results.finalScores]=apply_model(results.finalModel,dataset.X);
save(fullfile(outdir,'nonlinear_analysis.mat'),'results');
writetable(results.summary,fullfile(outdir,'method_comparison.csv'));
make_plots(results,y,outdir);
write_report(results,y,outdir);
fprintf('Completed nonlinear comparison: %s\n',outdir);
end

function configs=candidate_configs
% Columns have method-specific meanings, documented in train_model.
[C,W]=ndgrid([.1 1 10],[1 3 10]);
svm=[C(:),W(:)];
configs={svm,svm,[.1 2;1 2;.1 3;1 3],...
    [3 0;7 0;15 0;31 0;7 1;15 1],...
    [1 100;5 100;15 100],...
    [1 .1;3 .1;7 .1],...
    [8 .01;8 .1;16 .01;16 .1]};
end

function f=cell_folds(y,K,seed)
rng(seed,'twister'); c=cvpartition(y,'KFold',K); f=zeros(size(y));
for k=1:K, f(test(c,k))=k; end
end

function f=group_folds(y,groups,K,seed)
% Choose a label-count-balanced assignment, not by model performance.
[~,~,g]=unique(groups); G=max(g); assert(G>=K);
rng(seed,'twister'); best=inf; f=[];
for attempt=1:200
    perm=randperm(G); assign=zeros(G,1); assign(perm)=mod(0:G-1,K)+1;
    trial=assign(g); counts=accumarray([trial y],1,[K 3]);
    if any(counts==0,'all'), continue; end
    score=sum(((counts-sum(counts,1)/K)./(sum(counts,1)/K)).^2,'all');
    if score<best, best=score; f=trial; end
end
assert(~isempty(f),'Could not form group folds containing all layers.');
for k=1:K, assert(isempty(intersect(groups(f==k),groups(f~=k)))); end
end

function [pred,details]=nested_evaluation(X,y,folds,configs,seed,groups)
pred=nan(numel(y),8); details=cell(max(folds),1);
for f=1:max(folds)
    tr=folds~=f; te=~tr;
    if isempty(groups), inner=cell_folds(y(tr),3,seed+f);
    else, inner=group_folds(y(tr),groups(tr),3,seed+f); end
    [best,innerBA]=tune_models(X(tr,:),y(tr),inner,configs,seed+100*f);
    for m=1:7
        model=train_model(X(tr,:),y(tr),best{m},m,seed+100*f+m);
        pred(te,m)=apply_model(model,X(te,:));
    end
    [~,winner]=max(innerBA); pred(te,8)=pred(te,winner);
    details{f}=struct('config',{best},'innerBA',innerBA,'selectedFamily',winner,...
        'trainRows',find(tr),'testRows',find(te),'innerFolds',inner);
    fprintf('Outer fold %d/%d complete; inner-selected family %d.\n',f,max(folds),winner);
end
assert(all(isfinite(pred),'all'));
end

function [best,bestBA]=tune_models(X,y,folds,configs,seed)
best=cell(7,1); bestBA=nan(7,1);
for m=1:7
    values=nan(size(configs{m},1),1);
    for c=1:numel(values)
        pr=nan(size(y));
        for f=1:max(folds)
            tr=folds~=f; te=~tr;
            model=train_model(X(tr,:),y(tr),configs{m}(c,:),m,seed+10*f+m);
            pr(te)=apply_model(model,X(te,:));
        end
        values(c)=ba(y,pr);
    end
    [bestBA(m),ix]=max(values); best{m}=configs{m}(ix,:);
end
end

function model=train_model(X,y,c,m,seed)
rng(seed,'twister'); model.family=m; model.config=c;
model.transformScale=ones(1,size(X,2));
if m==2
    scale=median(abs(X),1); scale(scale<eps)=1;
    model.transformScale=scale; X=asinh(X./scale);
end
model.mu=mean(X,1); model.sd=std(X,0,1); model.sd(model.sd<eps)=1;
Z=(X-model.mu)./model.sd;
switch m
    case {1,2} % C, Gaussian kernel width in standardized coordinates.
        learner=templateSVM('KernelFunction','gaussian','BoxConstraint',c(1),...
            'KernelScale',c(2),'Standardize',false);
        mdl=fitcecoc(Z,y,'Learners',learner,'Coding','onevsone','Prior','uniform');
    case 3 % C, polynomial order; scale sqrt(number of parameters).
        learner=templateSVM('KernelFunction','polynomial','PolynomialOrder',c(2),...
            'KernelScale',sqrt(size(Z,2)),'BoxConstraint',c(1),'Standardize',false);
        mdl=fitcecoc(Z,y,'Learners',learner,'Coding','onevsone','Prior','uniform');
    case 4 % k, use inverse distance weighting.
        weight='equal'; if c(2)==1, weight='inverse'; end
        mdl=fitcknn(Z,y,'NumNeighbors',c(1),'DistanceWeight',weight,...
            'Standardize',false,'Prior','uniform');
    case 5 % Minimum leaf size, number of trees.
        learner=templateTree('MinLeafSize',c(1),'NumVariablesToSample',ceil(sqrt(size(Z,2))));
        mdl=fitcensemble(Z,y,'Method','Bag','NumLearningCycles',c(2),...
            'Learners',learner,'Prior','uniform');
    case 6 % Maximum splits per weak learner, learning rate.
        learner=templateTree('MaxNumSplits',c(1),'MinLeafSize',5);
        mdl=fitcensemble(Z,y,'Method','AdaBoostM2','NumLearningCycles',100,...
            'LearnRate',c(2),'Learners',learner,'Prior','uniform');
    case 7 % Hidden units per layer, L2 penalty. Two layers for 16-unit option.
        layers=c(1); if c(1)==16, layers=[16 8]; end
        % Inverse-frequency weights give equal total weight to each class.
        counts=accumarray(y,1,[3 1]); weights=1./counts(y);
        mdl=fitcnet(Z,y,'LayerSizes',layers,'Activations','relu','Lambda',c(2),...
            'IterationLimit',200,'Standardize',false,'Weights',weights);
end
model.classifier=compact(mdl);
end

function [pr,scores]=apply_model(model,X)
if model.family==2, X=asinh(X./model.transformScale); end
[pr,scores]=predict(model.classifier,(X-model.mu)./model.sd);
assert(isequal(model.classifier.ClassNames(:),(1:3)'));
end

function v=ba(y,pr)
assert(all(isfinite(pr)));
v=mean(arrayfun(@(k)mean(pr(y==k)==k),1:3));
end

function make_plots(r,y,outdir)
fig=figure('Visible','off','Color','w','Position',[50 50 1450 620]);
t=tiledlayout(fig,1,2,'TileSpacing','compact'); ax=nexttile(t);
bars=bar(ax,100*[mean(r.repeatBA,1)' r.summary.SubjectGroupedBA]); hold(ax,'on');
yline(ax,100/3,'--k','Chance balanced accuracy'); ylim(ax,[0 100]);
xticks(ax,1:8); xticklabels(ax,r.methods); xtickangle(ax,35);
ylabel(ax,'Held-out balanced accuracy (%)');
legend(ax,bars,{'Repeated cell CV','Subject-grouped CV'},'Location','northwest');
title(ax,'Full 12-parameter nonlinear classification');
ax=nexttile(t); C=zeros(3);
for rep=1:size(r.predictions,2), C=C+confusionmat(y,r.predictions(:,rep,8),'Order',1:3); end
C=100*C./sum(C,2); imagesc(ax,C,[0 100]);
for i=1:3, for j=1:3, text(ax,j,i,sprintf('%.1f%%',C(i,j)),'HorizontalAlignment','center'); end; end
xticks(ax,1:3); yticks(ax,1:3); xticklabels(ax,r.classes); yticklabels(ax,r.classes);
xlabel(ax,'Predicted layer'); ylabel(ax,'Actual layer');
title(ax,{'Out-of-fold confusion','Family selected within each training fold'});
colormap(ax,[linspace(1,.45,256)' linspace(1,.7,256)' ones(256,1)]);
title(t,'All preprocessing and tuning are training-only; unknown layers excluded');
exportgraphics(fig,fullfile(outdir,'nonlinear_comparison.png'),'Resolution',170);
savefig(fig,fullfile(outdir,'nonlinear_comparison.fig')); close(fig);
end

function write_report(r,y,outdir)
fid=fopen(fullfile(outdir,'README_results.md'),'w'); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'# Nonlinear layer classification\n\nMATLAB %s. All 12 parameters; same 196 known-layer cells and SPIKE-selected batches as the earlier exploration. No two-dimensional projection constraint.\n\n',version);
fprintf(fid,'Balanced accuracy averages recall equally across L2/3, L4, and L5/6; chance is 33.3%%. Ordinary majority-class accuracy baseline is %.1f%%.\n\n',100*max(accumarray(y,1))/numel(y));
fprintf(fid,'| Method | Cell CV BA | Split SD | Accuracy | Subject-grouped BA |\n|---|---:|---:|---:|---:|\n');
for m=1:8
    fprintf(fid,'| %s | %.1f%% | %.1f points | %.1f%% | %.1f%% |\n',r.methods(m),...
        100*mean(r.repeatBA(:,m)),100*std(r.repeatBA(:,m)),100*mean(r.repeatAccuracy(:,m)),100*r.summary.SubjectGroupedBA(m));
end
fprintf(fid,'\nFive outer cell folds repeated %d times, using the same seeds as the earlier analysis. Three-fold inner CV chooses hyperparameters separately for each family. The last row additionally chooses the family using inner CV only: it evaluates the entire model-selection procedure without choosing a winner from outer test labels. Split SD is not an independent-data confidence interval.\n\n',size(r.predictions,2));
fprintf(fid,'Subject validation uses one five-fold group-balanced partition, with three subject-grouped inner folds. No subject string crosses a training/test boundary. This differs from the earlier leave-one-subject-out protocol.\n\n');
fprintf(fid,'All scaling and asinh transforms are fitted to training rows only. Equal priors are used except in fitcnet, which receives inverse-class-frequency observation weights. Search grids are declared in candidate_configs. Random forest and boosted models use 100 trees; neural networks use 8 units or 16/8 units with 200 training iterations. No independent untouched dataset or permutation significance test was added in this extension; small gains after exploratory searches need confirmation.\n\n');
fprintf(fid,'Final full-data selected family: %s. Its training scores are saved for descriptive use only; do not report training accuracy as validation.\n\n',r.finalMethod);
fprintf(fid,'Run `results = explore_nonlinear_layer_classifiers;` from Analysis. Models, preprocessing, original cell IDs, all outer predictions, fold membership and selected configurations are saved in nonlinear_analysis.mat. Existing analysis files and source data were not altered.\n\n');
fprintf(fid,'Implementation references: [MATLAB SVM/ECOC](https://www.mathworks.com/help/stats/fitcecoc.html), [ensembles](https://www.mathworks.com/help/stats/fitcensemble.html), [nearest neighbors](https://www.mathworks.com/help/stats/fitcknn.html), [neural networks](https://www.mathworks.com/help/stats/fitcnet.html).\n');
end
