function results=explore_umap_tsne_layers
% Exploratory nonlinear 2D maps, classification, and original-cell tracing.
% MATLAB handles t-SNE, preprocessing, classification, validation and plots.
% Official Python umap-learn is invoked only to fit/transform UMAP embeddings.
root=fileparts(mfilename('fullpath')); outdir=fullfile(root,'embedding_layer_results');
if ~isfolder(outdir), mkdir(outdir); end
diary(fullfile(outdir,'analysis_log.txt')); cleanup=onCleanup(@()diary('off'));
v=load(fullfile(root,'layer_parameter_results','layer_analysis.mat'),'results');
d=v.results.dataset; classes=v.results.classes; [known,yall]=ismember(d.layer,classes);
assert(all(isfinite(d.X),'all')); y=yall(known); subjects=d.subject(known);
results=struct('dataset',d,'classes',classes,'known',known);
seeds=[20260914 20260915 20260916];
maps=struct('method',{},'preprocessing',{},'neighborhood',{},'minDist',{},...
    'seed',{},'supervised',{},'Y',{},'job',{});
jobs=struct('X',{},'query',{},'y',{},'neighbors',{},'minDist',{},...
    'targetWeight',{},'seed',{},'supervised',{});
% Full-data exploratory maps. Unsupervised maps never receive layer labels.
for prep=1:2
    [Z,~]=preprocess(d.X,prep);
    for seed=seeds
        for perplexity=[5 15 30 50]
            rng(seed,'twister');
            Y=tsne(Z,'Algorithm','exact','NumDimensions',2,'Perplexity',perplexity,...
                'LearnRate',100,'Exaggeration',4,'Standardize',false,...
                'Options',statset('MaxIter',1000));
            maps(end+1)=make_map("t-SNE",prep,perplexity,NaN,seed,false,Y,0);
        end
        for neighbors=[5 15 40]
            for minDist=[.01 .3]
                j=make_job(Z,zeros(0,size(Z,2)),yall,neighbors,minDist,seed,false);
                jobs(end+1)=j;
                maps(end+1)=make_map("UMAP",prep,neighbors,minDist,seed,false,[],numel(jobs));
            end
        end
        % Layer-guided exploratory view, explicitly NOT a validation result.
        jobs(end+1)=make_job(Z,zeros(0,size(Z,2)),yall,15,.1,seed,true);
        maps(end+1)=make_map("Layer-guided UMAP",prep,15,.1,seed,true,[],numel(jobs));
        fprintf('Prepared full maps: preprocessing %d, seed %d\n',prep,seed);
    end
end
% Inductive UMAP checks: fit on training cells and transform unseen cells.
% Fixed settings avoid selecting an embedding from outer test labels.
% Four pipelines: raw/asinh, each with unsupervised/layer-guided UMAP.
checks=struct('repeat',{},'fold',{},'kind',{},'pipeline',{},'trainRows',{},'testRows',{},'job',{});
for kind=1:2
    nr=3; if kind==2, nr=1; end
    for rep=1:nr
        if kind==1, folds=cell_folds(y,5,seeds(rep));
        else, folds=group_folds(y,subjects,5,20261201); end
        for f=1:5
            tr=folds~=f; te=~tr;
            for pipe=1:4
                prep=1+mod(pipe-1,2); supervised=pipe>2;
                [Z,p]=preprocess(d.X(known,:),prep,tr);
                assert(all(isfinite(p.mu)));
                jobs(end+1)=make_job(Z(tr,:),Z(te,:),y(tr),15,.1,seeds(rep)+f,supervised);
                checks(end+1)=struct('repeat',rep,'fold',f,'kind',kind,'pipeline',pipe,...
                    'trainRows',find(tr),'testRows',find(te),'job',numel(jobs));
            end
        end
    end
end
save(fullfile(outdir,'umap_jobs.mat'),'jobs','-v7');
python=fullfile(root,'dependencies','umap_env','Scripts','python.exe');
assert(isfile(python),'Run the documented UMAP environment setup first.');
cmd=sprintf('"%s" "%s" "%s" "%s"',python,fullfile(root,'umap_matlab_bridge.py'),...
    fullfile(outdir,'umap_jobs.mat'),fullfile(outdir,'umap_embeddings.mat'));
status=system(cmd,'-echo'); assert(status==0,'UMAP worker failed.');
embedded=load(fullfile(outdir,'umap_embeddings.mat'));
for m=1:numel(maps)
    if maps(m).job>0, maps(m).Y=double(embedded.(sprintf('train_%d',maps(m).job))); end
    assert(isequal(size(maps(m).Y),[size(d.X,1),2]));
end
results.maps=maps;
% Label-held-out classification CONDITIONAL on full-data unsupervised maps.
% This is transductive exploration, not independent new-cell validation.
mapBA=nan(numel(maps),1); svmBA=mapBA; neighborhoodOverlap=mapBA;
neighbors=cell(numel(maps),1); distances=neighbors;
for m=1:numel(maps)
    if maps(m).supervised, continue; end
    S=maps(m).Y(known,:); [idx,dist]=neighbors7(S);
    neighbors{m}=idx; distances{m}=dist;
    mapBA(m)=balanced_accuracy(y,loo_knn(y,idx,dist));
    allBA=nan(3,1);
    for rep=1:3
        folds=cell_folds(y,5,seeds(rep)); pr=nan(size(y));
        for f=1:5
            tr=folds~=f; te=~tr; pr(te)=classify2D(S(tr,:),y(tr),S(te,:),2);
        end
        allBA(rep)=balanced_accuracy(y,pr);
    end
    svmBA(m)=mean(allBA);
    Z=preprocess(d.X,maps(m).preprocessing); hi=neighbors7(Z(known,:));
    neighborhoodOverlap(m)=mean(arrayfun(@(i)numel(intersect(idx(i,:),hi(i,:)))/7,(1:numel(y))'));
end
results.mapSummary=table((1:numel(maps))',string({maps.method})',[maps.preprocessing]',...
    [maps.neighborhood]',[maps.minDist]',[maps.seed]',[maps.supervised]',mapBA,svmBA,neighborhoodOverlap,...
    'VariableNames',{'MapID','Method','Preprocessing','Neighborhood','MinDist','Seed','UsesLayerLabels',...
    'ConditionalLOO_KNN_BA','ConditionalCV_SVM_BA','OriginalNeighborOverlap'});
% Correct exploratory KNN map search for 199 shuffled label assignments.
% Map coordinates and neighbor geometry are fixed; supervised maps excluded.
nullMax=nan(199,1); unsup=find(~[maps.supervised]);
for perm=1:199
    rng(84000+perm,'twister'); yp=y(randperm(numel(y))); vals=nan(size(unsup));
    for a=1:numel(unsup)
        m=unsup(a); vals(a)=balanced_accuracy(yp,loo_knn(yp,neighbors{m},distances{m}));
    end
    nullMax(perm)=max(vals);
end
results.mapSearchPermutationP=(1+sum(nullMax>=max(mapBA)))/200;
results.mapSearchNullMax=nullMax;
% Genuine unseen-cell classification from UMAP's transformed coordinates.
pred=nan(numel(y),3,4,2); groupPred=nan(numel(y),4,2);
for c=1:numel(checks)
    q=checks(c); S=double(embedded.(sprintf('train_%d',q.job)));
    T=double(embedded.(sprintf('test_%d',q.job)));
    assert(isempty(intersect(q.trainRows,q.testRows)));
    if q.kind==2, assert(isempty(intersect(subjects(q.trainRows),subjects(q.testRows)))); end
    for clf=1:2
        pr=classify2D(S,y(q.trainRows),T,clf);
        if q.kind==1, pred(q.testRows,q.repeat,q.pipeline,clf)=pr;
        else, groupPred(q.testRows,q.pipeline,clf)=pr; end
    end
end
assert(all(isfinite(pred),'all') && all(isfinite(groupPred),'all'));
results.inductivePredictions=pred; results.groupPredictions=groupPred; results.checks=checks;
pipeNames=["UMAP";"Asinh UMAP";"Layer-guided UMAP";"Asinh layer-guided UMAP"];
methods=strings(8,1); BA=nan(8,1); SD=BA; Acc=BA; GroupBA=BA; index=0;
for pipe=1:4
    for clf=1:2
        index=index+1; cname="7NN"; if clf==2, cname="RBF SVM"; end
        methods(index)=pipeNames(pipe)+" + "+cname;
        rb=arrayfun(@(rep)balanced_accuracy(y,pred(:,rep,pipe,clf)),1:3);
        BA(index)=mean(rb); SD(index)=std(rb); Acc(index)=mean(pred(:,:,pipe,clf)==y,'all');
        GroupBA(index)=balanced_accuracy(y,groupPred(:,pipe,clf));
    end
end
results.inductiveSummary=table(methods,BA,SD,Acc,GroupBA,...
    'VariableNames',{'Pipeline','BalancedAccuracy','SplitSD','Accuracy','SubjectGroupedBA'});
results.bestMaps=nan(1,3);
for a=1:2
    name="t-SNE"; if a==2, name="UMAP"; end
    ids=find(string({maps.method})==name); [~,j]=max(mapBA(ids)); results.bestMaps(a)=ids(j);
end
results.bestMaps(3)=find([maps.supervised] & [maps.preprocessing]==1,1);
T=table(d.cellID,d.layer,d.subject,d.selection,'VariableNames',{'CellID','Layer','Subject','SelectedBatch'});
T=[T array2table(d.X,'VariableNames',d.names)]; results.cellTable=T;
save(fullfile(outdir,'embedding_analysis.mat'),'results','-v7');
writetable(results.mapSummary,fullfile(outdir,'map_comparison.csv'));
writetable(results.inductiveSummary,fullfile(outdir,'inductive_comparison.csv'));
for m=1:numel(maps)
    coords=table(maps(m).Y(:,1),maps(m).Y(:,2),'VariableNames',{'Embedding1','Embedding2'});
    writetable([T coords],fullfile(outdir,sprintf('map_%02d_cells.csv',m)));
end
make_overview(results,outdir);
for m=results.bestMaps
    [fig,~]=inspect_layer_embedding(m,false);
    exportgraphics(fig,fullfile(outdir,sprintf('map_%02d_parameter_colors.png',m)),'Resolution',160);
    savefig(fig,fullfile(outdir,sprintf('map_%02d_parameter_colors.fig',m))); close(fig);
end
write_report(results,outdir);
disp(results.inductiveSummary);
fprintf('Best conditional KNN map BA %.3f; search-adjusted shuffle p %.3f\n',max(mapBA),results.mapSearchPermutationP);
fprintf('Completed embedding exploration: %s\n',outdir);
end

function m=make_map(method,prep,n,md,seed,sup,Y,job)
m=struct('method',method,'preprocessing',prep,'neighborhood',n,'minDist',md,...
    'seed',seed,'supervised',sup,'Y',Y,'job',job);
end
function j=make_job(X,query,y,n,md,seed,sup)
y=double(y); y(y==0)=-1;
j=struct('X',X,'query',query,'y',y,'neighbors',n,'minDist',md,...
    'targetWeight',.5,'seed',seed,'supervised',sup);
end
function [Z,p]=preprocess(X,prep,tr)
if nargin<3, tr=true(size(X,1),1); end
p.scale=ones(1,size(X,2));
if prep==2
    p.scale=median(abs(X(tr,:)),1); p.scale(p.scale<eps)=1; X=asinh(X./p.scale);
end
p.mu=mean(X(tr,:),1); p.sd=std(X(tr,:),0,1); p.sd(p.sd<eps)=1;
Z=(X-p.mu)./p.sd;
end
function f=cell_folds(y,K,seed)
rng(seed,'twister'); c=cvpartition(y,'KFold',K); f=zeros(size(y));
for k=1:K, f(test(c,k))=k; end
end
function f=group_folds(y,groups,K,seed)
[~,~,g]=unique(groups); G=max(g); rng(seed,'twister'); best=inf; f=[];
for a=1:200
    perm=randperm(G); assign=zeros(G,1); assign(perm)=mod(0:G-1,K)+1;
    trial=assign(g); c=accumarray([trial y],1,[K 3]); if any(c==0,'all'), continue; end
    s=sum(((c-sum(c,1)/K)./(sum(c,1)/K)).^2,'all');
    if s<best, best=s; f=trial; end
end
assert(~isempty(f));
end
function [idx,dist]=neighbors7(S)
D=pdist2(S,S); D(1:size(D,1)+1:end)=Inf;
[dist,idx]=mink(D,7,2);
end
function pr=loo_knn(y,idx,dist)
votes=zeros(numel(y),3); counts=accumarray(y,1,[3 1]); w=1./max(dist,1e-10);
for k=1:3, votes(:,k)=sum(w.*(y(idx)==k),2)./(counts(k)-(y==k)); end
[~,pr]=max(votes,[],2);
end
function pr=classify2D(S,y,T,clf)
if clf==1
    model=fitcknn(S,y,'NumNeighbors',7,'DistanceWeight','inverse','Prior','uniform','Standardize',false);
else
    mu=mean(S,1); sd=std(S,0,1); sd(sd<eps)=1; S=(S-mu)./sd; T=(T-mu)./sd;
    learner=templateSVM('KernelFunction','gaussian','KernelScale',1,'BoxConstraint',1,'Standardize',false);
    model=fitcecoc(S,y,'Learners',learner,'Prior','uniform','Coding','onevsone');
end
pr=predict(model,T);
end
function b=balanced_accuracy(y,pr)
assert(all(isfinite(pr))); b=mean(arrayfun(@(k)mean(pr(y==k)==k),1:3));
end
function make_overview(r,outdir)
colors=[.15 .45 .75;.9 .48 .15;.3 .64 .42;.5 .5 .5];
fig=figure('Visible','off','Color','w','Position',[60 60 1450 520]); t=tiledlayout(fig,1,3,'TileSpacing','compact');
for j=1:3
    id=r.bestMaps(j); m=r.maps(id); ax=nexttile(t); hold(ax,'on');
    [~,labels]=ismember(r.dataset.layer,r.classes); labels(labels==0)=4; hs=gobjects(4,1);
    for k=1:4
        ix=labels==k; hs(k)=scatter(ax,m.Y(ix,1),m.Y(ix,2),28,colors(k,:),'filled');
        add_embedding_datatips(hs(k),r.cellTable(ix,:));
    end
    title(ax,sprintf('%s | map %d',m.method,id)); xlabel(ax,'Embedding axis 1'); ylabel(ax,'Embedding axis 2');
    subtitle(ax,sprintf('Neighborhood %d | seed %d',m.neighborhood,m.seed));
    if m.supervised, text(ax,.03,.97,'USES LAYER LABELS — descriptive only','Units','normalized','VerticalAlignment','top','FontSize',9); end
    if j==1, legend(ax,hs,[r.classes;"Unknown"],'Location','southoutside','Orientation','horizontal'); end
    box(ax,'off');
end
title(t,{'Nonlinear parameter maps: colored by cortical layer',...
    'Unsupervised maps shown are selected exploratory views; layer-guided separation is imposed by labels'});
exportgraphics(fig,fullfile(outdir,'embedding_overview.png'),'Resolution',170);
savefig(fig,fullfile(outdir,'embedding_overview.fig')); close(fig);
end
function write_report(r,outdir)
fid=fopen(fullfile(outdir,'README_results.md'),'w'); c=onCleanup(@()fclose(fid));
fprintf(fid,'# t-SNE and UMAP parameter exploration\n\nAll 220 cells retain original IDs; 196 known-layer cells enter classification. The 24 unknowns are gray in maps. Standardization and optional asinh transform use all cells only for descriptive full-data maps.\n\n');
fprintf(fid,'There are 24 t-SNE maps (perplexity 5/15/30/50, two preprocessings, three seeds), 36 unsupervised UMAP maps (neighbors 5/15/40, min_dist .01/.3, two preprocessings, three seeds), and six layer-guided UMAP maps (neighbors 15, min_dist .1, target_weight .5, two preprocessings, three seeds).\n\n');
fprintf(fid,'## Exploratory classification conditional on each full-data map\n\nUnsupervised map coordinates include held-out predictors but no layer labels. Inverse-distance 7NN leave-one-label-out scores and fixed-RBF-SVM three-repeat five-fold scores describe these particular maps; they are NOT independent new-cell performance. Selecting a map by these scores is label-informed exploratory selection.\n\n');
for j=1:2
    id=r.bestMaps(j); row=r.mapSummary(id,:);
    fprintf(fid,'Selected %s map %d: conditional 7NN BA %.1f%%, conditional SVM BA %.1f%%; neighborhood %d, preprocessing %d, seed %d.\n\n',row.Method,id,100*row.ConditionalLOO_KNN_BA,100*row.ConditionalCV_SVM_BA,row.Neighborhood,row.Preprocessing,row.Seed);
end
fprintf(fid,'Across all 60 unsupervised maps, a 199-shuffle maximum-statistic check of the best conditional 7NN balanced accuracy gives p=%.3f. This controls the KNN map search only, assumes exchangeable cells, and is not an animal-level test or correction for earlier analyses. Supervised maps and SVM scores are not included in that test.\n\n',r.mapSearchPermutationP);
fprintf(fid,'## Genuine held-out UMAP classification\n\nEach training fold fits its own preprocessing and UMAP. Test cells are transformed without their layer labels; even supervised UMAP receives training labels only. Fixed settings: 15 neighbors, min_dist .1, target_weight .5, 500 epochs. Classifiers: fixed inverse-distance 7NN and fixed Gaussian SVM (C=1, kernel scale=1 after 2D training-only scaling), uniform priors. No hyperparameter search in this evaluation.\n\n');
fprintf(fid,'| Pipeline | Cell BA | Split SD | Ordinary accuracy | Subject-grouped BA |\n|---|---:|---:|---:|---:|\n');
for i=1:height(r.inductiveSummary)
    q=r.inductiveSummary(i,:); fprintf(fid,'| %s | %.1f%% | %.1f points | %.1f%% | %.1f%% |\n',q.Pipeline,100*q.BalancedAccuracy,100*q.SplitSD,100*q.Accuracy,100*q.SubjectGroupedBA);
end
fprintf(fid,'\nCell evaluation is three repeats of five folds; subject evaluation is one five-fold group-balanced split. Chance BA=33.3%%. Split SD is not an independent-data confidence interval. MATLAB t-SNE has no native unseen-point transform, so no equivalent inductive t-SNE claim is made.\n\n');
fprintf(fid,'## Trace cells and parameters\n\nRun `inspect_layer_embedding(mapID)` for layer, subject and 12 parameter-colored panels with cell-ID data tips. Run `selected = trace_embedding_region(mapID)` and click a polygon, then Enter, to return original IDs/layers/subjects/batches and all parameter values. `map_XX_cells.csv` contains every point and coordinate; `embedding_analysis.mat` saves all maps and evaluation predictions. Representative map IDs: %d (t-SNE), %d (UMAP), %d (layer-guided UMAP).\n\n',r.bestMaps);
fprintf(fid,'Color limits in parameter panels use the 2nd–98th percentiles for contrast, but data tips and tables contain unmodified values. Between-island distances/areas in these maps should not be read as quantitative parameter differences. Check the original parameters and seed/setting stability. Layer-guided separation cannot itself validate a biological difference because the labels created it.\n\n');
fprintf(fid,'Implementation: MATLAB R2023b tsne and classifiers; official Python umap-learn 0.5.9.post2 called by MATLAB, isolated in dependencies/umap_env. Version details: umap_versions.json. [MATLAB t-SNE limitations](https://www.mathworks.com/help/stats/t-sne.html), [UMAP new-data transform](https://umap-learn.readthedocs.io/en/latest/transform.html), [supervised UMAP](https://umap-learn.readthedocs.io/en/latest/supervised.html).\n');
end
