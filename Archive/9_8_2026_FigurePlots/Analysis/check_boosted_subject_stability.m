function stability=check_boosted_subject_stability
% Post-exploration sensitivity check for boosted trees. This is NOT a new
% independent test set or a selection-adjusted significance test.
root=fileparts(mfilename('fullpath')); outdir=fullfile(root,'nonlinear_layer_results');
v=load(fullfile(outdir,'nonlinear_analysis.mat'),'results'); r=v.results;
X=r.dataset.X(r.valid,:); groups=r.dataset.subject(r.valid);
[~,y]=ismember(r.dataset.layer(r.valid),r.classes);
seeds=20261201+(0:4); predictions=nan(numel(y),5); chosen=nan(5,5);
for rep=1:5
    seed=seeds(rep); folds=group_folds(y,groups,5,seed);
    for f=1:5
        tr=folds~=f; te=~tr; inner=group_folds(y(tr),groups(tr),3,seed+f);
        values=nan(3,1); splits=[1 3 7];
        for c=1:3
            p=nan(sum(tr),1); A=X(tr,:); b=y(tr);
            for j=1:3
                a=inner~=j;
                p(~a)=fitpredict(A(a,:),b(a),A(~a,:),splits(c),seed+100*f+10*j+6);
            end
            values(c)=ba(b,p);
        end
        [~,ix]=max(values); chosen(rep,f)=splits(ix);
        predictions(te,rep)=fitpredict(X(tr,:),y(tr),X(te,:),splits(ix),seed+100*f+6);
    end
    fprintf('Boosted subject-grouped repeat %d BA %.3f\n',rep,ba(y,predictions(:,rep)));
end
assert(isequal(predictions(:,1),r.subjectPredictions(:,6)),...
    'First repeat should reproduce the original grouped result exactly.');
BA=arrayfun(@(j)ba(y,predictions(:,j)),1:5)';
accuracy=mean(predictions==y,1)';
stability=struct('seeds',seeds,'predictions',predictions,'chosenSplits',chosen,...
    'balancedAccuracy',BA,'accuracy',accuracy,'meanBA',mean(BA),'splitSD',std(BA));
save(fullfile(outdir,'boosted_subject_stability.mat'),'stability');
writetable(table(seeds',BA,accuracy,'VariableNames',{'Seed','BalancedAccuracy','Accuracy'}),...
    fullfile(outdir,'boosted_subject_stability.csv'));
fprintf('Mean boosted subject-grouped BA %.3f; split SD %.3f.\n',mean(BA),std(BA));
end

function pred=fitpredict(X,y,testX,splits,seed)
rng(seed,'twister'); mu=mean(X,1); sd=std(X,0,1); sd(sd<eps)=1;
t=templateTree('MaxNumSplits',splits,'MinLeafSize',5);
m=fitcensemble((X-mu)./sd,y,'Method','AdaBoostM2','NumLearningCycles',100,...
    'LearnRate',.1,'Learners',t,'Prior','uniform');
pred=predict(m,(testX-mu)./sd);
end

function f=group_folds(y,groups,K,seed)
[~,~,g]=unique(groups); G=max(g); rng(seed,'twister'); best=inf; f=[];
for attempt=1:200
    perm=randperm(G); assign=zeros(G,1); assign(perm)=mod(0:G-1,K)+1;
    trial=assign(g); counts=accumarray([trial y],1,[K 3]);
    if any(counts==0,'all'), continue; end
    score=sum(((counts-sum(counts,1)/K)./(sum(counts,1)/K)).^2,'all');
    if score<best, best=score; f=trial; end
end
assert(~isempty(f));
for k=1:K, assert(isempty(intersect(groups(f==k),groups(f~=k)))); end
end

function v=ba(y,pred)
assert(all(isfinite(pred)));
v=mean(arrayfun(@(k)mean(pred(y==k)==k),1:3));
end
