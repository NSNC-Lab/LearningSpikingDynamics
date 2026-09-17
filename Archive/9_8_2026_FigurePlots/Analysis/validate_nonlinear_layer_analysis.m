function validate_nonlinear_layer_analysis
root=fileparts(mfilename('fullpath'));
outdir=fullfile(root,'nonlinear_layer_results');
v=load(fullfile(outdir,'nonlinear_analysis.mat'),'results'); r=v.results;
[~,y]=ismember(r.dataset.layer(r.valid),r.classes);
groups=r.dataset.subject(r.valid); n=numel(y);
for rep=1:numel(r.seeds)
    rng(r.seeds(rep),'twister'); c=cvpartition(y,'KFold',5);
    seen=zeros(n,1);
    for f=1:5
        d=r.details{rep}{f};
        assert(isequal(d.testRows,find(test(c,f))));
        assert(isempty(intersect(d.trainRows,d.testRows)));
        seen(d.testRows)=seen(d.testRows)+1;
        [~,winner]=max(d.innerBA); assert(winner==d.selectedFamily);
        assert(isequal(r.predictions(d.testRows,rep,8),r.predictions(d.testRows,rep,winner)));
    end
    assert(all(seen==1));
    for m=1:8
        pred=r.predictions(:,rep,m);
        b=mean(arrayfun(@(k)mean(pred(y==k)==k),1:3));
        assert(abs(b-r.repeatBA(rep,m))<1e-12);
    end
end
for f=1:5
    d=r.subjectDetails{f}; tr=d.trainRows; te=d.testRows;
    assert(isempty(intersect(groups(tr),groups(te))));
    for k=1:3
        in=d.innerFolds==k;
        assert(isempty(intersect(groups(tr(in)),groups(tr(~in)))));
    end
    [~,winner]=max(d.innerBA);
    assert(isequal(r.subjectPredictions(te,8),r.subjectPredictions(te,winner)));
end
[pr,scores]=predict_layer_classifier(r.finalModel,r.dataset.X);
assert(isequal(pr,r.finalPredictions));
assert(max(abs(scores-r.finalScores),[],'all')<1e-12);
fprintf('PASS: same outer cell splits as original analysis; disjoint folds; grouped inner/outer subject isolation; inner-only family selection; recomputed metrics; reusable predictions.\n');
end
