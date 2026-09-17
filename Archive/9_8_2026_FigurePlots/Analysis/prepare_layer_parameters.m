function dataset = prepare_layer_parameters(repo, outdir)
% Match ParameterMDS: final epoch, lowest mean data-model SPIKE distance.
% Cell order is retained, including cells with unknown layer.
if ~isfolder(outdir), mkdir(outdir); end
simfile = fullfile(repo,'100_epoch_all_cells_Eprop.mat');
datafile = fullfile(repo,'Data','Data','all_units_info_with_polished_criteria_modified_perf.mat');
info = [dir(simfile); dir(datafile)];
signature = [[info.bytes]' [info.datenum]'];
cache = fullfile(outdir,'selected_parameters.mat');
if isfile(cache)
    old = load(cache,'dataset');
    if isequal(old.dataset.signature,signature)
        dataset = old.dataset; fprintf('Using validated parameter cache.\n'); return
    end
end
sim = load(simfile,'params','output');
dat = load(datafile,'all_data');
addpath(genpath(fullfile(repo,'SPIKY_SPIKEMEASURE','cSPIKE','cSPIKE')));
n = size(sim.output,3); nb = size(sim.output,1); nt = size(sim.output,2);
assert(n==numel(dat.all_data) && nt==10,'Unexpected cell/trial layout.');
names = fieldnames(sim.params); X = nan(n,numel(names)); Xinitial = X;
layer = strings(n,1); subject = strings(n,1); selection = nan(n,1);
distance = nan(n,nb); Xmedian = X;
for cellidx = 1:n
    cellinfo = dat.all_data(cellidx);
    layer(cellidx) = string(cellinfo.layer);
    subject(cellidx) = string(cellinfo.subject);
    tuning = lower(string(cellinfo.tuning_type));
    if contains(tuning,'contra'), focus=1;
    elseif contains(tuning,'45'), focus=2;
    elseif contains(tuning,'center'), focus=3;
    elseif contains(tuning,'ipsi'), focus=4;
    else, error('Unrecognized tuning for cell %d',cellidx); end
    for batch = 1:nb
        spikes = cell(1,2*nt);
        for trial = 1:nt
            spikes{trial} = cellinfo.ctrl_tar1_timestamps{trial,focus}(:)';
            spikes{trial+nt} = find(sim.output(batch,trial,cellidx,:))'/10000;
        end
        sts = SpikeTrainSet(spikes,0,3);
        D = sts.SPIKEdistanceMatrix();
        distance(cellidx,batch) = mean(D(1:nt,nt+1:2*nt),'all');
    end
    assert(all(isfinite(distance(cellidx,:))),'Nonfinite selection distances.');
    [~,selection(cellidx)] = min(distance(cellidx,:));
    for j=1:numel(names)
        values = sim.params.(names{j});
        assert(size(values,1)==nb && size(values,2)==n);
        X(cellidx,j) = values(selection(cellidx),cellidx,end);
        Xinitial(cellidx,j) = values(selection(cellidx),cellidx,1);
        Xmedian(cellidx,j) = median(values(:,cellidx,end));
    end
    if mod(cellidx,20)==0, fprintf('Selected SPIKE batches: %d/%d\n',cellidx,n); end
end
dataset = struct('X',X,'Xinitial',Xinitial,'Xmedian',Xmedian,'layer',layer,...
    'subject',subject,'cellID',(1:n)','names',{names},'selection',selection,...
    'distance',distance,'signature',signature,'simfile',simfile,'datafile',datafile);
save(cache,'dataset');
disp(table(unique(layer),arrayfun(@(s)sum(layer==s),unique(layer)),...
    'VariableNames',{'Layer','Count'}));
disp(table(unique(subject),arrayfun(@(s)sum(subject==s),unique(subject)),...
    'VariableNames',{'Subject','Count'}));
end
