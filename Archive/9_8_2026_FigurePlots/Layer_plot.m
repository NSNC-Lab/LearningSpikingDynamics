%L2/3 = red
%L4 = Green
%L5/6 = Blue



%Import stuff
close all; clear all;
L2_c = 'r';
L4_c = 'g';
L6_c = 'b';
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE")
addpath("C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\SPIKY_SPIKEMEASURE\cSPIKE\cSPIKE\cSPIKEmex")
InitializecSPIKE;
sim_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\100_epoch_all_cells_Eprop';
data_location = 'C:\Users\ipboy\Documents\GitHub\LearningSpikingDynamics\Data\Data\all_units_info_with_polished_criteria_modified_perf.mat';
data_object = load(data_location);
sim_object = load(sim_location);
Distance_measure = 'RMSE';

%Do selection
Choose_by = 'SPIKE';
selection = calculate_selction(Choose_by, sim_object, data_object);

%%

%Calculate feature distributions
fr_data = [];
fr_sim = [];
cv_data = [];
cv_sim = [];
ls_data = [];
ls_sim = [];
re_data = [];
re_sim = [];
for k = 1:220
    %Calculate Data spikes beforehand for efficiency
    tuning = lower(char(string(data_object.all_data(k).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);

    fr_data = [fr_data, calc_fr_data(spikes)];
    fr_sim = [fr_sim,calc_fr_sim(sim_object,k, selection)];
    cv_data = [cv_data, calc_cv_data(spikes)];
    cv_sim = [cv_sim,calc_cv_sim(sim_object,k, selection)];
    ls_data = [ls_data, calc_ls_data(spikes)];
    ls_sim = [ls_sim, calc_ls_sim(sim_object,k, selection);];
    re_data = [re_data, calc_re_data(spikes)];
    re_sim = [re_sim, calc_re_sim(sim_object,k, selection)];
end


color_vecs = [];
f_vals = fields(sim_object.params);
for m = 1:12
    sub_vec = [];
    for k = 1:220
        values = sim_object.params.(f_vals{m});
        sub_vec = [sub_vec, squeeze(values(selection(k),k,end))];
    end
    color_vecs = [color_vecs; sub_vec];
end
%%
%Olsen and Hausenstaub scatter figure

%Look at just a subset of highly steryotyped cells
unusable = [2	8	11	16	17	18	20	23	24	26	34	43	46	47	48	50	54	57	59	60	62	63	66	70	72	77	79	81	82	98	100	105	114	115	117	123	125	130	134	135	137	144	147	148	151	153	155	156	158	159	160	164	165	166	167	168	172	173	174	179	180	181	182	183	184	185	187	190	198	204	206	207	209	212	214];
borderline_unusable = [5	21	22	25	29	33	37	73	75	35	55	80	86	93	94	99	103	104	109	119	120	131	140	152	161	162	169	170	177	189	195	196	197	199	201	205	211	213];
poor = [3	4	9	10	27	31	32	38	39	40	42	45	49	51	58	65	67	69	71	78	83	85	87	88	89	95	101	107	110	111	112	113	116	118	121	122	136	138	141	142	145	146	150	154	171	175	176	193	208	216];
medium = [6	12	13	14	28	30	36	41	44	56	61	74	76	84	90	91	97	106	108	126	139	143	149	157	163	178	188	191	192	194	202	203	215	217	218	219	220];
good = [1	15	19	52	53	64	92	96	124	127	128	129	132	186	200	210];
great = [7  68  102 133];

chosen_nums = [great,good];
chosen_nums2 = 1:220;

balance_vecs = color_vecs./max(color_vecs,[],2);
figure(Position=[100,100,1200,500]);
subplot(1,2,1);
xlabel('Onset Contribution')
ylabel('Offset Contribution')
%create_contour(balance_vecs(8,chosen_nums),balance_vecs(9,chosen_nums),balance_vecs(11,chosen_nums),balance_vecs(12,chosen_nums))
scatter(balance_vecs(8,chosen_nums),balance_vecs(9,chosen_nums),'b','filled'); hold on
scatter(balance_vecs(11,chosen_nums),balance_vecs(12,chosen_nums),'r','filled'); 
title('Highly Steryotyped Cells');
legend({'E \rightarrow E', 'E \rightarrow PV'},'Location','northwest')
subplot(1,2,2);
xlabel('Onset Contribution')
ylabel('Offset Contribution')
%create_contour(balance_vecs(8,chosen_nums2),balance_vecs(9,chosen_nums2),balance_vecs(11,chosen_nums2),balance_vecs(12,chosen_nums2))
scatter(balance_vecs(8,chosen_nums2),balance_vecs(9,chosen_nums2),'b','filled'); hold on
scatter(balance_vecs(11,chosen_nums2),balance_vecs(12,chosen_nums2),'r','filled'); 
title('All Cells');

L2 = [];
L4 = [];
L6 = [];
for k = 1:220
    if strcmp(data_object.all_data(k).layer, 'L2/3'); L2 = [L2, k]; end
    if strcmp(data_object.all_data(k).layer, 'L4'); L4 = [L4, k]; end
    if strcmp(data_object.all_data(k).layer, 'L5/6'); L6 = [L6, k]; end
end


figure(Position=[100,100,1300,900]);
t = tiledlayout(3,4,'TileSpacing','compact','Padding','compact');
%edges = 0:2:65;
%centers = edges(1:(length(edges)-1)) + 1;
%[N_p,edges] = histcounts(fr_data(L2),edges);
%[N_p,edges] = histcounts(fr_data(L2),edges);
%plot(centers,N_p,L2_c); hold on

datas = {fr_data,re_data,ls_data,cv_data};
sims = {fr_sim,re_sim,ls_sim,cv_sim};
layers = {L2,L4,L6};
Colors = {L2_c,L4_c,L6_c};
labels = {'L2/3','L4','L5/6'};
features = {'Firing Rate','Reliability','Lifetime Sparsity','CV'};


for m = 1:3
    for k = 1:4
        ax((m-1)*4 + k) = nexttile;
        histogram(datas{k}(layers{m}),20,'FaceColor',[0,0,0],'FaceAlpha',0.2,'Normalization','probability'); hold on
        histogram(sims{k}(layers{m}),20,'FaceColor',Colors{m},'FaceAlpha',0.7,'Normalization','probability'); hold on
        xlim([min([datas{k},sims{k}]),max([datas{k},sims{k}])])
        if (k==1)
            legend({'Data','Model'})
            ylabel(labels{m})
        end
        if (m==1)
            title(features{k})
        end

    end
end
% histogram(fr_data(L2),50,'FaceColor',L2_c,'FaceAlpha',0.2); hold on
% histogram(fr_sim(L2),50,'FaceColor',L2_c,'FaceAlpha',1); hold on
% ylabel('Layer 2/3')
% title('Firing Rate')
% ax(2) = nexttile;
% histogram(re_data(L2),50,'FaceColor',L2_c,'FaceAlpha',0.2); hold on
% histogram(re_sim(L2),50,'FaceColor',L2_c,'FaceAlpha',1); hold on
% title('Reliability')
% ax(3) = nexttile;
% histogram(ls_data(L2),50,'FaceColor',L2_c,'FaceAlpha',0.2); hold on
% histogram(ls_sim(L2),50,'FaceColor',L2_c,'FaceAlpha',1); hold on
% title('Lifetime Sparseness')
% ax(4) = nexttile;
% histogram(cv_data(L2),50,'FaceColor',L2_c,'FaceAlpha',0.2); hold on
% histogram(cv_sim(L2),50,'FaceColor',L2_c,'FaceAlpha',1); hold on
% %histogram(fr_data(L4),50,'FaceColor',L4_c,'FaceAlpha',0.2); hold on
% %histogram(fr_sim(L4),50,'FaceColor',L4_c,'FaceAlpha',1); hold on
% %histogram(fr_data(L6),50,'FaceColor',L6_c,'FaceAlpha',0.2); hold on
% %histogram(fr_sim(L6),50,'FaceColor',L6_c,'FaceAlpha',1); hold on
% title('CV')
% ax(5) = nexttile;
% histogram(fr_data(L4),50,'FaceColor',L4_c,'FaceAlpha',0.2); hold on
% histogram(fr_sim(L4),50,'FaceColor',L4_c,'FaceAlpha',1); hold on
% ylabel('Layer 4')
% ax(6) = nexttile;
% histogram(re_data(L4),50,'FaceColor',L4_c,'FaceAlpha',0.2); hold on
% histogram(re_sim(L4),50,'FaceColor',L4_c,'FaceAlpha',1); hold on
% ax(7) = nexttile;
% histogram(ls_data(L4),50,'FaceColor',L4_c,'FaceAlpha',0.2); hold on
% histogram(ls_sim(L4),50,'FaceColor',L4_c,'FaceAlpha',1); hold on
% ax(8) = nexttile;
% histogram(cv_data(L4),50,'FaceColor',L4_c,'FaceAlpha',0.2); hold on
% histogram(cv_sim(L4),50,'FaceColor',L4_c,'FaceAlpha',1); hold on
% %L5/6
% ax(9) = nexttile;
% histogram(fr_data(L6),50,'FaceColor',L6_c,'FaceAlpha',0.2); hold on
% histogram(fr_sim(L6),50,'FaceColor',L6_c,'FaceAlpha',1); hold on
% ylabel('Layer 5/6')
% ax(10) = nexttile;
% histogram(re_data(L6),50,'FaceColor',L6_c,'FaceAlpha',0.2); hold on
% histogram(re_sim(L4),50,'FaceColor',L6_c,'FaceAlpha',1); hold on
% ax(11) = nexttile;
% histogram(ls_data(L6),50,'FaceColor',L6_c,'FaceAlpha',0.2); hold on
% histogram(ls_sim(L6),50,'FaceColor',L6_c,'FaceAlpha',1); hold on
% ax(12) = nexttile;
% histogram(cv_data(L6),50,'FaceColor',L6_c,'FaceAlpha',0.2); hold on
% histogram(cv_sim(L6),50,'FaceColor',L6_c,'FaceAlpha',1); hold on

figure;
t = tiledlayout(3,12,'TileSpacing','compact','Padding','compact');
colors = {L2_c,L4_c,L6_c};
indicies = {L2,L4,L6};
titles = {'L2/3','L4','L5/6'};

for m = 1:3
    for k = 1:12
        ax((m-1)*12 + k) = nexttile;
        histogram(color_vecs(k,indicies{m}),20,'FaceColor',colors{m},'Normalization','probability'); hold on
        if (m == 1)
            title(strrep(f_vals{k},'_',' '))
        end
        if (k == 1)
            ylabel(titles{m})
        end
        xlim([min(color_vecs(k,:)),max(color_vecs(k,:))])
    end
end

hs = [];
ps = [];
for k = 1:12
    [h, p, ci, stats] = ttest2(color_vecs(k,indicies{1}), color_vecs(k,indicies{2}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs = [hs,h];
    ps = [ps,p];
    [h, p, ci, stats] = ttest2(color_vecs(k,indicies{2}), color_vecs(k,indicies{3}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs = [hs,h];
    ps = [ps,p];
    [h, p, ci, stats] = ttest2(color_vecs(k,indicies{1}), color_vecs(k,indicies{3}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs = [hs,h];
    ps = [ps,p];
end

hs2 = [];
ps2 = [];
for k = 1:4
    [h, p, ci, stats] = ttest2(sims{k}(layers{1}), sims{k}(layers{2}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs2 = [hs2,h];
    ps2 = [ps2,p];
    [h, p, ci, stats] = ttest2(sims{k}(layers{2}), sims{k}(layers{3}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs2 = [hs2,h];
    ps2 = [ps2,p];
    [h, p, ci, stats] = ttest2(sims{k}(layers{1}), sims{k}(layers{3}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs2 = [hs2,h];
    ps2 = [ps2,p];
end

hs3 = [];
ps3 = [];
for k = 1:4
    [h, p, ci, stats] = ttest2(datas{k}(layers{1}), datas{k}(layers{2}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs3 = [hs3,h];
    ps3 = [ps3,p];
    [h, p, ci, stats] = ttest2(datas{k}(layers{2}), datas{k}(layers{3}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs3 = [hs3,h];
    ps3 = [ps3,p];
    [h, p, ci, stats] = ttest2(datas{k}(layers{1}), datas{k}(layers{3}),'Vartype', 'unequal', 'Alpha', 0.05);
    hs3 = [hs3,h];
    ps3 = [ps3,p];
end

function create_contour(dist1x,dist1y,dist2x,dist2y)

    edges = linspace(0,1,50);
    
    counts1 = histcounts2(dist1x,dist1y,edges,edges);
    counts2 = histcounts2(dist2x,dist2y,edges,edges);
    
    % Gaussian smoothing kernel
    x = 0:0.1:10;
    g = normpdf(x,5,0.5);
    g_filt = g' * g;
    
    counts1 = conv2(counts1,g_filt,"same");
    counts2 = conv2(counts2,g_filt,"same");
    
    centers = edges(1:end-1) + diff(edges)/2;
    
    % Normalize each density independently
    density1 = rescale(counts1',0,1);
    density2 = rescale(counts2',0,1);
    
    % Only values between these limits will be visible
    lowerLevel = 0.35;
    upperLevel = 0.55;
    
    ring1 = density1 >= lowerLevel & density1 <= upperLevel;
    ring2 = density2 >= lowerLevel & density2 <= upperLevel;
    
    % Solid-color RGB images
    blueImage = zeros([size(density1),3]);
    blueImage(:,:,1) = 0.10;
    blueImage(:,:,2) = 0.45;
    blueImage(:,:,3) = 0.90;
    
    redImage = zeros([size(density2),3]);
    redImage(:,:,1) = 0.85;
    redImage(:,:,2) = 0.10;
    redImage(:,:,3) = 0.10;
    
    hold on
    
    h1 = imagesc(centers,centers,blueImage);
    h1.AlphaData = 0.5 * ring1;
    
    h2 = imagesc(centers,centers,redImage);
    h2.AlphaData = 0.5 * ring2;
    
    set(gca,'YDir','normal')
    axis tight
    box on

end



function selection = calculate_selction(choose_by, sim_object, data_object)

    selection = [];
    if strcmp(choose_by, 'none')
        selection = repmat(1:12,220,1);
    elseif strcmp(choose_by, 'SSE')
        for k = 1:220
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
            sses = [];
            for m = 1:12
                for z = 1:10
                    spikes{z+10} = find(sim_object.output(m,z,k,:))/10000;
                end
                data_times = [];
                for m = 1:10
                    data_times = [data_times;spikes{m}];
                end
                sim_times = [];
                for d = 11:20
                    sim_times = [sim_times;spikes{d}];
                end
                bin_edges = (0:100:29799)/10000;
                data_PSTH = histcounts(data_times,bin_edges);
                sim_PSTH = histcounts(sim_times,bin_edges);
                sses = [sses, sum((data_PSTH-sim_PSTH).^2)]; 
            end
            [val,idx] = min(sses);
            selection = [selection, idx];
        end

    elseif strcmp(choose_by, 'NCCC')

        for k = 1:220
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
            NCCCs = [];
            for m = 1:12
                for z = 1:10
                    spikes{z+10} = find(sim_object.output(m,z,k,:))/10000;
                end
                data_times = [];
                ris = [];
                bin_edges = (0:100:29799)/10000;
                for m = 1:10
                    data_times = [data_times;spikes{m}];
                    ris = [ris;histcounts(spikes{m},bin_edges)];
                end
                
                sim_times = [];
                
                for d = 11:20
                    sim_times = [sim_times;spikes{d}];
                    
                end
                
                data_PSTH = histcounts(data_times,bin_edges);
                sim_PSTH = histcounts(sim_times,bin_edges);
                
                cor_vals = zeros([10, 10]);
                for mm = 1:10
                   for zz = 1:10
                       corr_mat = corrcoef(ris(mm,:),ris(zz,:));
                       cor_vals(mm,zz) = corr_mat(2,1);
                   end
                end
                
                TTRC = mean(cor_vals(triu(true(size(cor_vals)), 1)), 'omitnan');
                
                pred_cor = zeros([10, 1]);
                for mm = 1:10
                    corr_mat = corr(ris(mm,:),sim_PSTH);
                    pred_cor(mm) = corr(ris(mm,:)', sim_PSTH','Rows', 'complete');
                end

                NCCCs = [NCCCs, (1/sqrt(TTRC))*mean(pred_cor)];

            end
            [val,idx] = max(NCCCs);
            selection = [selection, idx];
        end
    elseif strcmp(choose_by, 'SPIKE')
        for k = 1:220
            distances = [];
            for m = 1:12
                tuning = lower(char(string(data_object.all_data(k).tuning_type)));
                if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
                spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
                for z = 1:10
                    spikes{z+10} = find(sim_object.output(m,z,k,:))/10000;
                end
                spikes = cellfun(@transpose, spikes, 'UniformOutput', false)';
                STS = SpikeTrainSet(spikes,0,3);
                dist_mat = STS.SPIKEdistanceMatrix();
                distances = [distances,mean(mean(dist_mat(1:10,11:20)))];
            end
            [val,idx] = min(distances);
            selection = [selection, idx];
        end
    elseif strcmp(choose_by, 'CV')
        for k = 1:220
            tuning = lower(char(string(data_object.all_data(k).tuning_type)));
            if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
            spikes = data_object.all_data(k).ctrl_tar1_timestamps(:,focus);
            data_cv = calc_cv_data(spikes);
            cv_val = [];
            for m = 1:12 %For all batches
                isi_s = []; 
                for z  = 1:10 % For all trials
                    isi_s = [isi_s,squeeze(diff(find(sim_object.output(m,z,k,:)))/10000)'];
                end
                cv_val = [cv_val,std(isi_s)/mean(isi_s)];
            end
            [val,idx] = min((data_cv - cv_val).^2);
            selection = [selection, idx];
        end
    end

        
end

function re_val = calc_re_data(spikes)
    
    %Split half reliability with Spearman Brown correction
    % train1 = [];
    % train2 = [];
    % for k = 1:10
    %     if mod(k,2) == 0
    %         train1 = [train1;spikes{k}];
    %     else
    %         train2 = [train2;spikes{k}];
    %     end
    % end
    % 
    % bin_edges = (0:100:29799)/10000;
    % data_PSTH1 = histcounts(train1,bin_edges);
    % data_PSTH2 = histcounts(train2,bin_edges);
    % 
    % re_val_no_spear = corr(data_PSTH1',data_PSTH2');
    % 
    % re_val = 2*re_val_no_spear/(1 + re_val_no_spear);


    %Using UT of spike distance mat
    spikes = cellfun(@transpose, spikes, 'UniformOutput', false)';
    STS = SpikeTrainSet(spikes,0,3);
    dist_mat = STS.SPIKEdistanceMatrix();
    re_val = mean(dist_mat(triu(true(size(dist_mat)),1)),'omitnan');

end


function re_val = calc_re_sim(sim_object,k, selection)
    
    %Split Half
    % re_vals = [];
    % for m = 1:12
    %     spikes = {};
    %     for z = 1:10
    %         spikes{z} = find(sim_object.output(m,z,k,:))/10000;
    %     end
    %     train1 = [];
    %     train2 = [];
    %     for qq = 1:10
    %         if mod(qq,2) == 0
    %             train1 = [train1;spikes{qq}];
    %         else
    %             train2 = [train2;spikes{qq}];
    %         end
    %     end
    %     bin_edges = (0:100:29799)/10000;
    %     sim_PSTH1 = histcounts(train1,bin_edges);
    %     sim_PSTH2 = histcounts(train2,bin_edges);
    % 
    %     re_val_no_spear = corr(sim_PSTH1',sim_PSTH2');
    % 
    %     re_vals = [re_vals, 2*re_val_no_spear/(1 + re_val_no_spear)];
    % 
    % end
    % 
    % if nnz(size(selection) > 1) > 1
    %     re_val = re_vals(selection(k,:));
    % else
    %     re_val = re_vals(selection(k));
    % end


    %Using UT of spike distance mat
    re_vals = [];
    for m = 1:12
        spikes = {};
        for z = 1:10
            spikes{z} = find(sim_object.output(m,z,k,:))/10000;
        end

        spikes = cellfun(@transpose, spikes, 'UniformOutput', false);
        STS = SpikeTrainSet(spikes,0,3);
        dist_mat = STS.SPIKEdistanceMatrix();
        re_vals = [re_vals,mean(dist_mat(triu(true(size(dist_mat)),1)),'omitnan')];

    end

    if nnz(size(selection) > 1) > 1
        re_val = re_vals(selection(k,:));
    else
        re_val = re_vals(selection(k));
    end

end

function ls_val = calc_ls_data(spikes)
    data_times = [];
    for m = 1:10
        data_times = [data_times;spikes{m}];
    end
    bin_edges = (0:100:29799)/10000;
    data_PSTH = histcounts(data_times,bin_edges);
    N = length(data_PSTH);
    ls_val = (1-((((1/N)*sum(data_PSTH))^2)/(((1/N)*sum(data_PSTH.^2)))))/(1-(1/N));
end


function ls_val = calc_ls_sim(sim_object,k, selection)
    
    
    lss = [];
    for m = 1:12
        spikes = {};
        for z = 1:10
            spikes{z} = find(sim_object.output(m,z,k,:))/10000;
        end
        sim_times = [];
        for d = 1:10
            sim_times = [sim_times;spikes{d}];
        end
        bin_edges = (0:100:29799)/10000;
        sim_PSTH = histcounts(sim_times,bin_edges);
        N = length(sim_PSTH);
        lss = [lss, (1-((((1/N)*sum(sim_PSTH))^2)/(((1/N)*sum(sim_PSTH.^2)))))/(1-(1/N))]; 
    end


    if nnz(size(selection) > 1) > 1
        ls_val = lss(selection(k,:));
    else
        ls_val = lss(selection(k));
    end
    
end

function cv_val = calc_cv_data(spikes)
    isiarr = cell2mat(cellfun(@(x) diff(x(x >= 0 & x <= 2.9801)),spikes, UniformOutput=false));
    cv_val = std(isiarr)/mean(isiarr);
end

function cv_val = calc_cv_sim(sim_object,k, selection)
    cv_val = [];
    for m = 1:12 %For all batches
        isi_s = []; 
        for z  = 1:10 % For all trials
            isi_s = [isi_s,squeeze(diff(find(sim_object.output(m,z,k,:)))/10000)'];
        end
        cv_val = [cv_val,std(isi_s)/mean(isi_s)];
    end
    if nnz(size(selection) > 1) > 1
        cv_val = cv_val(selection(k,:));
    else
        cv_val = cv_val(selection(k));
    end
end


function Hz = calc_fr_sim(sim_object,k, selection)
    if nnz(size(selection) > 1) > 1
        Hz = (sum(sum(sim_object.output(selection(k,:),:,k,:),2),4)/10/2.9801)';
    else
        Hz = (sum(sum(sim_object.output(selection(k),:,k,:),2),4)/10/2.9801)';;
    end
    
end

function Hz = calc_fr_data(spikes)
    Hz = sum(cellfun(@(x) nnz(x >= 0 & x <= 2.9801), spikes)/2.9801)/10; %Get in terms of spikes per second
end

function spike_times = calc_spike_times(spike_object,data_location,data_cell,epoch)
    data_object = load(data_location);
    tuning = lower(char(string(data_object.all_data(data_cell).tuning_type)));
    if contains(tuning,'contra') focus = 1; elseif contains(tuning,'45') focus = 2; elseif contains(tuning,'center') focus = 3; elseif contains(tuning,'ipsi') focus = 4; end
    %Bring in the data spike times
    spike_times = data_object.all_data(data_cell).ctrl_tar1_timestamps(:,focus);
    %Append a simulations spike times
    for k = 1:10
        spike_times{k+10} = (find(squeeze(spike_object(epoch,k,:)))-1)/10000; %Putting this in real time. The - 1 accounts for find being matlab indexing and the /10000 converts it into seconds
    end
    %Transpose things so they work with cSPIKE
    for k = 1:20
        spike_times{k} = transpose(spike_times{k});
    end
    spike_times = transpose(spike_times);
end