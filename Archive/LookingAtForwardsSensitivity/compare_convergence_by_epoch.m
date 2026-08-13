clear
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(here));
files={fullfile(root,'100_epoch_wide_eprop_cell_7.mat'),fullfile(root,'100_epoch_wide_forwards_sensitivity_cell_7.mat')}; labels={'eprop','forward'};
names={'strf_gain','strf_alpha','output_ad','abs_ref','rel_ref_a','rel_ref_b','rel_ref_c','on_ron_gsyn','off_ron_gsyn','sonoff_ron_gsyn','on_sonoff_gsyn','off_sonoff_gsyn'};
scale=reshape(single([.02 100 .005 2 1 10 1 repmat(.05,1,5)]),1,1,12); W=5; tol=1e-3;
for r=1:2
    v=who('-file',files{r}); v=v(ismember(v,{'params','checkpoint_params','adam','losses','sse_losses_all','cv_losses_all'})); D=load(files{r},v{:}); B=size(D.params.(names{1}),1); E=size(D.params.(names{1}),3); theta=zeros(B,E+1,12,'single');
    for p=1:12, theta(:,1:E,p)=reshape(D.params.(names{p}),B,E); theta(:,E+1,p)=reshape(D.checkpoint_params.(names{p}),B,1); end
    d=diff(theta,1,2)./scale; rms_step{r}=sqrt(mean(d.^2,3)); max_step{r}=max(abs(d),[],3); sse_step{r}=sqrt(mean(d(:,:,[1:3 8:12]).^2,3)); cv_step{r}=sqrt(mean(d(:,:,4:7).^2,3));
    rolling_step{r}=movmedian(rms_step{r},[W-1 0],2); rolling_diameter{r}=sqrt(mean(((movmax(theta,[W 0],2)-movmin(theta,[W 0],2))./scale).^2,3)); rolling_diameter{r}=rolling_diameter{r}(:,2:end); epoch{r}=(double(D.adam.t)-E+1:double(D.adam.t))';
    if isfield(D,'sse_losses_all'), L=reshape(D.sse_losses_all,B,E); else, L=reshape(D.losses,1,E); end
    q=2*abs(diff(L,1,2))./(abs(L(:,1:end-1))+abs(L(:,2:end))+eps('single')); relative_sse_change{r}=[nan(size(L,1),1,'single') q]; rolling_sse_change{r}=movmedian(relative_sse_change{r},[W-1 0],2,'omitnan');
    if isfield(D,'cv_losses_all'), L=reshape(D.cv_losses_all,B,E); q=2*abs(diff(L,1,2))./(abs(L(:,1:end-1))+abs(L(:,2:end))+eps('single')); relative_cv_change{r}=[nan(B,1,'single') q]; else, relative_cv_change{r}=nan(1,E,'single'); end
    rolling_cv_change{r}=movmedian(relative_cv_change{r},[W-1 0],2,'omitnan'); summary{r}=[epoch{r} median(rms_step{r},1,'omitnan')' prctile(rms_step{r},90,1)' prctile(rms_step{r},95,1)' median(max_step{r},1,'omitnan')' median(sse_step{r},1,'omitnan')' median(cv_step{r},1,'omitnan')' mean(rolling_step{r}<tol,1)' median(rolling_diameter{r},1,'omitnan')' median(rolling_sse_change{r},1,'omitnan')' median(rolling_cv_change{r},1,'omitnan')'];
    out=fullfile(here,[labels{r} '_convergence_by_epoch.csv']); writecell({'epoch','median_scaled_RMS_step','p90_scaled_RMS_step','p95_scaled_RMS_step','median_scaled_max_step','median_SSE_parameter_step','median_CV_parameter_step','fraction_rolling_step_below_tolerance','median_rolling_diameter','rolling_symmetric_SSE_change','rolling_symmetric_CV_change'},out); writematrix(summary{r},out,'WriteMode','append');
    bad_adam{r}=find(any(reshape(~isfinite(D.adam.m)|~isfinite(D.adam.v),B,[]),2)); fprintf('%s: updates %d-%d; nonfinite Adam batches: %s\n',labels{r},epoch{r}(1),epoch{r}(end),mat2str(bad_adam{r}'))
end
save(fullfile(here,'convergence_by_batch.mat'),'epoch','rms_step','max_step','sse_step','cv_step','rolling_step','rolling_diameter','relative_sse_change','rolling_sse_change','relative_cv_change','rolling_cv_change','summary','bad_adam','names','scale','W','tol','labels')
figure('Position',[100 100 850 750]); tiledlayout(3,1)
nexttile; for r=1:2, semilogy(epoch{r},median(rms_step{r},1,'omitnan'),'LineWidth',2,'DisplayName',[labels{r} ' median']); hold on; semilogy(epoch{r},prctile(rms_step{r},95,1),'--','DisplayName',[labels{r} ' p95']); end; ylabel('Scaled RMS step'); legend('Location','best'); grid on
nexttile; for r=1:2, plot(epoch{r},mean(rolling_step{r}<tol,1),'LineWidth',2,'DisplayName',labels{r}); hold on; end; ylabel(sprintf('Fraction below %g',tol)); ylim([0 1]); legend('Location','best'); grid on
nexttile; for r=1:2, semilogy(epoch{r},median(rolling_sse_change{r},1,'omitnan'),'LineWidth',2,'DisplayName',labels{r}); hold on; end; xlabel('Global optimizer update'); ylabel(sprintf('%d-update loss change',W)); legend('Location','best'); grid on
exportgraphics(gcf,fullfile(here,'convergence_by_epoch.pdf'),'ContentType','vector')
