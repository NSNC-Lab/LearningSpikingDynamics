figure;

choice = 1;

plot(squeeze(params.on_ron_gsyn(choice,:,:)/max(params.on_ron_gsyn(choice,:,:)))); hold on
plot(squeeze(params.off_ron_gsyn(choice,:,:)/max(params.off_ron_gsyn(choice,:,:))))
plot(squeeze(params.sonoff_ron_gsyn(choice,:,:)/max(params.sonoff_ron_gsyn(choice,:,:))))
plot(squeeze(params.on_sonoff_gsyn(choice,:,:)/max(params.on_sonoff_gsyn(choice,:,:))))
plot(squeeze(params.off_sonoff_gsyn(choice,:,:)/max(params.off_sonoff_gsyn(choice,:,:))))

plot(squeeze(params.output_ad(choice,:,:)/max(params.output_ad(choice,:,:))))
plot(squeeze(params.strf_gain(choice,:,:)/max(params.strf_gain(choice,:,:))))
plot(squeeze(params.strf_alpha(choice,:,:)/max(params.strf_alpha(choice,:,:))))

legend({'on->ron','off->ron','sonoff->ron','on->sonoff','off->sonoff','adaptation','strf gain','strf alpha'})


%%
figure;
for k = [1,7,133]
    for m = 1:12
        %plot(squeeze(params.strf_gain(m,k,:))); hold on
        %plot(squeeze(params.strf_alpha(m,k,:))); hold on
        
        %plot(squeeze(params.abs_ref(m,k,:))); hold on
        %plot(squeeze(params.rel_ref_c(m,k,:))); hold on
        %plot(squeeze(params.rel_ref_b(m,k,:))); hold on

        %plot(squeeze(params.abs_ref(m,k,:)/max(params.abs_ref(m,k,:)))); hold on
        %plot(squeeze(params.strf_alpha(m,k,:)/max(params.strf_alpha(m,k,:)))); hold on
        %plot(squeeze(params.strf_alpha(m,k,:))); hold on
        %plot(squeeze(params.on_ron_gsyn(m,k,:)/max(params.on_ron_gsyn(m,k,:)))); hold on
        %plot(squeeze(params.on_ron_gsyn(m,k,:))); hold on
        %plot(squeeze(params.off_ron_gsyn(m,k,:))); hold on
        %plot(squeeze(params.on_sonoff_gsyn(m,k,:))); hold on
        %plot(squeeze(params.sonoff_ron_gsyn(m,k,:))); hold on
        %plot(squeeze(params.off_sonoff_gsyn(m,k,:))); hold on
    end

end


